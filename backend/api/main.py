import json
import logging
import os
import shutil
import tempfile

from core.config import AppConfig
from core.db_manager import DBManager
from core.logger import setup_logging
from core.query_engine import QueryEngine
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Logger'i kur — app.py'deki ile aynı kurulum,
# setup_logging idempotent olduğu için tekrar çağrı sorun değil.
setup_logging()
log = logging.getLogger(__name__)

# FastAPI uygulaması — sunucunun "kalbi"
app = FastAPI(title="Yerel RAG Asistani API")

# DBManager tek bir tane oluşturulur, sunucu açık olduğu
# sürece yaşar. Terminal'deki App.__init__ içindeki self.db
# ne işe yarıyorsa burada da aynı görev.
db = DBManager()


@app.get("/collections")
def list_collections():
    """Tüm koleksiyonları ve hangisinin aktif olduğunu döner."""
    return {
        "active": db.active_collection,
        "all": db.list_collections(),
    }


@app.get("/documents")
def list_documents():
    """Aktif koleksiyondaki dokümanları döner."""
    return {
        "collection": db.active_collection,
        "documents": db.list_documents(),
    }


class CreateCollectionRequest(BaseModel):
    """Yeni koleksiyon oluştururken frontend'in yollayacağı veri şeması."""

    name: str


@app.post("/collections")
def create_collection(body: CreateCollectionRequest):
    """Yeni koleksiyon oluşturur."""
    success = db.create_collection(body.name)
    if not success:
        # DBManager False döndürdü — koleksiyon adı geçersiz veya zaten var.
        # 400 Bad Request: 'istek hatalı, sebebi şu' demek.
        raise HTTPException(
            status_code=400,
            detail=f"'{body.name}' oluşturulamadı. Geçersiz ad veya zaten var.",
        )
    return {"created": body.name}


@app.delete("/collections/{name}")
def delete_collection(name: str):
    """Koleksiyonu ve içindeki tüm dokümanları siler."""
    if name == db.DEFAULT_COLLECTION:
        raise HTTPException(
            status_code=400,
            detail=f"'{db.DEFAULT_COLLECTION}' koleksiyonu silinemez.",
        )

    success = db.delete_collection(name)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' adında koleksiyon bulunamadı.",
        )
    return {"deleted": name}


@app.post("/collections/{name}/activate")
def set_active_collection(name: str):
    """Aktif koleksiyonu değiştirir."""
    success = db.set_active_collection(name)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' adında koleksiyon bulunamadı.",
        )
    return {"active": db.active_collection}


class DeleteDocumentsRequest(BaseModel):
    """Silinecek doküman adlarının listesi."""

    file_names: list[str]


@app.delete("/documents")
def delete_documents(body: DeleteDocumentsRequest):
    """Aktif koleksiyondan bir veya birden çok dokümanı siler."""
    if not body.file_names:
        raise HTTPException(
            status_code=400,
            detail="Silinecek doküman listesi boş.",
        )

    result = db.delete_documents(body.file_names)

    # 'failed' boş değilse 207 dönmek REST geleneğinde "kısmi başarı"
    # anlamına gelir. Şu an basit tutuyoruz, sade 200 ile döndürüp
    # detayı body'de veriyoruz; frontend hem 'deleted' hem 'failed'
    # listesini görüp UI'da gösterebilir.
    return result


@app.post("/documents")
def add_documents(
    files: list[UploadFile] = File(...),
    decisions: str = "{}",
):
    """
    PDF yükler. Çakışan dosyalar için frontend, /documents/check'ten
    aldığı bilgiyle her dosyanın kararını söyler.

    decisions parametresi JSON string olarak gelir, örnek:
        {"test1.pdf": "overwrite", "test2.pdf": "skip"}
    Çakışmayan dosyaları decisions'a yazmaya gerek yok.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Dosya gönderilmedi.")

    # Karar dict'ini JSON'dan parse et
    try:
        decisions_dict = json.loads(decisions)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="decisions parametresi geçerli JSON değil.",
        )

    # Boyut limiti kontrolü — terminal'deki gibi
    too_big: list[dict] = []
    accepted: list[UploadFile] = []
    for file in files:
        # UploadFile'ın .size özelliği var ama Starlette sürümüne göre
        # her zaman dolu olmayabilir. Güvenli yol: stream'in sonuna gidip pozisyonu ölç.
        file.file.seek(0, 2)  # 2 = dosyanın sonu
        size_bytes = file.file.tell()
        file.file.seek(0)  # başa geri sar, sonra okunacak

        size_mb = size_bytes / (1024 * 1024)
        if size_mb > AppConfig.MAX_FILE_SIZE_MB:
            too_big.append(
                {
                    "file_name": file.filename,
                    "reason": f"Boyut limiti aşıldı ({size_mb:.1f} MB > {AppConfig.MAX_FILE_SIZE_MB} MB)",
                }
            )
        else:
            accepted.append(file)

    if not accepted:
        return {"success": [], "failed": too_big, "skipped": []}

    # Kabul edilenleri geçici dizine yaz
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        saved_paths = []
        for file in accepted:
            dest = os.path.join(tmp_dir, file.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(dest)

            # Karar verilmemiş çakışmalar için güvenli default: skip
            # DBManager dict.get(file_name, "ask") yapıyor; dict varsayılanını
            # "skip" yapamayız ama önce string olarak "skip" gönderip dict ile
            # override etmek de mümkün değil. En temizi: çakışan ama kararı
            # olmayan dosyalar için dict'e elle "skip" yazmak.
            if file.filename not in decisions_dict and db.document_exists(
                file.filename
            ):
                decisions_dict[file.filename] = "skip"

        result = db.add_documents(file_paths=saved_paths, on_conflict=decisions_dict)

        # Boyut yüzünden atlananları failed listesine ekle
        result["failed"].extend(too_big)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class CheckDocumentsRequest(BaseModel):
    """Yüklenmeden önce çakışma kontrolü için dosya adları."""

    file_names: list[str]


@app.post("/documents/check")
def check_documents(body: CheckDocumentsRequest):
    """
    Verilen dosya adlarından hangileri aktif koleksiyonda zaten var,
    hangileri yeni — onu söyler. Henüz hiçbir şey yüklenmez.
    """
    existing: list[str] = []
    new: list[str] = []
    for name in body.file_names:
        if db.document_exists(name):
            existing.append(name)
        else:
            new.append(name)
    return {"existing": existing, "new": new}


class QueryRequest(BaseModel):
    """Sorgu için gerekli bilgiler."""

    question: str
    file_name: str


@app.post("/query")
def query(body: QueryRequest):
    """
    Aktif koleksiyondaki belirli bir doküman üzerinde sorgu çalıştırır.
    Modeller yüklenip cevap üretildiği için uzun sürebilir.
    """
    # Boş soru kontrolü — Pydantic str doğrulamasını geçer ama anlamsız iş.
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    # Doküman aktif koleksiyonda var mı? Yoksa boşa model yükleme.
    if not db.document_exists(body.file_name):
        raise HTTPException(
            status_code=404,
            detail=f"'{body.file_name}' aktif koleksiyonda bulunamadı.",
        )

    engine = QueryEngine(collection_name=db.active_collection)
    answer = engine.run(question=body.question, file_name=body.file_name)
    return {"answer": answer}


@app.post("/query/stream")
def query_stream(body: QueryRequest):
    """
    Sorguyu çalıştırır, cevabı token token akıtır.
    Frontend ReadableStream ile parçaları okur ve ekrana yazar.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    if not db.document_exists(body.file_name):
        raise HTTPException(
            status_code=404,
            detail=f"'{body.file_name}' aktif koleksiyonda bulunamadı.",
        )

    engine = QueryEngine(collection_name=db.active_collection)
    return StreamingResponse(
        engine.run_stream(question=body.question, file_name=body.file_name),
        media_type="text/plain; charset=utf-8",
    )
