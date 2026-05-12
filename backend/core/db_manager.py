import json
import os
import re

import chromadb
from core.ingestion_engine import IngestionEngine


class DBManager:
    """
    Veritabanı yönetiminin merkezi sınıfı.

    Tüm doküman ve koleksiyon işlemleri buradan geçer. App.py UI katmanı
    olarak kalır, iş mantığı bu sınıfta toplanır.

    Catalog (documents.json) hangi koleksiyonda hangi doküman var bilgisini
    tutar. ChromaDB ile senkron olmak zorunda — eklenen/silinen her şey
    ikisinde de güncellenmelidir.
    """

    DEFAULT_COLLECTION = "default"
    CATALOG_FILE = "documents.json"
    # ChromaDB kuralı: 3-512 karakter, alfanumerik + . _ -, başı/sonu alfanumerik
    COLLECTION_NAME_PATTERN = re.compile(
        r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,48}[a-zA-Z0-9]$"
    )

    def __init__(self, persist_dir: str = "./backend/data/database"):
        self.persist_dir = persist_dir
        self.catalog_path = os.path.join(persist_dir, self.CATALOG_FILE)
        self.active_collection: str = self.DEFAULT_COLLECTION
        self.sections_path = os.path.join(persist_dir, "sections.json")
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        # Dizin yoksa oluştur — ChromaDB de aynı dizine yazacak
        os.makedirs(self.persist_dir, exist_ok=True)

        # Catalog yoksa boş bir tane oluştur, default koleksiyon hep var olsun
        if not os.path.exists(self.catalog_path):
            initial = {"collections": {self.DEFAULT_COLLECTION: {"documents": {}}}}
            self._save_catalog(initial)
            print(f"[SİSTEM] Yeni catalog oluşturuldu: {self.catalog_path}")

        # Önceki oturumdan kalan orphan segment klasörlerini temizle
        self._cleanup_orphan_segments()

    # ── Catalog dosyası okuma/yazma ──────────────────────────────────────────

    def _load_catalog(self) -> dict:
        """Catalog'u diskten okur, dict olarak döndürür."""
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_catalog(self, catalog: dict) -> None:
        """Catalog'u diske yazar. Tüm yazma işlemleri buradan geçer."""
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

    # ── Koleksiyon yönetimi ──────────────────────────────────────────────────

    def list_collections(self) -> list[str]:
        """Catalog'daki tüm koleksiyon adlarını döndürür."""
        catalog = self._load_catalog()
        return list(catalog["collections"].keys())

    def create_collection(self, name: str) -> bool:
        """
        Yeni koleksiyon oluşturur. Catalog'a ekler ve ChromaDB'de de boş
        bir koleksiyon açar (ikisi senkron olsun).
        """
        name = name.strip()
        if not name:
            print("[HATA] Koleksiyon adı boş olamaz.")
            return False

        if not self.COLLECTION_NAME_PATTERN.match(name):
            print(
                "[HATA] Geçersiz koleksiyon adı. Kurallar:\n"
                "  - 3 ila 50 karakter\n"
                "  - Sadece harf, rakam, . _ -\n"
                "  - Başı ve sonu harf veya rakam olmalı"
            )
            return False

        catalog = self._load_catalog()
        if name in catalog["collections"]:
            print(f"[HATA] '{name}' adında bir koleksiyon zaten var.")
            return False

        self.chroma_client.get_or_create_collection(name)
        catalog["collections"][name] = {"documents": {}}
        self._save_catalog(catalog)
        print(f"[SİSTEM] Koleksiyon oluşturuldu: '{name}'")
        return True

    def delete_collection(self, name: str) -> bool:
        """
        Koleksiyonu siler: ChromaDB + sections.json + catalog. Default
        koleksiyon silinemez. Aktif koleksiyon siliniyorsa default'a düşer.
        """
        if name == self.DEFAULT_COLLECTION:
            print(f"[HATA] '{self.DEFAULT_COLLECTION}' koleksiyonu silinemez.")
            return False

        catalog = self._load_catalog()
        if name not in catalog["collections"]:
            print(f"[HATA] '{name}' adında koleksiyon yok.")
            return False

        try:
            self.chroma_client.delete_collection(name)
        except Exception as e:
            print(f"[UYARI] ChromaDB tarafında silme hatası: {e}")

        # SQLite kaydı silindi, mmap'leri serbest bırakmak için client'ı yenile
        self._reset_chroma_client()

        self._delete_sections_by_collection(name)

        del catalog["collections"][name]
        self._save_catalog(catalog)

        if self.active_collection == name:
            self.active_collection = self.DEFAULT_COLLECTION
            print(f"[SİSTEM] Aktif koleksiyon '{self.DEFAULT_COLLECTION}'a alındı.")

        self._vacuum_db()
        print(f"[SİSTEM] Koleksiyon silindi: '{name}'")
        print("[BİLGİ] Segment klasörleri programı yeniden başlattığında temizlenecek.")
        return True

    def set_active_collection(self, name: str) -> bool:
        """Aktif koleksiyonu değiştirir. Hedef koleksiyon var olmak zorunda."""
        catalog = self._load_catalog()
        if name not in catalog["collections"]:
            print(f"[HATA] '{name}' adında koleksiyon yok.")
            return False
        self.active_collection = name
        print(f"[SİSTEM] Aktif koleksiyon: '{name}'")
        return True

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _delete_sections_by_collection(self, collection_name: str) -> None:
        """sections.json'dan belirtilen koleksiyona ait section'ları siler."""
        if not os.path.exists(self.sections_path):
            return

        with open(self.sections_path, "r", encoding="utf-8") as f:
            sections = json.load(f)

        filtered = {
            sid: data
            for sid, data in sections.items()
            if data.get("metadata", {}).get("collection_name") != collection_name
        }

        with open(self.sections_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    # ── Doküman yönetimi ─────────────────────────────────────────────────────

    def add_documents(
        self,
        file_paths: list[str],
        collection: str | None = None,
        on_conflict: str = "ask",
    ) -> dict:
        """
        Çoklu PDF ekleme. IngestionEngine'i çağırır, catalog'u günceller.

        on_conflict:
            "ask"       → çağıran tarafa bırak (app.py kullanıcıya sorar)
            "overwrite" → mevcut dokümanı sil, yeniyi yaz
            "skip"      → mevcut dokümanı atla
        """

        collection = collection or self.active_collection
        catalog = self._load_catalog()

        if collection not in catalog["collections"]:
            print(f"[HATA] '{collection}' adında koleksiyon yok.")
            return {"success": [], "failed": [], "skipped": []}

        # ── Çakışma filtresi ──
        to_process: list[str] = []
        skipped: list[dict] = []

        for path in file_paths:
            file_name = os.path.basename(path)
            exists = file_name in catalog["collections"][collection]["documents"]

            if not exists:
                to_process.append(path)
                continue

            if on_conflict == "overwrite":
                # Eski kaydı sil, sonra yeniyi yaz
                self.delete_document(file_name, collection=collection)
                to_process.append(path)
            elif on_conflict == "skip":
                skipped.append({"file_name": file_name, "reason": "Zaten var"})
                print(f"[ATLA] '{file_name}' zaten var, atlanıyor.")
            else:  # "ask" — app.py burayı çağırmadan önce karar vermeli
                print(
                    f"[HATA] '{file_name}' zaten var. App katmanı on_conflict "
                    "kararını vermeden bu method çağrılmamalı."
                )
                skipped.append(
                    {"file_name": file_name, "reason": "Çakışma — karar verilmemiş"}
                )

        if not to_process:
            return {"success": [], "failed": [], "skipped": skipped}

        # ── IngestionEngine'i çalıştır ──
        engine = IngestionEngine(persist_dir=self.persist_dir)
        result = engine.run(file_paths=to_process, collection_name=collection)

        # ── Catalog güncelle ──
        catalog = self._load_catalog()  # yeniden oku (delete_document yazmış olabilir)
        for item in result["success"]:
            file_name = item.pop("file_name")
            catalog["collections"][collection]["documents"][file_name] = item
        self._save_catalog(catalog)

        return {
            "success": result["success"],
            "failed": result["failed"],
            "skipped": skipped,
        }

    def list_documents(self, collection: str | None = None) -> list[dict]:
        """
        Belirtilen koleksiyondaki tüm dokümanları döndürür.
        collection=None ise aktif koleksiyon kullanılır.

        Dönüş örneği:
            [
                {"file_name": "test1.pdf", "added_at": "...", "chunk_count": 42, ...},
                ...
            ]
        """
        collection = collection or self.active_collection
        catalog = self._load_catalog()

        if collection not in catalog["collections"]:
            print(f"[HATA] '{collection}' adında koleksiyon yok.")
            return []

        docs = catalog["collections"][collection]["documents"]
        return [{"file_name": name, **info} for name, info in docs.items()]

    def document_exists(self, file_name: str, collection: str | None = None) -> bool:
        """Doküman aktif/belirtilen koleksiyonda kayıtlı mı?"""
        collection = collection or self.active_collection
        catalog = self._load_catalog()
        if collection not in catalog["collections"]:
            return False
        return file_name in catalog["collections"][collection]["documents"]

    def delete_documents(
        self, file_names: list[str], collection: str | None = None
    ) -> dict:
        """
        Birden fazla dokümanı toplu siler. VACUUM ve segment temizliği
        en sonda bir kez çalışır.

        Dönüş: {"deleted": [...], "failed": [...]}
        """
        collection = collection or self.active_collection
        deleted: list[str] = []
        failed: list[dict] = []

        catalog = self._load_catalog()
        if collection not in catalog["collections"]:
            print(f"[HATA] '{collection}' adında koleksiyon yok.")
            return {"deleted": [], "failed": []}

        for file_name in file_names:
            if file_name not in catalog["collections"][collection]["documents"]:
                failed.append({"file_name": file_name, "reason": "Koleksiyonda yok"})
                continue

            # ChromaDB
            try:
                chroma_col = self.chroma_client.get_or_create_collection(collection)
                chroma_col.delete(where={"file_name": file_name})
            except Exception as e:
                failed.append({"file_name": file_name, "reason": f"ChromaDB: {e}"})
                continue

            # sections.json
            self._delete_sections_by_document(file_name, collection)

            # catalog
            del catalog["collections"][collection]["documents"][file_name]
            deleted.append(file_name)

        self._save_catalog(catalog)

        # Toplu temizlik
        if deleted:
            self._vacuum_db()
            print(f"[SİSTEM] {len(deleted)} doküman silindi.")

        return {"deleted": deleted, "failed": failed}

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _delete_sections_by_document(
        self, file_name: str, collection_name: str
    ) -> None:
        """
        sections.json'dan belirtilen doküman+koleksiyon kombinasyonuna ait
        section'ları siler.
        """
        if not os.path.exists(self.sections_path):
            return

        with open(self.sections_path, "r", encoding="utf-8") as f:
            sections = json.load(f)

        filtered = {
            sid: data
            for sid, data in sections.items()
            if not (
                data.get("metadata", {}).get("file_name") == file_name
                and data.get("metadata", {}).get("collection_name") == collection_name
            )
        }

        with open(self.sections_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    def _vacuum_db(self) -> None:
        """
        SQLite freelist'i temizler, dosyayı kompakt yapar.

        ChromaDB silme yapsa bile SQLite sayfaları "freelist"e atıyor,
        dosya boyutunu küçültmüyor. VACUUM bunu çözer. Hızlı operasyon
        (küçük dosyalarda ~100ms), her silme sonrası çağırmak güvenli.
        """
        import sqlite3

        sqlite_path = os.path.join(self.persist_dir, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return
        try:
            conn = sqlite3.connect(sqlite_path)
            conn.execute("VACUUM")
            conn.close()
        except Exception as e:
            print(f"[UYARI] VACUUM sırasında: {e}")

    def _cleanup_orphan_segments(self) -> None:
        """
        ChromaDB'nin SQLite'taki aktif segment ID'lerini alır, persist_dir
        içindeki UUID'li klasörleri tarar, listede olmayanları siler.

        ChromaDB delete_collection() segment klasörlerini diskten silmiyor —
        bu method onu telafi eder. Silme operasyonlarından sonra çağrılır.
        """
        import shutil
        import sqlite3

        sqlite_path = os.path.join(self.persist_dir, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return

        # Aktif segment ID'lerini SQLite'tan oku
        try:
            conn = sqlite3.connect(sqlite_path)
            cur = conn.cursor()
            cur.execute("SELECT id FROM segments")
            active_ids = {row[0] for row in cur.fetchall()}
            conn.close()
        except Exception as e:
            print(f"[UYARI] Segment ID'leri okunamadı: {e}")
            return

        # persist_dir altındaki UUID klasörlerini tara
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        removed = 0
        for entry in os.listdir(self.persist_dir):
            full_path = os.path.join(self.persist_dir, entry)
            if not os.path.isdir(full_path):
                continue
            if not uuid_pattern.match(entry):
                continue
            if entry in active_ids:
                continue
            # Orphan — sil
            try:
                shutil.rmtree(full_path)
                removed += 1
            except Exception as e:
                print(f"[UYARI] Orphan segment silinemedi ({entry}): {e}")

        if removed:
            print(f"[SİSTEM] {removed} orphan segment klasörü temizlendi.")

    def _reset_chroma_client(self) -> None:
        """
        ChromaDB client'ını yıkıp yeniden oluşturur.

        Windows'ta segment dosyaları mmap ile açık tutuluyor; client
        canlıyken silinmiş koleksiyonların dosyaları kilitli kalıyor.
        Reset, bütün mmap'leri serbest bırakır.
        """
        import gc

        del self.chroma_client
        gc.collect()
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
