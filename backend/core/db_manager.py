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
        r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$"
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
                "  - 3 ila 512 karakter\n"
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

        self._delete_sections_by_collection(name)

        del catalog["collections"][name]
        self._save_catalog(catalog)

        if self.active_collection == name:
            self.active_collection = self.DEFAULT_COLLECTION
            print(f"[SİSTEM] Aktif koleksiyon '{self.DEFAULT_COLLECTION}'a alındı.")

        self._vacuum_db()
        print(f"[SİSTEM] Koleksiyon silindi: '{name}'")
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

    def delete_document(self, file_name: str, collection: str | None = None) -> bool:
        """
        Dokümanı siler: ChromaDB + sections.json + catalog.
        """
        collection = collection or self.active_collection
        catalog = self._load_catalog()

        if collection not in catalog["collections"]:
            print(f"[HATA] '{collection}' adında koleksiyon yok.")
            return False

        if file_name not in catalog["collections"][collection]["documents"]:
            print(f"[HATA] '{file_name}' bu koleksiyonda yok.")
            return False

        # ── ChromaDB'den sil ──
        try:
            chroma_col = self.chroma_client.get_or_create_collection(collection)
            chroma_col.delete(where={"file_name": file_name})
        except Exception as e:
            print(f"[UYARI] ChromaDB tarafında silme hatası: {e}")

        # ── sections.json'dan sil ──
        self._delete_sections_by_document(file_name, collection)

        # ── Catalog'dan sil ──
        del catalog["collections"][collection]["documents"][file_name]
        self._save_catalog(catalog)

        self._vacuum_db()
        print(f"[SİSTEM] Doküman silindi: '{file_name}' (koleksiyon: {collection})")
        return True

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
