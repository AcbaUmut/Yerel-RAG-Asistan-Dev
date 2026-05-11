import gc
import os
import time
from datetime import datetime

from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine
from core.vlm_engine import VLMEngine


class IngestionEngine:
    """
    Çoklu PDF'i sırayla işleyen ingestion orkestrasyonu.

    Donanım orkestrasyonu (8GB VRAM kısıtı):
        Faz 1 — VLM (GPU) yükle → tüm PDF'leri parse et → VLM unload
        Faz 2 — Jina (GPU) yükle → tüm chunk'ları ChromaDB'ye yaz → unload

    Modeller asla aynı anda VRAM'de durmaz. Tek dosya senaryosu da bu
    akışın özel hali — liste uzunluğu 1.

    Hata politikası (D1):
        Bir dosya hata verirse atlanır, diğerlerine devam edilir.
        Sonunda başarılı/başarısız özet raporu döndürülür.
    """

    def __init__(self, persist_dir: str = "./backend/data/database"):
        self.persist_dir = persist_dir

    def run(
        self,
        file_paths: list[str],
        collection_name: str = "default",
    ) -> dict:
        """
        Dönüş:
            {
                "success": [{"file_name": "...", "chunk_count": N, "section_count": M}, ...],
                "failed":  [{"file_name": "...", "reason": "..."}, ...]
            }
        """
        if not file_paths:
            print("[UYARI] İşlenecek dosya yok.")
            return {"success": [], "failed": []}

        print(f"\n=== INGESTION BAŞLATILDI — {len(file_paths)} dosya ===")
        start_time = time.time()

        # Her dosya için: file_path → parse sonucu (chunks) veya hata
        parsed: list[dict] = []  # {"file_path", "file_name", "chunks"}
        failed: list[dict] = []

        # ── Faz 1: VLM ile parse ─────────────────────────────────────────────
        print("\n--- FAZ 1: VLM YÜKLENİYOR ---")
        vlm_engine = None
        try:
            vlm_engine = VLMEngine()
        except Exception as e:
            print(f"[HATA] VLM yüklenemedi: {e}")
            # VLM olmadan da parse edilebilir, görseller atlanır
            # vlm_engine None kalır, parser bunu zaten handle ediyor

        parser = DocumentParser()

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            print(f"\n[{file_name}] parse ediliyor...")

            if not os.path.exists(file_path):
                failed.append({"file_name": file_name, "reason": "Dosya bulunamadı"})
                print(f"[HATA] Dosya bulunamadı: {file_path}")
                continue

            try:
                chunks = parser.parse(file_path=file_path, vlm_engine=vlm_engine)
                parsed.append(
                    {"file_path": file_path, "file_name": file_name, "chunks": chunks}
                )
            except Exception as e:
                failed.append({"file_name": file_name, "reason": f"Parse hatası: {e}"})
                print(f"[HATA] {file_name} parse edilemedi: {e}")

        # VLM unload — VRAM serbest
        if vlm_engine is not None:
            print("\n--- FAZ 1 BİTTİ — VLM TAHLİYE EDİLİYOR ---")
            vlm_engine.unload()
            del vlm_engine
        gc.collect()
        print("[SİSTEM] VRAM boşaltıldı.\n")

        if not parsed:
            print("[UYARI] Hiçbir dosya başarıyla parse edilemedi.")
            return {"success": [], "failed": failed}

        # ── Faz 2: Embedding + ChromaDB yazımı ───────────────────────────────
        print("--- FAZ 2: EMBEDDING (JINA) YÜKLENİYOR ---")
        vector_engine = VectorStoreEngine(
            persist_dir=self.persist_dir,
            collection_name=collection_name,
        )

        success: list[dict] = []

        for item in parsed:
            file_name = item["file_name"]
            chunks = item["chunks"]

            try:
                # Her chunk'a collection_name metadata'sı ekle (silme için kritik)
                for node in chunks:
                    node.metadata["collection_name"] = collection_name

                child_count, section_count = vector_engine.add_nodes(
                    nodes=chunks, file_name=file_name
                )
                success.append(
                    {
                        "file_name": file_name,
                        "chunk_count": child_count,
                        "section_count": section_count,
                        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception as e:
                failed.append({"file_name": file_name, "reason": f"Yazma hatası: {e}"})
                print(f"[HATA] {file_name} ChromaDB'ye yazılamadı: {e}")

        # ── Faz 2 sonu: embedding modelini VRAM'den at ──
        print("\n--- FAZ 2 BİTTİ — EMBEDDING MODELİ TAHLİYE EDİLİYOR ---")
        vector_engine.unload()
        del vector_engine
        gc.collect()

        # ── Özet rapor ───────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print(f"\n=== INGESTION TAMAMLANDI ({elapsed:.1f} sn) ===")
        print(f"  Başarılı: {len(success)}")
        print(f"  Başarısız: {len(failed)}")
        for f in failed:
            print(f"    - {f['file_name']}: {f['reason']}")

        return {"success": success, "failed": failed}
