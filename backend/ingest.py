import gc
import os
import time

from core.config import AppConfig  # YENİ: Merkezi Sinir Sistemini içe aktardık
from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine
from core.vlm_engine import VLMEngine


def main():
    # Artık yolları elle yazmıyoruz, statik pdf_path hariç her şeyi config'den alıyoruz.
    pdf_path = "test.pdf"

    print("=== VERİ YUTMA (INGESTION) SİSTEMİ BAŞLATILIYOR ===")

    if not os.path.exists(pdf_path):
        print(f"HATA: {pdf_path} dosyası bulunamadı!")
        return

    start_time = time.time()

    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 1 ---")
    try:
        # YENİ: Modeller Config'den çekiliyor
        vlm_engine = VLMEngine(
            model_path=AppConfig.VLM_MODEL_PATH, mmproj_path=AppConfig.VLM_MMPROJ_PATH
        )
    except Exception as e:
        print(f"[HATA] VLM Motoru başlatılamadı: {e}")
        return

    # YENİ: Chunk limitleri Config'den çekiliyor
    parser_engine = DocumentParser(
        chunk_size=AppConfig.CHUNK_SIZE, chunk_overlap=AppConfig.CHUNK_OVERLAP
    )

    chunks = parser_engine.parse(pdf_path, vlm_engine=vlm_engine)

    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 2 ---")
    vlm_engine.unload()
    del vlm_engine
    gc.collect()
    print("[SİSTEM] VRAM başarıyla boşaltıldı. Kutsal 8GB sınırı güvende.\n")

    vector_engine = VectorStoreEngine()
    print(
        "Düğümler Nomic/Jina ile vektörlenip ChromaDB'ye yazılıyor. CPU hızına bağlı olarak sürebilir..."
    )

    vector_engine.add_nodes(chunks, file_name=pdf_path)

    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")
    print("VLM analizleri metne gömüldü ve Veritabanı güncellendi.")


if __name__ == "__main__":
    main()
