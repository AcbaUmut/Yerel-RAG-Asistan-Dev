import gc
import os
import time

from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine
from core.vlm_engine import VLMEngine


def main():

    start_time = time.time()
    pdf_path = "test.pdf"

    print("=== VERİ YUTMA (INGESTION) SİSTEMİ BAŞLATILIYOR ===")

    if not os.path.exists(pdf_path):
        print(f"HATA: {pdf_path} dosyası bulunamadı!")
        return

    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 1 ---")
    try:
        vlm_engine = VLMEngine()
    except Exception as e:
        print(f"[HATA] VLM Motoru başlatılamadı: {e}")
        return

    parser_engine = DocumentParser()

    chunks = parser_engine.parse(file_path=pdf_path, vlm_engine=vlm_engine)

    print(
        f"\n=== DOSYA OKUNDU! (Toplam Süre: {time.time() - start_time:.2f} sn) ===\n\n"
    )

    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 2 ---")
    vlm_engine.unload()
    del vlm_engine
    gc.collect()
    print("[SİSTEM] VRAM başarıyla boşaltıldı. Kutsal 8GB sınırı güvende.\n")

    vector_engine = VectorStoreEngine()
    print(
        "Düğümler Jina ile vektörlenip ChromaDB'ye yazılıyor. CPU hızına bağlı olarak sürebilir..."
    )

    vector_engine.add_nodes(nodes=chunks, file_name=pdf_path)

    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")
    print("VLM analizleri metne gömüldü ve Veritabanı güncellendi.")


if __name__ == "__main__":
    main()
