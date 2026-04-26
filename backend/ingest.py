import gc
import os
import time

from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine
from core.vlm_engine import VLMEngine


def main():
    # PDF yolu şimdilik test amaçlı sabit kalıyor
    pdf_path = "test.pdf"

    print("=== VERİ YUTMA (INGESTION) SİSTEMİ BAŞLATILIYOR ===")

    if not os.path.exists(pdf_path):
        print(f"HATA: {pdf_path} dosyası bulunamadı!")
        return

    start_time = time.time()

    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 1 ---")
    try:
        # DÜZELTME: VLMEngine artık parametresiz çalışıyor, yolları Config'den alıyor!
        vlm_engine = VLMEngine()
    except Exception as e:
        print(f"[HATA] VLM Motoru başlatılamadı: {e}")
        return

    # Chunk limitleri Config'den çekiliyor, varsayılanlar üzerine yazılıyor
    parser_engine = DocumentParser()

    chunks = parser_engine.parse(file_path=pdf_path, vlm_engine=vlm_engine)

    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 2 ---")
    vlm_engine.unload()
    del vlm_engine
    gc.collect()
    print("[SİSTEM] VRAM başarıyla boşaltıldı. Kutsal 8GB sınırı güvende.\n")

    # VectorStoreEngine de doğrudan Config'den besleniyor
    vector_engine = VectorStoreEngine()
    print(
        "Düğümler Nomic/Jina ile vektörlenip ChromaDB'ye yazılıyor. CPU hızına bağlı olarak sürebilir..."
    )

    vector_engine.add_nodes(nodes=chunks, file_name=pdf_path)

    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")
    print("VLM analizleri metne gömüldü ve Veritabanı güncellendi.")


if __name__ == "__main__":
    main()
