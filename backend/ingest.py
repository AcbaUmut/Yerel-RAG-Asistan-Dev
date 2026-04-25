import gc  # RAM ve VRAM Çöp toplayıcı
import os
import time

# torch importunu şimdilik atlıyoruz, llama_cpp kendi bellek yönetimini yapıyor.
from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine
from core.vlm_engine import VLMEngine


def main():
    pdf_path = "test.pdf"
    vlm_model_path = "./backend/models/ZwZ-4B-Q4_K_M.gguf"
    mmproj_path = "./backend/models/mmproj-ZwZ-4B-F16.gguf"

    print("=== VERİ YUTMA (INGESTION) SİSTEMİ BAŞLATILIYOR ===")

    if not os.path.exists(pdf_path):
        print(f"HATA: {pdf_path} dosyası bulunamadı!")
        return

    start_time = time.time()

    # 1. VLM MOTORUNUN AYAĞA KALDIRILMASI (VRAM İŞGALİ BAŞLAR)
    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 1 ---")
    try:
        vlm_engine = VLMEngine(model_path=vlm_model_path, mmproj_path=mmproj_path)
    except Exception as e:
        print(f"[HATA] VLM Motoru başlatılamadı: {e}")
        return

    # 2. PDF'i OKU, VLM'İ KULLAN VE PARÇALA
    parser_engine = DocumentParser(chunk_size=450, chunk_overlap=150)

    # VLM motorunu parser'a veriyoruz. Resimleri bulup analiz edip metne gömecek.
    chunks = parser_engine.parse(pdf_path, vlm_engine=vlm_engine)

    # 3. VRAM ÇÖP TOPLAMA (GARBAGE COLLECTION) - ÇOK KRİTİK!
    # VLM'in işi bitti. Onu acımasızca hafızadan siliyoruz ki 8GB sınırımız korunabilsin.
    print("\n--- DONANIM ORKESTRASYONU: AŞAMA 2 ---")
    vlm_engine.unload()
    del vlm_engine
    gc.collect()  # Python çöp toplayıcısını zorla çalıştır
    print("[SİSTEM] VRAM başarıyla boşaltıldı. Kutsal 8GB sınırı güvende.\n")

    # 4. VEKTÖRLE VE KAYDET (CPU İŞLEMİ)
    # Artık VRAM boş. Nomic CPU'da rahatça çalışabilir.
    vector_engine = VectorStoreEngine()
    print(
        "Düğümler Nomic ile vektörlenip ChromaDB'ye yazılıyor. CPU hızına bağlı olarak sürebilir..."
    )

    vector_engine.add_nodes(chunks, file_name=pdf_path)

    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")
    print("VLM analizleri metne gömüldü ve Veritabanı güncellendi.")


if __name__ == "__main__":
    main()
