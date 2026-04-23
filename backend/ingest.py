import os
import time

# Kendi yazdığımız veri yutma modülleri
from core.document_parser import parse_document
from core.vector_store import add_to_vector_store


def main():
    pdf_path = "test.pdf"

    print("=== VERİ YUTMA (INGESTION) SİSTEMİ BAŞLATILIYOR ===")

    if not os.path.exists(pdf_path):
        print(f"HATA: {pdf_path} dosyası bulunamadı!")
        return

    start_time = time.time()

    # 1. PDF'i Oku ve Parçala (450/150 token ayarıyla)
    print(f"[{pdf_path}] okunuyor ve düğümlere ayrılıyor...")
    chunks = parse_document(pdf_path)

    # 2. Vektörle ve ChromaDB'ye Kaydet
    print(
        "Düğümler Nomic ile vektörlenip ChromaDB'ye yazılıyor. Bu işlem CPU hızına bağlı olarak sürebilir..."
    )
    add_to_vector_store(chunks)

    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")
    print(
        "Veritabanı güncellendi. Artık soru sormak için 'main.py' dosyasını çalıştırabilirsin."
    )


if __name__ == "__main__":
    main()
