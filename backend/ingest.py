import os
import time

# Nesne Yönelimli (OOP) olarak güncellenmiş modüllerimiz
from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine


def main():
    pdf_path = "test.pdf"

    print("=== VERİ YUTMA (INGESTION) SİSTEMİ BAŞLATILIYOR ===")

    if not os.path.exists(pdf_path):
        print(f"HATA: {pdf_path} dosyası bulunamadı!")
        return

    start_time = time.time()

    # 1. MOTORLARIN AYAĞA KALDIRILMASI (Instantiating Objects)
    # Bu satırlar çalıştığında parser hazır bekler, Nomic CPU'ya yerleşir.
    parser_engine = DocumentParser(chunk_size=450, chunk_overlap=150)
    vector_engine = VectorStoreEngine()

    # 2. PDF'i OKU VE PARÇALA
    chunks = parser_engine.parse(pdf_path)

    # 3. VEKTÖRLE VE KAYDET
    print(
        "Düğümler Nomic ile vektörlenip ChromaDB'ye yazılıyor. CPU hızına bağlı olarak sürebilir..."
    )
    vector_engine.add_nodes(chunks)

    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")
    print("Veritabanı güncellendi. Yeni OOP mimarisi kullanıma hazır.")


if __name__ == "__main__":
    main()
