import os
import time

from core.document_parser import DocumentParser
from core.vector_store import VectorStoreEngine


def main():
    pdf_path = "test.pdf"

    print("=== VERİ YUTMA (INGESTION) SİSTEMİ BAŞLATILIYOR ===")

    if not os.path.exists(pdf_path):
        print(f"HATA: {pdf_path} dosyası bulunamadı!")
        return

    start_time = time.time()

    # 1. MOTORLARIN AYAĞA KALDIRILMASI
    parser_engine = DocumentParser(chunk_size=450, chunk_overlap=150)
    vector_engine = VectorStoreEngine()

    # 2. PDF'i OKU VE PARÇALA
    # Not: DocumentParser şu an resimleri temp_images klasörüne çıkarıyor
    # ve metin içine ![](resim_yolu) şeklinde etiket bırakıyor. Bu altyapı hazır.
    chunks = parser_engine.parse(pdf_path)

    # 3. VEKTÖRLE VE KAYDET
    print(
        "Düğümler Nomic ile vektörlenip ChromaDB'ye yazılıyor. CPU hızına bağlı olarak sürebilir..."
    )

    # DÜZELTME: file_name parametresi eklendi. Artık çakışma ve zincirleme hata olmayacak.
    vector_engine.add_nodes(chunks, file_name=pdf_path)

    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")
    print("Veritabanı güncellendi. Yeni OOP mimarisi kullanıma hazır.")


if __name__ == "__main__":
    main()
