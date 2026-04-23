import os

import pymupdf4llm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class DocumentParser:
    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 150):
        """
        Doküman ayrıştırıcı motorunu başlatır.
        İleride 'Akıllı Yönlendirme' (Smart Routing) ve 'Ebeveyn-Çocuk' mimarisi
        doğrudan bu sınıfın metotları arasına inşa edilecektir.
        """
        print("[SİSTEM] DocumentParser (Ayrıştırıcı) başlatılıyor...")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Sınıf oluşturulduğunda parçalayıcı sadece bir kez RAM'e yerleşir.
        self.parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def parse(self, file_path: str):
        """
        Verilen PDF dosyasını okur, Markdown'a çevirir ve semantik düğümlere böler.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"HATA: Ayrıştırılacak belge bulunamadı -> {file_path}"
            )

        print(f"[{file_path}] ayrıştırıcıya (parser) alındı...")

        # 1. Aşama: Loader ve Markdown Dönüşümü (pymupdf4llm farkı)
        md_text = pymupdf4llm.to_markdown(file_path)

        # Metni LlamaIndex Document formatına sarıyoruz.
        # İleride Metadata Enjeksiyonu (Tarih, yazar vb.) tam olarak buraya eklenecek.
        doc = Document(text=md_text, metadata={"file_name": file_path})

        # 2. Aşama: Semantik Parçalama (Nodes)
        nodes = self.parser.get_nodes_from_documents([doc])

        print(
            f"Başarılı! Doküman semantik yapısı korunarak {len(nodes)} adet düğüme ayrıştırıldı."
        )
        return nodes


# --- Test Alanı ---
if __name__ == "__main__":
    test_dosyasi = "test.pdf"
    try:
        parser_engine = DocumentParser()
        uretilen_dugumler = parser_engine.parse(test_dosyasi)
        print(f"\nÖrnek Çıktı: {uretilen_dugumler[0].text[:100]}...")
    except Exception as e:
        print(f"HATA OLUŞTU: {e}")
