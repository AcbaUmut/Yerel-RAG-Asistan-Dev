import os
import re

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

    def _clean_markdown(self, text: str) -> str:
        """
        PyMuPDF4LLM'in ürettiği Markdown metnindeki gürültüleri temizler.
        """
        # 1. Çöp OCR Katliamı
        text = re.sub(
            r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*<br>",
            "",
            text,
            flags=re.DOTALL,
        )

        # 2. Görsel Etiketi Katliamı
        text = re.sub(
            r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*", "", text
        )

        # 3. Sayfa Numarası Uçurucusu
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # 4. Altbilgi / Üstbilgi Zehirlenmesini Giderme
        # Senin PDF'indeki "BSM 101-BİLGİSAYAR MÜHENDİSLİĞİNE GİRİŞ" aradaki tire (-) ile yazılmış, onu da ekledim.
        stop_phrases = [
            "SAKARYA ÜNİVERSİTESİ",
            "BSM 101-BİLGİSAYAR MÜHENDİSLİĞİNE GİRİŞ",
            "BSM 101 BİLGİSAYAR MÜHENDİSLİĞİNE GİRİŞ",
            "BSM 101 – BİLGİSAYAR MÜHENDİSLİĞİNE GİRİŞ",
        ]
        for phrase in stop_phrases:
            text = re.compile(re.escape(phrase), re.IGNORECASE).sub("", text)

        # 5. Boşluk Daraltma (Kozmetik Temizlik - GÜNCELLENDİ)
        # Sadece \n'leri değil, arada kalmış gizli boşlukları (space, tab vb.) da kapsayacak şekilde
        # ardışık boşlukları standart paragraf aralığına (çift satır) indirger.
        text = re.sub(r"(?:\n[ \t\x0b\f\r\xa0]*){3,}", "\n\n", text)

        return text.strip()

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

        # --- YENİ EKLENEN TEMİZLİK AŞAMASI ---
        # Metni LlamaIndex'e vermeden hemen önce kendi süzgecimizden geçiriyoruz.
        clean_text = self._clean_markdown(md_text)

        # Metni LlamaIndex Document formatına sarıyoruz. (Senin eklediğin metadata burada güvende)
        doc = Document(text=clean_text, metadata={"file_name": file_path})

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
        print(f"\nÖrnek Çıktı: {uretilen_dugumler[0].text[:500]}...")
    except Exception as e:
        print(f"HATA OLUŞTU: {e}")
