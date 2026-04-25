import os
import re

import pymupdf4llm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class DocumentParser:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        print("[SİSTEM] DocumentParser (Ayrıştırıcı) başlatılıyor...")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Not: Faz 2'de buradaki SentenceSplitter'ı kaldırıp DocStore mimarisine geçeceğiz.
        # Şimdilik sistemin hata vermemesi için yerinde bırakıyoruz.
        self.parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _clean_markdown(self, text: str) -> str:
        """PyMuPDF4LLM'in ürettiği Markdown metnindeki gürültüleri temizler."""
        text = re.sub(
            r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*<br>",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*", "", text
        )
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        stop_phrases = [
            "SAKARYA ÜNİVERSİTESİ",
            "BSM 101-BİLGİSAYAR MÜHENDİSLİĞİNE GİRİŞ",
            "BSM 101 BİLGİSAYAR MÜHENDİSLİĞİNE GİRİŞ",
            "BSM 101 – BİLGİSAYAR MÜHENDİSLİĞİNE GİRİŞ",
        ]
        for phrase in stop_phrases:
            text = re.compile(re.escape(phrase), re.IGNORECASE).sub("", text)

        text = re.sub(r"(?:\n[ \t\x0b\f\r\xa0]*){3,}", "\n\n", text)
        return text.strip()

    def _inject_vlm_analysis(self, text: str, vlm_engine) -> str:
        """
        Metin içindeki ![](resim_yolu) etiketlerini bulur, resmi VLM'e okutur
        ve yer tutucuyu <VLM_START> analiz <VLM_END> bloku ile değiştirir.
        """
        if vlm_engine is None:
            print("[UYARI] VLM Motoru sağlanmadı. Görseller metne dahil edilmeyecek.")
            return text

        # Markdown içindeki resim etiketlerini bulan Regex: ![] (yol)
        pattern = r"!\[\]\((.*?)\)"
        matches = list(re.finditer(pattern, text))

        if not matches:
            print("[SİSTEM] Dokümanda analiz edilecek görsel bulunamadı.")
            return text

        print(
            f"[SİSTEM] Dokümanda {len(matches)} adet görsel bulundu. VLM enjeksiyonu başlıyor..."
        )

        # Bulunan her bir resim etiketi için döngüye gir
        for match in matches:
            original_tag = match.group(0)  # Örn: ![](backend/data/temp_images/...)
            image_path = match.group(1)  # Örn: backend/data/temp_images/...

            # Resmi VLM'e gönder ve analizi al
            vlm_result = vlm_engine.extract_text(image_path)

            if vlm_result:
                # Ebeveyn-Çocuk mimarisi için kullanacağımız ID (Resmin dosya adı)
                img_id = os.path.basename(image_path)

                # Yeni değiştirilecek metin bloğu
                replacement = f"\n<VLM_START id='{img_id}'>\n{vlm_result}\n<VLM_END>\n"

                # Metnin içindeki ![](...) etiketini analizle değiştir
                text = text.replace(original_tag, replacement)
            else:
                # Eğer VLM boş dönerse etiketi sessizce sil (Çöp görseldir)
                text = text.replace(original_tag, "")

        print("[SİSTEM] VLM Enjeksiyonu başarıyla tamamlandı.")
        return text

    def parse(self, file_path: str, vlm_engine=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"HATA: Ayrıştırılacak belge bulunamadı -> {file_path}"
            )

        print(f"[{file_path}] ayrıştırıcıya (parser) alındı...")

        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        temp_img_dir = os.path.join("backend", "data", "temp_images", name_without_ext)
        os.makedirs(temp_img_dir, exist_ok=True)

        print(f"[SİSTEM] Görseller geçici olarak çıkarılıyor: {temp_img_dir}")

        # PyMuPDF4LLM ile Markdown'a çevir (resimler klasöre çıkar, yerlerine ![](yol) konur)
        md_text = pymupdf4llm.to_markdown(
            doc=file_path,
            write_images=True,
            image_path=temp_img_dir,
        )

        # 1. Metni gürültülerden temizle
        clean_text = self._clean_markdown(md_text)

        # 2. YENİ: VLM'i devreye sok ve resimlerin yerine analizleri göm!
        enriched_text = self._inject_vlm_analysis(clean_text, vlm_engine)

        doc = Document(text=enriched_text, metadata={"file_name": file_path})

        # Not: Şu an standart parçalayıcı aktif. Ebeveyn-Çocuk gelene kadar VLM metni bölünebilir.
        nodes = self.parser.get_nodes_from_documents([doc])

        print(
            f"Başarılı! Doküman semantik yapısı korunarak {len(nodes)} adet düğüme ayrıştırıldı."
        )
        return nodes


if __name__ == "__main__":
    print("Test için lütfen ingest.py dosyasını kullanınız.")
