import os
import re
from collections import Counter

import pymupdf4llm
from core.config import AppConfig
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class DocumentParser:
    def __init__(
        self,
        chunk_size: int = 1100,
        chunk_overlap: int = 200,
        hf_threshold: float = 0.6,
    ):
        print("[SİSTEM] DocumentParser (Ayrıştırıcı) başlatılıyor...")
        self.chunk_size = (
            AppConfig.CHUNK_SIZE if AppConfig.CHUNK_SIZE is not None else chunk_size
        )
        self.chunk_overlap = (
            AppConfig.CHUNK_OVERLAP
            if AppConfig.CHUNK_OVERLAP is not None
            else chunk_overlap
        )
        self.hf_threshold = hf_threshold

        self.parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _remove_frequent_headers_footers(self, pages: list) -> str:
        """
        Sayfa sayfa ayrılmış metinlerdeki tekrar eden üstbilgi ve altbilgileri
        istatistiksel (frekans) olarak tespit eder ve temizler.
        """
        total_pages = len(pages)
        if total_pages <= 2:
            return "\n\n".join([p["text"] for p in pages])

        candidate_lines = []

        for page in pages:
            lines = page["text"].split("\n")
            lines = [line.strip() for line in lines if line.strip()]

            if not lines:
                continue

            head_candidates = lines[:3]
            tail_candidates = lines[-3:]

            candidate_lines.extend(head_candidates)
            candidate_lines.extend(tail_candidates)

        line_counts = Counter(candidate_lines)

        stop_lines = set()
        for line, count in line_counts.items():
            if (count / total_pages) >= self.hf_threshold:
                stop_lines.add(line)

        if stop_lines:
            print(
                f"[SİSTEM] Otonom Temizleyici şu tekrar eden satırları sildi: {stop_lines}"
            )

        cleaned_pages = []
        for page in pages:
            lines = page["text"].split("\n")
            kept_lines = [line for line in lines if line.strip() not in stop_lines]
            cleaned_pages.append("\n".join(kept_lines))

        return "\n\n".join(cleaned_pages)

    def _clean_markdown(self, text: str) -> str:
        """PyMuPDF4LLM'in ürettiği Markdown metnindeki genel gürültüleri temizler."""
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

        pattern = r"!\[.*?\]\((.*?)\)"
        matches = list(re.finditer(pattern, text))

        if not matches:
            print("[SİSTEM] Dokümanda analiz edilecek görsel bulunamadı.")
            return text

        print(
            f"[SİSTEM] Dokümanda {len(matches)} adet görsel bulundu. VLM enjeksiyonu başlıyor..."
        )

        for match in matches:
            original_tag = match.group(0)
            image_path = match.group(1)

            vlm_result = vlm_engine.extract_text(image_path)

            if vlm_result:
                img_id = os.path.basename(image_path)
                replacement = f"\n<VLM_START id='{img_id}'>\n{vlm_result}\n<VLM_END>\n"
                text = text.replace(original_tag, replacement)
            else:
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

        md_pages = pymupdf4llm.to_markdown(
            doc=file_path, write_images=True, image_path=temp_img_dir, page_chunks=True
        )

        joined_text = self._remove_frequent_headers_footers(md_pages)

        clean_text = self._clean_markdown(joined_text)

        enriched_text = self._inject_vlm_analysis(clean_text, vlm_engine)

        doc = Document(text=enriched_text, metadata={"file_name": file_path})

        nodes = self.parser.get_nodes_from_documents([doc])

        print(
            f"Başarılı! Doküman semantik yapısı korunarak {len(nodes)} adet düğüme ayrıştırıldı."
        )
        return nodes


if __name__ == "__main__":
    print("Test için lütfen ingest.py dosyasını kullanınız.")
