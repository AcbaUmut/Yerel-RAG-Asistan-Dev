import os
import re
import time
from collections import Counter

import numpy as np
import pymupdf4llm
from core.config import AppConfig
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from PIL import Image


class DocumentParser:
    """
    PDF ayrıştırıcı.

    pymupdf4llm 1.27+ sürümünden itibaren pymupdf_layout paketi otomatik
    olarak kurulur ve import sırasında aktifleşir. Ekstra bir ayar gerekmez.
    Layout modu; başlık tespiti, tablo algılama ve okuma sırası yeniden
    yapılandırması için AI tabanlı bir Graph Neural Network kullanır.

    Sorumluluklar:
        pymupdf4llm  →  Metin, başlık, madde işaretleri, paragraflar ve
                        görsel referansları (![]()). dpi=300 ile görseller
                        yüksek çözünürlükte diske yazılır.

        PIL          →  Görsel filtresi. Dekoratif şeritler, logolar ve
                        küçük öğeleri VLM'e göndermeden eleyerek gereksiz
                        VLM çağrılarını önler. Diske yazılan görsel sayısını
                        değiştirmez; yalnızca VLM'e gidecek olanları belirler.

        VLM          →  Yalnızca filtreden geçen gerçek içerik görselleri
                        için çalışır.
    """

    def __init__(
        self,
        chunk_size: int = 1100,
        chunk_overlap: int = 200,
        hf_threshold: float = 0.6,
    ):
        print("[SİSTEM] DocumentParser başlatılıyor...")
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

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 1 — Tekrar Eden Üstbilgi / Altbilgi Temizleyici
    # ──────────────────────────────────────────────────────────────────────────

    def _remove_frequent_headers_footers(self, pages: list) -> str:
        """
        Sayfa listesindeki her sayfanın ilk ve son birkaç satırını inceler.
        Belirli bir eşiğin üzerinde tekrar eden satırları (sayfa numarası,
        üniversite adı, belge başlığı vb.) tüm sayfalardan siler ve
        temizlenmiş metni tek bir string olarak döndürür.
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
            candidate_lines.extend(lines[:3])
            candidate_lines.extend(lines[-3:])

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

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 2 — Markdown Temizleyici
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_markdown(self, text: str) -> str:
        """
        pymupdf4llm çıktısındaki gürültüleri temizler.
        """
        # pymupdf4llm görsel metin bloklarını temizle
        text = re.sub(
            r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*<br>",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"\*\*==> picture \[.*?\] intentionally omitted <==\*\*", "", text
        )
        # Tek başına duran sayfa numaralarını sil
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Üç veya daha fazla boş satırı ikiye indir
        text = re.sub(r"(?:\n[ \t\x0b\f\r\xa0]*){3,}", "\n\n", text)
        return text.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 3 — Dekoratif Görsel Filtresi (PIL)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_decorative_image(self, image_path: str) -> bool:
        """
        Bir görselin dekoratif/gürültü mü yoksa gerçek içerik mi olduğunu
        üç bağımsız kuralla belirler. Herhangi bir kural tetiklenirse
        görsel dekoratif kabul edilir ve VLM'e gönderilmez.

        Kural 1 — Boyut:
            Toplam piksel < 4000 veya en kısa kenar < 20px → simge/çizgi.

        Kural 2 — En-boy oranı:
            Oran > 20 (çok geniş şerit) veya < 0.05 (çok uzun dikey şerit)
            → dekoratif süs elemanı.
            Örnek: 1001x21 piksel mavi şerit → oran 47.7 → dekoratif.

        Kural 3 — Renk varyansı:
            Std sapma < 18 → neredeyse tek renkli → düz renkli arka plan.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            # Kural 1
            if w * h < 4000:
                return True
            if min(w, h) < 20:
                return True

            # Kural 2
            ratio = w / h
            if ratio > 20.0 or ratio < 0.05:
                return True

            # Kural 3 — büyük görsellerde örnekleme yaparak hız kazanıyoruz
            arr = np.array(img, dtype=np.float32)
            step_y = max(1, h // 60)
            step_x = max(1, w // 60)
            sampled = arr[::step_y, ::step_x]
            if np.std(sampled) < 18.0:
                return True

            return False

        except Exception:
            return True  # Okunamayan görsel → atla

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 4 — VLM Enjeksiyonu (Filtreli)
    # ──────────────────────────────────────────────────────────────────────────

    def _inject_vlm_analysis(self, text: str, vlm_engine) -> str:
        """
        Metin içindeki ![]() görsel referanslarını bulur.
        Her görsel önce _is_decorative_image filtresinden geçer:
            → Dekoratif ise: referans metinden silinir, VLM çağrılmaz.
            → Gerçek içerik ise: VLM'e gönderilir, analiz sonucu görselin
               orijinal konumuna yerleştirilir.

        Görsel analizinin görselin bulunduğu konuma yerleştirilmesi,
        ChromaDB chunk'larında görsel içeriğinin ilgili metinle birlikte
        bulunmasını sağlar.
        """
        if vlm_engine is None:
            print("[UYARI] VLM Motoru sağlanmadı. Görseller metne dahil edilmeyecek.")
            return text

        pattern = r"!\[.*?\]\((.*?)\)"
        matches = list(re.finditer(pattern, text))

        if not matches:
            print("[SİSTEM] Dokümanda analiz edilecek görsel bulunamadı.")
            return text

        # Tüm görselleri önceden filtrele
        decorative_paths = {
            m.group(1) for m in matches if self._is_decorative_image(m.group(1))
        }
        meaningful_count = len(matches) - len(decorative_paths)

        print(
            f"[SİSTEM] {len(matches)} görsel bulundu → "
            f"{len(decorative_paths)} dekoratif filtrelendi → "
            f"{meaningful_count} görsel VLM'e gönderilecek."
        )

        if meaningful_count == 0:
            for match in matches:
                text = text.replace(match.group(0), "")
            print("[SİSTEM] Tüm görseller dekoratif, VLM çalıştırılmadı.")
            return text

        start_time = time.time()
        for match in matches:
            original_tag = match.group(0)
            image_path = match.group(1)

            if image_path in decorative_paths:
                text = text.replace(original_tag, "")
                continue

            vlm_result = vlm_engine.extract_text(image_path)

            if vlm_result:
                img_id = os.path.basename(image_path)
                replacement = f"\n<VLM_START id='{img_id}'>\n{vlm_result}\n<VLM_END>\n"
                text = text.replace(original_tag, replacement)
            else:
                text = text.replace(original_tag, "")

        print(f"      [Toplam VLM İşlemi: {time.time() - start_time:.2f} sn]\n")
        print("[SİSTEM] VLM Enjeksiyonu başarıyla tamamlandı.")
        return text

    # ──────────────────────────────────────────────────────────────────────────
    # ANA METOD
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, file_path: str, vlm_engine=None):
        """
        Ayrıştırma akışı:

            Adım 1 — pymupdf4llm:
                Tüm doküman işlenir. Layout modu otomatik aktiftir.
                Metin, başlıklar, madde işaretleri, paragraflar ve görsel
                referansları (![]()) çıkarılır. Görseller dpi=300 ile
                yüksek çözünürlükte diske yazılır.

            Adım 2 — Üstbilgi/altbilgi temizliği:
                Tekrar eden sayfa numaraları, üniversite adları vb. silinir.

            Adım 3 — Markdown temizliği:
                pymupdf4llm kalıntıları, gereksiz boş satırlar temizlenir.

            Adım 4 — VLM enjeksiyonu:
                PIL filtresi dekoratif görselleri eler.
                Gerçek içerik görselleri VLM'e gönderilir.
                Analiz sonucu görselin orijinal konumuna yerleştirilir.

            Adım 5 — Chunking:
                SentenceSplitter ile semantik parçalara bölünür.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"HATA: Ayrıştırılacak belge bulunamadı → {file_path}"
            )

        print(f"[{file_path}] ayrıştırıcıya (parser) alındı...")

        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        temp_img_dir = os.path.join("backend", "data", "temp_images", name_without_ext)
        os.makedirs(temp_img_dir, exist_ok=True)

        # ── Adım 1: pymupdf4llm ───────────────────────────────────────────────
        print("[SİSTEM] pymupdf4llm (Layout modu) çalıştırılıyor...")
        md_pages = pymupdf4llm.to_markdown(
            doc=file_path,
            write_images=True,
            image_path=temp_img_dir,
            dpi=300,  # Yüksek çözünürlük — VLM için kritik
            page_chunks=True,
        )

        # ── Adım 2: Üstbilgi/altbilgi temizliği ──────────────────────────────
        joined_text = self._remove_frequent_headers_footers(md_pages)

        # ── Adım 3: Markdown temizliği ────────────────────────────────────────
        clean_text = self._clean_markdown(joined_text)

        # ── Adım 4: VLM enjeksiyonu ───────────────────────────────────────────
        enriched_text = self._inject_vlm_analysis(clean_text, vlm_engine)

        # ── Adım 5: Chunking ──────────────────────────────────────────────────
        doc = Document(text=enriched_text, metadata={"file_name": file_path})
        nodes = self.parser.get_nodes_from_documents([doc])

        print(
            f"Başarılı! Doküman semantik yapısı korunarak "
            f"{len(nodes)} adet düğüme ayrıştırıldı."
        )
        return nodes


if __name__ == "__main__":
    print("Test için lütfen ingest.py dosyasını kullanınız.")
