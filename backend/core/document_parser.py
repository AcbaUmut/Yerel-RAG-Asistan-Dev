import os
import re
import time
from collections import Counter

import numpy as np
import pymupdf4llm
from core.config import AppConfig
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode  # ← YENİ
from PIL import Image

# ── Sabitler ──────────────────────────────────────────────────────────────────

# VLM bloğunu baştan sona eşleyen kalıp.
# re.split() ile kullanıldığında capturing group sayesinde
# ayraçların kendisi de sonuç listesine dahil olur.
VLM_BLOCK_PATTERN = re.compile(r"(<VLM_START\b[^>]*>.*?<VLM_END>)", re.DOTALL)

# Bu uzunluğun altındaki saf metin segmentleri (başlıklar, tek satırlar vb.)
# komşularıyla birleştirilir; tek başına node olmaz.
MIN_SEGMENT_LEN = 150


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
                        VLM çağrılarını önler.

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

    # ──────────────────────────────────────────────────────────────────────────
    # BÖLÜM 3 — Dekoratif Görsel Filtresi (PIL)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_decorative_image(self, image_path: str) -> bool:
        """
        Bir görselin dekoratif/gürültü mü yoksa gerçek içerik mi olduğunu
        üç bağımsız kuralla belirler.

        Kural 1 — Boyut:   Toplam piksel < 4000 veya en kısa kenar < 20px
        Kural 2 — En-boy:  Oran > 20 veya < 0.05
        Kural 3 — Renk:    Std sapma < 18 → neredeyse tek renkli
        """
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            if w * h < 4000:
                return True
            if min(w, h) < 20:
                return True

            ratio = w / h
            if ratio > 20.0 or ratio < 0.05:
                return True

            arr = np.array(img, dtype=np.float32)
            step_y = max(1, h // 60)
            step_x = max(1, w // 60)
            sampled = arr[::step_y, ::step_x]
            if np.std(sampled) < 18.0:
                return True

            return False

        except Exception:
            return True

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
        """
        if vlm_engine is None:
            print("[UYARI] VLM Motoru sağlanmadı. Görseller metne dahil edilmeyecek.")
            return text

        pattern = r"!\[.*?\]\((.*?)\)"
        matches = list(re.finditer(pattern, text))

        if not matches:
            print("[SİSTEM] Dokümanda analiz edilecek görsel bulunamadı.")
            return text

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
    # BÖLÜM 5 — VLM Farkındalıklı Hibrit Chunker  ← YENİ
    # ──────────────────────────────────────────────────────────────────────────

    def _chunking_with_vlm_awareness(self, enriched_text: str, file_path: str) -> list:
        """
        VLM bloklarını asla bölmeyen, başlık bağlamını koruyan hibrit chunk sistemi.

        Üç aşamalı çalışır:

        Aşama 1 — Tip etiketleme:
            Metin VLM_BLOCK_PATTERN ile parçalanır.
            Her parça "vlm" veya "text" olarak etiketlenir.

        Aşama 2 — Küçük segment birleştirme:
            MIN_SEGMENT_LEN altındaki metin segmentleri (başlıklar, tek
            satırlar) komşularıyla birleştirilir.
            Birleştirme önceliği:
                1. Sonraki segment metin ise → ona önek olarak ekle
                2. Önceki segment metin ise → ona sonek olarak ekle
                3. Her iki taraf da VLM ise → node olarak kalır,
                   başlık yine de son_baslik değişkenine kaydedilir

        Aşama 3 — Node üretimi:
            - Metin segmentleri → SentenceSplitter (overlap korunur)
            - VLM segmentleri → Atomik TextNode (asla bölünmez)
              Her VLM node'una [Bölüm: <başlık>] satırı eklenir.
              Embedding modeli görselin hangi başlık altında olduğunu görür.

        node_index:
            Her node'a sıralı bir tam sayı atanır.
            Retriever bu indeksi kullanarak komşu node'ları getirebilir.
        """
        # ── Aşama 1: Tip etiketleme ───────────────────────────────────────────
        raw_segments = VLM_BLOCK_PATTERN.split(enriched_text)
        typed: list[dict] = []
        for seg in raw_segments:
            if not seg.strip():
                continue
            is_vlm = bool(VLM_BLOCK_PATTERN.match(seg.strip()))
            typed.append({"type": "vlm" if is_vlm else "text", "content": seg.strip()})

        # ── Aşama 2: Küçük segment birleştirme ───────────────────────────────
        merged: list[dict] = []
        i = 0
        while i < len(typed):
            seg = typed[i]

            if seg["type"] == "text" and len(seg["content"]) < MIN_SEGMENT_LEN:
                # Önce sonraki metin segmentiyle birleştirmeyi dene
                if i + 1 < len(typed) and typed[i + 1]["type"] == "text":
                    typed[i + 1]["content"] = (
                        seg["content"] + "\n\n" + typed[i + 1]["content"]
                    )
                    i += 1
                    continue
                # Sonraki VLM veya yok; önceki metin varsa ona ekle
                elif merged and merged[-1]["type"] == "text":
                    merged[-1]["content"] += "\n\n" + seg["content"]
                    i += 1
                    continue
                # İki tarafı da VLM: node olarak kalır (başlık bilgisi kaybolmasın)

            merged.append(dict(seg))
            i += 1

        # ── Aşama 3: Node üretimi ─────────────────────────────────────────────
        base_metadata = {"file_name": file_path}
        nodes: list = []
        last_heading = ""  # Son görülen başlık metni
        last_text_tail = ""  # Bir önceki metin segmentinin ham içeriği

        for seg in merged:
            if seg["type"] == "vlm":
                # Başlık bağlamını VLM metnine göm.
                # Embedding modeli artık VLM içeriğiyle birlikte başlığı da görür.
                prefix = f"[Bölüm: {last_heading}]\n" if last_heading else ""

                node = TextNode(
                    text=prefix + seg["content"],
                    metadata={
                        **base_metadata,
                        "node_type": "vlm",
                        "context_prefix": last_text_tail[-300:].strip(),
                        "node_index": len(nodes),
                    },
                )
                nodes.append(node)

            else:  # text
                # Bu segmentteki başlıkları tara, son başlığı güncelle.
                # Regex: # ile başlayan satırları eşler, ** bold marker'larını temizler.
                heading_matches = re.findall(
                    r"^#{1,6}\s+(.+?)$", seg["content"], re.MULTILINE
                )
                if heading_matches:
                    last_heading = (
                        heading_matches[-1]
                        .strip()
                        .replace("**", "")
                        .replace("*", "")
                        .strip()
                    )

                doc = Document(
                    text=seg["content"],
                    metadata={**base_metadata, "node_type": "text"},
                )
                text_nodes = self.parser.get_nodes_from_documents([doc])

                # Her text node'una sıralı index ata
                for tn in text_nodes:
                    tn.metadata["node_index"] = len(nodes)
                    nodes.append(tn)

                last_text_tail = seg["content"]

        vlm_count = sum(1 for n in nodes if n.metadata.get("node_type") == "vlm")
        text_count = len(nodes) - vlm_count
        print(
            f"[SİSTEM] Hibrit chunker tamamlandı: "
            f"{text_count} metin node + {vlm_count} VLM node = {len(nodes)} toplam"
        )
        return nodes

    # ──────────────────────────────────────────────────────────────────────────
    # ANA METOD
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, file_path: str, vlm_engine=None):
        """
        Ayrıştırma akışı:

            Adım 1 — pymupdf4llm:  Metin + görsel referansları çıkar
            Adım 2 — Üstbilgi/altbilgi temizliği
            Adım 3 — Markdown temizliği
            Adım 4 — VLM enjeksiyonu
            Adım 5 — VLM farkındalıklı hibrit chunking
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

        # ── Adım 1 ───────────────────────────────────────────────────────────
        print("[SİSTEM] pymupdf4llm (Layout modu) çalıştırılıyor...")
        md_pages = pymupdf4llm.to_markdown(
            doc=file_path,
            write_images=True,
            image_path=temp_img_dir,
            dpi=300,
            page_chunks=True,
        )

        # ── Adım 2 ───────────────────────────────────────────────────────────
        joined_text = self._remove_frequent_headers_footers(md_pages)

        # ── Adım 3 ───────────────────────────────────────────────────────────
        clean_text = self._clean_markdown(joined_text)

        # ── Adım 4 ───────────────────────────────────────────────────────────
        enriched_text = self._inject_vlm_analysis(clean_text, vlm_engine)

        # ── Adım 5 ───────────────────────────────────────────────────────────
        nodes = self._chunking_with_vlm_awareness(enriched_text, file_path)

        print(f"Başarılı! Doküman {len(nodes)} adet düğüme ayrıştırıldı.")
        return nodes


if __name__ == "__main__":
    print("Test için lütfen ingest.py dosyasını kullanınız.")
