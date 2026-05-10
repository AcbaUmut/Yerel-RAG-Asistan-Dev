"""
VLM Engine — ZwZ-4B (Qwen3VL tabanlı)

Tek geçişli mimari — kısa Türkçe evrensel prompt (~130 token).

Motor optimizasyonları:
  • n_batch=2048: vision token prefill hızlandırması.
  • n_ctx=6144: ≤1280 vision + ~130 prompt + 1536 response için güvenli.
  • flash_attn_type=True, swa_full=True: Qwen3VL uyumlu dikkat mekanizması.
  • type_k=8, type_v=8: Q8_0 KV cache.
  • PIL resize max 1.31MP: vision token sayısını ≤1280 ile sınırlar.
  • repeat_penalty=1.08: tekrarı önler, tablo yapısını bozmaz.

Prompt düzeltmeleri:
  • SEMA_GRAFIK: "GENEL AMAÇ:" / "YAPI VE AKIŞ:" bölüm başlıkları (echo önleme).
  • GRAFİK/CHART: ayrı kategori rehberi (yanlış sınıflandırma önleme).
  • FOTOĞRAF/HARİTA: hayvan türü ve bölge tespiti rehberi.
"""

import base64
import io
import os
import time

from core.config import AppConfig
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────

_PATCH_SIZE: int = 32

# Qwen3VL patch: 32×32 px → 1 vision token.
# 1280 × (32²) = 1 310 720 px → ≤1280 vision token.
# Detaylı şemalarda okunabilirliği korurken bağlam taşmasını önler.
_VLM_MAX_PIXELS: int = 1_310_720

_VALID_CATEGORIES = frozenset({"DÜZ_METİN", "TABLO", "SEMA_GRAFIK", "GORSEL_BETİMLEME"})


class VLMEngine:
    """
    Tek Geçişli VLM Motoru.

    Sınıflandırma ayrı bir aşama değildir; model görseli görüp
    doğal olarak uygun formatta yazar. Çıktıdaki ilk satırdan
    ("TÜR: X") kategori parse edilir.
    """

    def __init__(self) -> None:
        self.model_path = f"./backend/models/{AppConfig.VLM_MODEL_NAME}"
        self.mmproj_path = f"./backend/models/{AppConfig.VLM_MMPROJ_NAME}"

        if not os.path.exists(self.model_path) or not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(
                "[HATA] VLM veya mmproj model dosyaları bulunamadı!\n"
                f"  model : {self.model_path}\n"
                f"  mmproj: {self.mmproj_path}"
            )

        print("[SİSTEM] VLM Motoru (ZwZ-4B) VRAM'e yükleniyor...")

        self.chat_handler = Qwen3VLChatHandler(clip_model_path=self.mmproj_path)

        self.llm = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,
            n_ctx=(AppConfig.VLM_N_CTX if AppConfig.VLM_N_CTX is not None else 6144),
            n_gpu_layers=-1,
            n_batch=2048,
            flash_attn_type=True,
            swa_full=True,
            type_k=8,
            type_v=8,
            verbose=False,
        )

        print("[SİSTEM] VLM Motoru başarıyla ayağa kalktı.")

    # ──────────────────────────────────────────────────────────────────────────
    # Görsel hazırlama
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_image(self, file_path: str) -> str:
        """
        Görseli okur, gerekirse max_pixels sınırına küçültür,
        base64 data-URI döndürür.
        """
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if w * h > _VLM_MAX_PIXELS:
                scale = (_VLM_MAX_PIXELS / (w * h)) ** 0.5
                new_w = max(_PATCH_SIZE, (int(w * scale) // _PATCH_SIZE) * _PATCH_SIZE)
                new_h = max(_PATCH_SIZE, (int(h * scale) // _PATCH_SIZE) * _PATCH_SIZE)
                img = img.resize((new_w, new_h), Image.LANCZOS)

                old_tok = (w * h) // (_PATCH_SIZE**2)
                new_tok = (new_w * new_h) // (_PATCH_SIZE**2)
                print(
                    f"      Boyutlandırıldı: {w}×{h} → {new_w}×{new_h} "
                    f"(~{old_tok} → ~{new_tok} vision token)"
                )
            else:
                tok = (w * h) // (_PATCH_SIZE**2)
                print(f"      Boyut: {w}×{h} (~{tok} vision token)")

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{b64}"

    # ──────────────────────────────────────────────────────────────────────────
    # Evrensel prompt
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt() -> str:
        """
        Tüm görsel türlerini kapsayan kısa Türkçe evrensel prompt.
        ~130 token — sınıflandırma aşaması yok.
        """
        return (
            "Bu görseli detaylı analiz et. Yanıtını Türkçe yaz.\n\n"
            "İlk satıra görselin türünü yaz "
            "(TÜR: DÜZ_METİN | TABLO | SEMA_GRAFIK | GORSEL_BETİMLEME), "
            "ardından analizine başla:\n\n"
            "METİN ise: ne hakkında olduğunu özetle, "
            "tüm metni Markdown olarak eksiksiz çıkar.\n\n"
            "TABLO ise: ne tablosu olduğunu açıkla, "
            "tüm veriyi Markdown tablosu olarak çiz; satır/sütun yapısını bozma.\n\n"
            "ŞEMA/DİYAGRAM/AKIŞ ise:\n"
            "  GENEL AMAÇ: Amacını 1-2 cümleyle açıkla.\n"
            "  YAPI VE AKIŞ: Katmanları, blokları ve akış yönünü detaylıca anlat. "
            "Sözde kod yazma.\n\n"
            "GRAFİK/CHART ise (çubuk, pasta, çizgi vb.): "
            "Başlığını, eksen etiketlerini ve tüm değerleri oku; trendi yorumla.\n\n"
            "FOTOĞRAF/HARİTA ise: detaylıca betimle. "
            "Hayvan varsa türünü, harita ise bölgeyi belirt. "
            "Emin olmadığında 'muhtemelen' de; uydurma.\n\n"
            "Teknik terimleri ve yabancı etiketleri çevirme; orijinal dilinde bırak.\n"
            "Son olarak 'Görsel İçeriği:' bölümünde görseldeki "
            "TÜM metin etiketlerini listele.\n"
            "Tamamlayınca '[ANALİZ_BİTTİ]' yaz."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Yanıt ayrıştırma
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, str]:
        """Model çıktısını (kategori, içerik) ikilisine ayırır."""
        raw = raw.replace("[ANALİZ_BİTTİ]", "").strip()
        lines = raw.split("\n", 1)

        first = lines[0].strip().upper()
        if first.startswith("TÜR:"):
            cat_raw = first[4:].strip()
            category = "GORSEL_BETİMLEME"
            for valid in _VALID_CATEGORIES:
                if valid in cat_raw:
                    category = valid
                    break
            content = lines[1].strip() if len(lines) > 1 else ""
        else:
            category = "GORSEL_BETİMLEME"
            content = raw

        return category, content

    # ──────────────────────────────────────────────────────────────────────────
    # Ana giriş noktası
    # ──────────────────────────────────────────────────────────────────────────

    def extract_text(self, image_path: str) -> str:
        """
        Tek geçişli analiz — görsel vision encoder'dan BİR kez geçer.
        """
        if not os.path.exists(image_path):
            print(f"[VLM UYARI] Dosya bulunamadı: {image_path}")
            return ""

        img_name = os.path.basename(image_path)
        print(f"\n[VLM] Görsel analiz ediliyor: {img_name}")

        # Görsel hazırlama
        prep_start = time.time()
        try:
            data_uri = self._prepare_image(image_path)
        except Exception as e:
            print(f"[VLM HATA] Görsel hazırlanamadı ({img_name}): {e}")
            return ""
        print(f"      Hazırlama: {time.time() - prep_start:.2f}s")

        # Inference
        prompt = self._build_prompt()

        try:
            inf_start = time.time()
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sen yapılandırılmış analiz yapan, öz ve net konuşan "
                            "bir asistansın. Gereksiz tekrarlara girmez, isteneni "
                            "verir ve sözünü bitirirsin. Teknik terimleri ve "
                            "yabancı dil etiketlerini asla Türkçe'ye çevirmezsin."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=(
                    AppConfig.VLM_MAX_TOKENS
                    if AppConfig.VLM_MAX_TOKENS is not None
                    else 1024
                ),
                temperature=(
                    AppConfig.VLM_TEMPERATURE
                    if AppConfig.VLM_TEMPERATURE is not None
                    else 0.0
                ),
                repeat_penalty=1.08,
                stop=["[ANALİZ_BİTTİ]"],
            )
            inf_duration = time.time() - inf_start

            raw = response["choices"][0]["message"]["content"].strip()
            category, content = self._parse_response(raw)

            print(f"      Tür: {category} | Inference: {inf_duration:.2f}s")

            if not content:
                print(f"[VLM UYARI] Model boş içerik döndürdü ({img_name})")
                return ""

            return f"--- [GÖRSEL TÜRÜ: {category}] ---\n{content}"

        except Exception as e:
            print(f"[VLM HATA] Inference başarısız ({img_name}): {e}")
            return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Bellek yönetimi
    # ──────────────────────────────────────────────────────────────────────────

    def unload(self) -> None:
        print("[SİSTEM] VLM Motoru VRAM'den tahliye ediliyor...")
        del self.llm
        del self.chat_handler
        print("[SİSTEM] VLM belleği temizlendi.")
