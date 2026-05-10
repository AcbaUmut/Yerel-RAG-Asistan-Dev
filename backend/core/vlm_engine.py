"""
VLM Engine — ZwZ-4B (Qwen3VL tabanlı)
Kategorisiz, kural tabanlı evrensel prompt.
"""

import base64
import io
import os
import time

from core.config import AppConfig
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler
from PIL import Image

_PATCH_SIZE: int = 32
_VLM_MAX_PIXELS: int = 1_310_720


class VLMEngine:
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

    def _prepare_image(self, file_path: str) -> str:
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

    @staticmethod
    def _build_prompt() -> str:
        return (
            "Bu görseli analiz et.\n\n"
            "Görselin içeriğine göre:\n"
            "- Metin varsa tamamını Markdown olarak eksiksiz çıkar.\n"
            "- Tablo varsa Markdown tablosu olarak çiz; "
            "satır/sütun yapısını bozma, tablo içine not veya açıklama satırı ekleme.\n"
            "- Şema veya diyagram varsa yapısını, bileşenlerini ve akışını anlat.\n"
            "- Grafik veya chart varsa eksen etiketlerini, değerleri oku ve trendi yorumla.\n"
            "- Fotoğraf veya görsel sahne varsa içeriği detaylıca betimle.\n"
            "- Kategori dışı kalıyorsa görselden kısaca bahset.\n\n"
            "Tüm metin, sayı, etiket ve formülleri eksiksiz aktar.\n"
            "Teknik terimler ve yabancı etiketler orijinal dilinde kalsın.\n"
            "Emin olmadığın yerlerde 'muhtemelen' de, uydurma.\n"
            "Sözde kod yazma.\n\n"
            "'[ANALİZ_BİTTİ]' ile bitir."
        )

    def extract_text(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            print(f"[VLM UYARI] Dosya bulunamadı: {image_path}")
            return ""

        img_name = os.path.basename(image_path)
        print(f"\n[VLM] Görsel analiz ediliyor: {img_name}")

        prep_start = time.time()
        try:
            data_uri = self._prepare_image(image_path)
        except Exception as e:
            print(f"[VLM HATA] Görsel hazırlanamadı ({img_name}): {e}")
            return ""
        print(f"      Hazırlama: {time.time() - prep_start:.2f}s")

        prompt = self._build_prompt()

        try:
            inf_start = time.time()
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sen bir görsel analiz asistanısın. "
                            "Sadece isteneni yap, ekstra yorum ekleme."
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

            content = response["choices"][0]["message"]["content"].strip()
            content = content.replace("[ANALİZ_BİTTİ]", "").strip()

            print(f"      Inference: {inf_duration:.2f}s")

            if not content:
                print(f"[VLM UYARI] Model boş içerik döndürdü ({img_name})")
                return ""

            return content

        except Exception as e:
            print(f"[VLM HATA] Inference başarısız ({img_name}): {e}")
            return ""

    def unload(self) -> None:
        print("[SİSTEM] VLM Motoru VRAM'den tahliye ediliyor...")
        del self.llm
        del self.chat_handler
        print("[SİSTEM] VLM belleği temizlendi.")
