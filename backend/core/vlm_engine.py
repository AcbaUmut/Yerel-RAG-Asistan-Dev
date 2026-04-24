import base64
import os

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


class VLMEngine:
    def __init__(self, model_path: str, mmproj_path: str):
        """
        Görsel Dil Modelini (VLM) başlatır.
        """
        if not os.path.exists(model_path) or not os.path.exists(mmproj_path):
            raise FileNotFoundError(
                "[HATA] VLM veya mmproj model dosyaları bulunamadı!"
            )

        print("[SİSTEM] VLM Motoru (ZwZ-4B) VRAM'e yükleniyor...")

        # Görsel veriyi dil modelinin anlayacağı vektörlere çeviren köprü
        self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)

        # Ana Modeli Yükleme (VRAM'i dolduracak olan kısım)
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,  # Ekran kartını tam kapasite kullan
            verbose=False,  # Terminali gereksiz loglarla doldurmaması için
        )
        print("[SİSTEM] VLM Motoru başarıyla ayağa kalktı.")

    def _image_to_base64_data_uri(self, file_path: str) -> str:
        """
        llama.cpp'nin resmi okuyabilmesi için onu Base64 formatına çevirir.
        """
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

    def extract_text(self, image_path: str) -> str:
        """
        Acımasız OCR Modu: Resimdeki metni/tabloyu okur, yorum yapmadan döndürür.
        """
        if not os.path.exists(image_path):
            return ""

        print(f"[VLM] Görsel analiz ediliyor: {os.path.basename(image_path)}")
        data_uri = self._image_to_base64_data_uri(image_path)

        # VLM'in gevezeliğini susturduğumuz ama şema yeteneği eklediğimiz "Gelişmiş Diktatör İstemi"
        system_prompt = (
            "Sen bir veri çıkarma (OCR) ve yapılandırılmış analiz asistanısın. "
            "Görevlerin şunlardır:\n"
            "1. Sadece ve sadece görseldeki metinleri, formülleri veya tabloları oku.\n"
            "2. ÖNEMLİ: Eğer görsel bir AKIŞ DİYAGRAMI, ŞEMA veya ALGORİTMA ise, sadece yazıları alt alta yazma! Mantıksal akışı, hiyerarşiyi ve okların yönünü koruyarak yapılandırılmış Markdown (pseudo-code veya girintili liste) olarak çıktı ver.\n"
            "3. Görselde ne gördüğüne dair hiçbir yorum yapma, sohbet etme (Örn: 'Bu resimde bir şema var' DEME).\n"
            "4. Eğer görsel sadece okunacak metni olmayan bir obje, manzara veya fotoğraf ise (formül/tablo/yazı yoksa) hiçbir şey yazmadan boş bırak."
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {
                                "type": "text",
                                "text": "Bu görseldeki metinleri veya tabloları Markdown formatında çıkar.",
                            },
                        ],
                    },
                ],
                max_tokens=1024,  # Görseldeki metin uzun olabileceği için sınırı geniş tuttuk
                temperature=0.0,  # Halüsinasyonu sıfırlamak için (Yaratıcılık kapalı)
            )

            extracted_text = response["choices"][0]["message"]["content"].strip()
            return extracted_text

        except Exception as e:
            print(f"[VLM HATA] Görsel okunurken bir sorun oluştu: {e}")
            return ""

    def unload(self):
        """
        VRAM Nöbet Değişimi: İşlem bitince VLM'i RAM/VRAM'den siler.
        Bu sayede Gemma (LLM) yüklendiğinde 'Out of Memory' hatası almayız.
        """
        print("[SİSTEM] VLM Motoru VRAM'den tahliye ediliyor...")
        del self.llm
        del self.chat_handler
