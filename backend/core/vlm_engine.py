import base64
import os
import time

from core.config import AppConfig  # YENİ: Merkezi Sinir Sistemini içe aktardık
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


class VLMEngine:
    def __init__(self):
        """
        Görsel Dil Modelini (VLM) başlatır.
        İki Aşamalı Mimari (Gözcü Sınıflandırıcı + Uzman Çıkarıcı)
        """
        # Parametre gelmezse Config'den çek
        self.model_path = f"./backend/models/{AppConfig.VLM_MODEL_NAME}"
        self.mmproj_path = f"./backend/models/{AppConfig.VLM_MMPROJ_NAME}"

        if not os.path.exists(self.model_path) or not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(
                "[HATA] VLM veya mmproj model dosyaları bulunamadı!"
            )

        print("[SİSTEM] VLM Motoru (ZwZ-4B) VRAM'e yükleniyor...")

        self.chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)

        self.llm = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,
            n_ctx=(
                AppConfig.VLM_N_CTX if AppConfig.VLM_N_CTX is not None else 4096
            ),  # Config'den çekildi (4096)
            n_gpu_layers=-1,
            verbose=False,
        )
        print("[SİSTEM] VLM Motoru başarıyla ayağa kalktı.")

    def _image_to_base64_data_uri(self, file_path: str) -> str:
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

    def _classify_image(self, data_uri: str) -> str:
        """
        AŞAMA 1: GÖZCÜ (Sınıflandırma)
        """
        classifier_prompt = (
            "Bu görselin türü nedir? Sadece şu 4 kelimeden BİRİNİ yaz: "
            "DÜZ_METİN, TABLO, SEMA_GRAFIK, GORSEL_BETİMLEME. "
            "İPUCU: Eğer görselde oklar olmasa bile, katmanlı yapılar (layers), hiyerarşik bloklar, "
            "sistem topolojileri, mimari tasarımlar veya algoritmik akış diyagramları varsa KESİNLİKLE 'SEMA_GRAFIK' seç. "
            "Başka hiçbir kelime veya açıklama ekleme."
        )
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Sen bir görsel sınıflandırma asistanısın. Sadece istenen kategoriyi döndürürsün.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": classifier_prompt},
                        ],
                    },
                ],
                max_tokens=15,  # Gözcü olduğu için sabit bırakıldı (Sadece 1 kelime üretecek)
                temperature=(
                    AppConfig.VLM_TEMPERATURE
                    if AppConfig.VLM_TEMPERATURE is not None
                    else 0.0
                ),  # Config'den çekildi
            )
            category = response["choices"][0]["message"]["content"].strip().upper()

            valid_categories = ["DÜZ_METİN", "TABLO", "SEMA_GRAFIK", "GORSEL_BETİMLEME"]
            for valid_cat in valid_categories:
                if valid_cat in category:
                    return valid_cat
            return "GORSEL_BETİMLEME"

        except Exception as e:
            print(f"[VLM SINIFLANDIRMA HATA] {e}")
            return "GORSEL_BETİMLEME"

    def _get_specialist_prompt(self, category: str) -> str:
        """
        Gözcüden gelen kategoriye göre Uzman'a verilecek zengin talimatı döndürür.
        RAG KELİME İNDEKSLEMESİ (Görsel İçeriği) VE SÖZDE KOD YASAĞI EKLENDİ.
        """
        end_rule = "\nAnalizini tamamladığında sonuna KESİNLİKLE tam olarak '[ANALİZ_BİTTİ]' yaz ve kelime üretmeyi bırak."

        if category == "DÜZ_METİN":
            return (
                "Önce bu görselin ne hakkında olduğunu (örneğin bir kitap sayfası, slayt, kod bloğu veya formül olduğunu) tek bir cümle ile özetle. "
                "Ardından görseldeki tüm metinleri, formülleri veya karakterleri Markdown formatında eksiksiz olarak çıkar."
                + end_rule
            )
        elif category == "TABLO":
            return (
                "Önce bu tablonun ne tablosu olduğunu ve hangi verileri içerdiğini tek bir cümle ile açıkla. "
                "Ardından satır ve sütun yapısını KESİNLİKLE bozmadan tüm veriyi Markdown tablosu olarak çiz."
                + end_rule
            )
        elif category == "SEMA_GRAFIK":
            return (
                "Bu görsel bir şema, mimari, katmanlı yapı, sistem topolojisi veya akış diyagramıdır. "
                "1. Görselin genel amacını 1-2 cümleyle açıkla.\n"
                "2. Görseldeki hiyerarşiyi, katmanları (yukarıdan aşağıya veya tam tersi), yapısal blokları ve eğer varsa akış yönünü detaylıca anlat. KESİNLİKLE uydurma sözde kod (pseudo-code) veya sahte algoritmalar YAZMA.\n"
                "3. En sona KESİNLİKLE 'Görsel İçeriği:' adında bir bölüm aç ve görselin içindeki tüm metinleri, etiketleri ve anahtar kelimeleri eksiksiz bir liste halinde alt alta yaz."
                + end_rule
            )
        else:  # GORSEL_BETİMLEME
            return (
                "Bu görsel bir fotoğraf, çizim, nesne veya sahnedir. "
                "Görselde genel olarak ne gördüğünü detaylı ve zengin bir dille betimle. "
                "En sona KESİNLİKLE 'Görsel İçeriği:' adında bir bölüm aç ve görselin içinde okunabilen tüm metin veya tabelaları liste halinde ekle."
                + end_rule
            )

    def extract_text(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return ""

        print(f"\n[VLM] Görsel analiz ediliyor: {os.path.basename(image_path)}")
        data_uri = self._image_to_base64_data_uri(image_path)

        total_start = time.time()

        # AŞAMA 1: Sınıflandırma
        print("[VLM] Aşama 1: Gözcü çalışıyor (Sınıflandırma)...")
        cat_start = time.time()
        category = self._classify_image(data_uri)
        cat_end = time.time()
        print(f"      Gözcü Çalışmayı Bitirdi! (Süre: {cat_end - cat_start:.2f} sn)")
        print(f"      Tespit edilen tür: {category}")

        # AŞAMA 2: Uzman Çıkarımı
        print("[VLM] Aşama 2: Uzman çalışıyor (Detaylı Analiz)...")
        specialist_prompt = self._get_specialist_prompt(category)

        try:
            exp_start = time.time()
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Sen yapılandırılmış analiz yapan, öz ve net konuşan bir asistansın. Gereksiz tekrarlara girmez, isteneni verir ve sözünü bitirirsin.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": specialist_prompt},
                        ],
                    },
                ],
                max_tokens=(
                    AppConfig.VLM_MAX_TOKENS
                    if AppConfig.VLM_MAX_TOKENS is not None
                    else 1024
                ),  # Config'den çekildi
                temperature=(
                    AppConfig.VLM_TEMPERATURE
                    if AppConfig.VLM_TEMPERATURE is not None
                    else 0.0
                ),  # Config'den çekildi
                repeat_penalty=1.20,
                stop=[
                    "[ANALİZ_BİTTİ]"  # ÖZEL ATEŞKES SİNYALİMİZ
                ],
            )
            exp_end = time.time()

            # Modelden gelen ham metni alıyoruz
            extracted_text = response["choices"][0]["message"]["content"].strip()

            # Eğer model [ANALİZ_BİTTİ] yazdıysa temizliyoruz
            extracted_text = extracted_text.replace("[ANALİZ_BİTTİ]", "").strip()

            print(
                f"      Uzman Çalışmayı Bitirdi! (Süre: {exp_end - exp_start:.2f} sn)"
            )
            print(f"      [Toplam VLM İşlemi: {exp_end - total_start:.2f} sn]\n")

            return f"--- [GÖRSEL TÜRÜ: {category}] ---\n{extracted_text}"

        except Exception as e:
            print(f"[VLM HATA] Görsel okunurken bir sorun oluştu: {e}")
            return ""

    def unload(self):
        print("[SİSTEM] VLM Motoru VRAM'den tahliye ediliyor...")
        del self.llm
        del self.chat_handler
