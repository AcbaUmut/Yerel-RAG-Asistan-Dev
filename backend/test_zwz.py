import base64
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

print("VLM Çift Motorlu Sistem Ateşleniyor... Lütfen Bekleyin.")
baslangic = time.time()

# 1. Aşama: Gözleri (Vision Projector) Yüklemek
# F16 ve Q4 denemelerini buradaki dosya adını değiştirerek yapacaksın!
gozler = Llava15ChatHandler(clip_model_path="./backend/models/mmproj-zwz-4b-f16.gguf")

# 2. Aşama: Beyni Yüklemek ve Gözlerle Bağlamak
llm = Llama(
    model_path="./backend/models/zwz-4b-q4_k_m.gguf",
    chat_handler=gozler,  # Gözleri beyne bağlıyoruz
    n_gpu_layers=-1,  # Ekran kartını tam kapasite kullan
    n_ctx=4096,  # DİKKAT: Resimler çok fazla hafıza kaplar, bu yüzden kapasiteyi 4096'ya çıkardık!
    verbose=False,
)
yukleme_suresi = time.time() - baslangic
print(f"Sistem {yukleme_suresi:.2f} saniyede VRAM'e yüklendi!\n")


# 3. Aşama: Resmi Matematiksel Veriye (Base64) Çevirmek
def resmi_oku(dosya_yolu):
    with open(dosya_yolu, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"


resim_verisi = resmi_oku("test_resmi.jpg")
print("Resim tarandı, modele iletiliyor...")

# 4. Aşama: Yönerge ve Kısıtlamalar (Senin istediğin kısa cevap ayarı)
# VLM modellerinde 'create_chat_completion' yapısı kullanılır.
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": """Sen uzman bir görsel analiz ve OCR asistansın. Çıktıların her zaman yapısal ve net olmalıdır.

Lütfen görseli incele ve aşağıdaki kategorilerden hangisine girdiğini belirleyerek ona uygun formatta çıktı ver:

1. EĞER GÖRSEL BİR ŞEMA/HİYERARŞİ İSE: Markdown listesi kullan (Örn: * Ana Başlık \n  - Alt Başlık). Tüm kutuları ve yazıları eksiksiz aktar. Bunlar dışında görselin genel yapısından ve ne olduğundan kısaca bahset.
2. EĞER GÖRSEL BİR TABLO İSE: Markdown tablosu kullan (| Sütun 1 | Sütun 2 |). Tüm verileri hücrelere doğru yerleştir. Bunlar dışında görselin genel yapısından ve ne olduğundan kısaca bahset.
3. EĞER GÖRSEL METİN İÇEREN STANDART BİR BELGE İSE: Metni paragraf düzenini koruyarak olduğu gibi (OCR) aktar. Bunlar dışında görselin genel yapısından ve ne olduğundan kısaca bahset.
4. EĞER GÖRSEL GÜNDELİK BİR FOTOĞRAF İSE (Manzara, Hayvan, Eşya vb.): Görseldeki en önemli 3 nesneyi ve genel durumu kısa bir paragrafla nesnel olarak açıkla.
""",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": resim_verisi}},
                {
                    "type": "text",
                    "text": "Bu görseli analiz et ve kurallara uygun şekilde çıktısını ver.",
                },
            ],
        },
    ],
    max_tokens=400,
    temperature=0.0,
)

# 5. Aşama: Sonucu Ekrana Yazdırma
cevap = response["choices"][0]["message"]["content"].strip()
print("\n--- ZWZ MODELİNİN CEVABI ---")
print(cevap)
print("----------------------------")
