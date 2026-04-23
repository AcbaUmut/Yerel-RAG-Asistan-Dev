import time

from llama_cpp import Llama

print(
    "Motor ateşleniyor... Lütfen bekleyin ve GPU-Z / Görev Yöneticisini kontrol edin."
)

# 1. Aşama: Modeli VRAM'e Yükleme
baslangic = time.time()
llm = Llama(
    model_path="./backend/models/Turkish-Gemma-9b-T1.Q4_K_M.gguf",  # Dosya adını kontrol et!
    n_gpu_layers=-1,  # ÇOK KRİTİK: -1 demek, modelin %100'ünü ekran kartına yükle demektir.
    n_ctx=2048,  # Bağlam penceresi. Şimdilik VRAM'i şişirmemek için 2048'de tutuyoruz.
    verbose=False,  # Terminali C++ loglarıyla kirletmemesi için kapattık.
)
yukleme_suresi = time.time() - baslangic
print(f"Model {yukleme_suresi:.2f} saniyede başarıyla VRAM'e yüklendi!\n")

# 2. Aşama: Modeli Konuşturma (Senin istediğin o kısa cevap kısıtıyla)
soru = "Yapay zeka nedir?"
prompt = f"Sen kısa ve öz konuşan bir asistansın. Soru: {soru}\nCevap:"

print(f"Soru soruluyor: '{soru}'...")
output = llm(
    prompt,
    max_tokens=64,  # Modelin vereceği cevabı fiziksel olarak sınırlandırıyoruz.
    temperature=0.3,  # Yaratıcılığı düşürüp, daha net ve mantıklı cevaplar vermesini sağlıyoruz.
    stop=[
        "Soru:",
        "\n\n",
        "User:",
    ],  # FREN SİSTEMİ: Model bu kelimelerden birini üretmeye kalkarsa, anında konuşmayı keser.
    echo=False,
)

# 3. Aşama: Sonucu Ekrana Yazdırma
cevap = output["choices"][0]["text"].strip()
print("\n--- GEMMA'NIN CEVABI ---")
print(cevap)
print("------------------------")
