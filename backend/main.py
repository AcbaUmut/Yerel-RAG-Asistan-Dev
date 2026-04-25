import time

from core.llm_engine import LLMEngine
from core.retriever import RetrieverEngine

# Yutma işlemleri zaten yapıldığı için parser ve vector_store importlarını kapattık.


def main():
    model_path = "./backend/models/Turkish-Gemma-9b-T1-Q4_K_M.gguf"  # Dosya adını kontrol et (Nokta yerine tire olabilir)

    print("=== YEREL RAG SİSTEMİ BAŞLATILIYOR ===")

    # 1. AŞAMA: GERİ ÇAĞIRMA (SADECE CPU VE RAM)
    # GPU bu aşamada tamamen boşta dinleniyor.
    print("[1/4] Arama Motoru ve Hakem (CPU) başlatılıyor...")
    start_time = time.time()
    retriever = RetrieverEngine()
    print(f"      Sistem hazır. (Süre: {time.time() - start_time:.2f} sn)\n")

    question = "C dili nedir?"

    print(f"Soru: {question}\n")

    print("[2/4] Veritabanında arama yapılıyor ve Hakem süzgecinden geçiriliyor...")
    check_time = time.time()
    context_text = retriever.get_relevant_context(question, top_n=3, threshold=0.0)
    print(f"      Bağlam süzüldü. (Süre: {time.time() - check_time:.2f} sn)\n")

    # 2. AŞAMA: LLM MOTORUNU AYAĞA KALDIRMA (GPU İŞGALİ BAŞLIYOR)
    # Veriyi bulduk, temizledik, artık cevap üretmek için ekran kartını devreye sokuyoruz.
    print("[3/4] Gemma Q4_K_M Modeli VRAM'e yükleniyor...")
    check_time = time.time()
    llm = LLMEngine(model_path=model_path)
    print(f"      Gemma yüklendi. (Süre: {time.time() - check_time:.2f} sn)\n")

    print("\n[4/4]Gemma düşünüyor ve yanıt üretiyor...")
    print("=" * 60)

    generation_start = time.time()
    answer = llm.generate_answer(context=context_text, question=question)

    print(answer)
    print("=" * 60)
    print(f"\nYanıt Üretim Süresi: {time.time() - generation_start:.2f} saniye")
    print("=== TEST BAŞARIYLA TAMAMLANDI ===")
    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")


if __name__ == "__main__":
    main()
