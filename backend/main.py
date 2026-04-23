import os  # noqa: F401
import time

# Kendi yazdığımız çekirdek modülleri içeri aktarıyoruz
# Not: Fonksiyon/Sınıf isimlerini kendi dosyalarına göre düzenlemelisin.
from core.document_parser import parse_pdf_to_nodes
from core.llm_engine import LLMEngine
from core.retriever import get_relevant_context
from core.vector_store import create_or_load_vector_store


def main():
    # 1. Dosya Yolları
    pdf_path = "test.pdf"
    # Modelin tam adını models klasöründeki isme göre güncelle
    model_path = "./backend/models/Turkish-Gemma-9b-T1.Q4_K_M.gguf"

    print("=== YEREL RAG SİSTEMİ BAŞLATILIYOR ===")

    # 2. Veri Yutma ve Vektörleme (Sadece ilk çalışmada veya belge değiştiğinde gerekir)
    print(f"[1/4] '{pdf_path}' okunuyor ve vektör veritabanına (CPU) işleniyor...")
    start_time = time.time()

    # Eğer ChromaDB'ye zaten kaydettiysen bu iki satırı yorum satırı (#) yapabilirsin.
    chunks = parse_pdf_to_nodes(pdf_path)
    create_or_load_vector_store(chunks)

    print(f"      Vektörleme tamamlandı. (Süre: {time.time() - start_time:.2f} sn)\n")

    # 3. LLM Motorunu Ayağa Kaldırma (GPU'ya Yükleme)
    print("[2/4] Gemma Q4_K_M Modeli tamamen GPU'ya yükleniyor...")
    start_time = time.time()
    llm = LLMEngine(model_path=model_path)
    print(f"      Model yüklendi ve hazır. (Süre: {time.time() - start_time:.2f} sn)\n")

    # 4. Test Sorusu
    # Sunumunda geçen spesifik bir bilgi üzerinden test ediyoruz.
    question = (
        "Charles Babbage'ın tasarladığı makinenin adı nedir ve temel amacı neydi?"
    )
    print(f"Soru: {question}\n")

    # 5. Geri Çağırma (Retriever) ve Hakem Filtresi
    print(
        "[3/4] Hakem (Reranker) çalışıyor, veritabanından sadece en alakalı bağlam süzülüyor..."
    )
    start_time = time.time()

    # Bu fonksiyonun sana direkt metin (string) döndürdüğünü varsayıyorum.
    # Eğer Document objesi döndürüyorsa, burada birleştirme işlemi (join) yapmalısın.
    context_text = get_relevant_context(question)

    print(f"      Bağlam süzüldü. (Süre: {time.time() - start_time:.2f} sn)\n")

    # 6. Üretim (Generation)
    print("[4/4] Gemma düşünüyor ve yanıt üretiyor...\n")
    print("=" * 60)

    start_time = time.time()
    answer = llm.generate_answer(context=context_text, question=question)

    print(answer)
    print("=" * 60)
    print(f"\nYanıt Üretim Süresi: {time.time() - start_time:.2f} saniye")
    print("=== TEST BAŞARIYLA TAMAMLANDI ===")


if __name__ == "__main__":
    main()
