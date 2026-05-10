# import os
import gc
import time

from core.config import AppConfig
from core.llm_engine import LLMEngine
from core.retriever import RetrieverEngine

# os.environ["HF_HUB_OFFLINE"] = "1"


def main():
    print("=== YEREL RAG SİSTEMİ BAŞLATILIYOR ===")

    print("[1/4] Arama Motoru ve Reranker (CPU) başlatılıyor...")
    start_time = time.time()
    retriever = RetrieverEngine()
    print(f"      Sistem hazır. (Süre: {time.time() - start_time:.2f} sn)\n")

    question = "Mitre Attack hakkında bilgi verir misin?"
    print(f"Soru: {question}\n")

    print("[2/4] Veritabanında arama yapılıyor ve Reranker süzgecinden geçiriliyor...")
    check_time = time.time()
    context_text = retriever.get_relevant_context(
        query=question, top_n=AppConfig.RERANKER_TOP_N, threshold=0.0
    )
    print(f"      Bağlam süzüldü. (Süre: {time.time() - check_time:.2f} sn)\n")

    retriever.unload()
    del retriever
    gc.collect()

    print("[3/4] Gemma Modeli VRAM'e yükleniyor...")
    check_time = time.time()
    llm = LLMEngine()
    print(f"      Gemma yüklendi. (Süre: {time.time() - check_time:.2f} sn)\n")

    print("\n[4/4] Gemma düşünüyor ve yanıt üretiyor...")
    print("=" * 60)
    generation_start = time.time()
    answer = llm.generate_answer(context=context_text, question=question)
    print(answer)
    print("=" * 60)
    print(f"\nYanıt Üretim Süresi: {time.time() - generation_start:.2f} saniye")
    print(f"\n=== İŞLEM BAŞARILI! (Toplam Süre: {time.time() - start_time:.2f} sn) ===")

    llm.unload()
    del llm
    gc.collect()


if __name__ == "__main__":
    main()
