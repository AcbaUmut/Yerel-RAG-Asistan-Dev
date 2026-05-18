import gc
import logging
import time

from core.config import AppConfig
from core.llm_engine import LLMEngine
from core.retriever import RetrieverEngine

log = logging.getLogger(__name__)


class QueryEngine:
    """
    Sorgu orkestrasyonu: retriever + LLM.

    Eski main.py'nin sınıflaştırılmış hali. Belirli bir koleksiyon ve
    doküman üzerinde çalışır, sonucu metin olarak döndürür.

    Donanım orkestrasyonu (8GB VRAM kısıtı):
        1) Retriever yükle (Jina CPU + Reranker CPU) — VRAM kullanmaz
        2) Bağlamı çek → retriever unload
        3) LLM (GPU) yükle → cevap üret → LLM unload

    Retriever CPU'da olduğu için aslında LLM ile aynı anda durabilirdi,
    ama bellek temizliği ve tutarlılık adına eski akış korunuyor.
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: str = str(AppConfig.DATABASE_DIR),
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

    def run(self, question: str, file_name: str) -> str:
        """
        Sorguyu çalıştırır, cevabı string olarak döndürür.

        file_name: Hangi dokümanda arama yapılacak. RetrieverEngine'a
                   metadata filtresi olarak iletilir.
        """
        log.info(
            f"Sorgu başlatıldı — koleksiyon: '{self.collection_name}', "
            f"doküman: '{file_name}', soru: {question!r}"
        )
        start_time = time.time()

        # ── 1. Retriever ──
        log.info("Retriever yükleniyor...")
        t = time.time()
        retriever = RetrieverEngine(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir,
        )
        log.debug(f"Retriever hazır ({time.time() - t:.2f} sn).")

        # ── 2. Bağlam çek ──
        log.info("Veritabanında arama + reranker çalışıyor...")
        t = time.time()
        context_text = retriever.get_relevant_context(
            query=question,
            top_n=AppConfig.RERANKER_TOP_N,
            threshold=0.0,
            file_name=file_name,
        )
        log.debug(f"Bağlam hazır ({time.time() - t:.2f} sn).")

        retriever.unload()
        del retriever
        gc.collect()

        if not context_text:
            log.warning("Bağlam boş — LLM çağrılmadan dönülüyor.")
            return "Bu doküman için sorguya uygun bir bağlam bulunamadı."

        # ── 3. LLM ──
        log.info("LLM yükleniyor ve cevap üretiyor...")
        t = time.time()
        llm = LLMEngine()
        log.debug(f"LLM hazır ({time.time() - t:.2f} sn).")

        gen_start = time.time()
        answer = llm.generate_answer(context=context_text, question=question)

        log.info(f"LLM cevabı üretildi ({time.time() - gen_start:.2f} sn).")

        llm.unload()
        del llm
        gc.collect()

        log.info(f"Sorgu tamamlandı ({time.time() - start_time:.2f} sn).")
        return answer

    def run_stream(self, question: str, file_name: str):
        """
        Sorguyu çalıştırır, cevabı token token yield eder.

        Akış run() ile aynı — retriever çek, LLM çağır — ama LLM
        aşamasında stream döner. Modellerin yüklenme/boşaltma sırası
        değişmez, sadece cevabı tek seferde değil parça parça veriyoruz.
        """
        log.info(
            f"Streaming sorgu — koleksiyon: '{self.collection_name}', "
            f"doküman: '{file_name}', soru: {question!r}"
        )
        start_time = time.time()

        # ── 1. Retriever ──
        log.info("Retriever yükleniyor...")
        retriever = RetrieverEngine(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir,
        )

        # ── 2. Bağlam çek ──
        log.info("Veritabanında arama + reranker çalışıyor...")
        context_text = retriever.get_relevant_context(
            query=question,
            top_n=AppConfig.RERANKER_TOP_N,
            threshold=0.0,
            file_name=file_name,
        )

        retriever.unload()
        del retriever
        gc.collect()

        if not context_text:
            log.warning("Bağlam boş — LLM çağrılmadan dönülüyor.")
            yield "Bu doküman için sorguya uygun bir bağlam bulunamadı."
            return

        # ── 3. LLM ──
        log.info("LLM yükleniyor ve cevap akıtılıyor...")
        llm = LLMEngine()

        gen_start = time.time()
        try:
            # generate_answer_stream her chunk'ı yield ediyor;
            # biz de aynısını yukarı aktarıyoruz.
            for chunk in llm.generate_answer_stream(
                context=context_text, question=question
            ):
                yield chunk
        finally:
            # Stream bitse de yarıda kesilse de LLM mutlaka temizlensin.
            # Frontend bağlantıyı koparırsa GeneratorExit fırlar, finally yine çalışır.
            log.info(f"LLM cevabı bitti ({time.time() - gen_start:.2f} sn).")
            llm.unload()
            del llm
            gc.collect()
            log.info(f"Sorgu tamamlandı ({time.time() - start_time:.2f} sn).")
