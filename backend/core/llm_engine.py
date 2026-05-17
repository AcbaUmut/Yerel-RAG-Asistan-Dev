import logging
import os
import time

from core.config import AppConfig
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

log = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self):
        self.model_path = str(AppConfig.LLM_MODEL_PATH)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

        load_start = time.time()

        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=AppConfig.LLM_TEMPERATURE,
            max_tokens=AppConfig.LLM_MAX_TOKENS,
            n_ctx=AppConfig.LLM_N_CTX,
            n_gpu_layers=-1,
            n_batch=512,
            repeat_penalty=1.1,
            verbose=False,
            # type_k / type_v LangChain'in bildiği parametreler değil.
            # model_kwargs içine koyunca LangChain uyarı vermeden
            # doğrudan llama.cpp'ye iletir. 8 = GGML_TYPE_Q8_0.
            # f16_kv=True yerine bu yöntem: KV cache belleği ~%50 düşer.
            model_kwargs={"type_k": 8, "type_v": 8},
        )

        prompt_text = """<start_of_turn>user
Sen verilen bağlama dayanarak cevap üreten bir asistansın.

TEMEL KURAL:
- Sadece bağlamdaki bilgileri kullan, dış bilgi (önceden eğitildiğin bilgiler) ekleme.
- Bağlam içinde sentez ve çıkarım yapabilirsin: farklı parçaları birleştirebilir, bilgiyi yeniden ifade edebilir, eldeki bilgilerden mantıksal sonuçlar çıkarabilirsin.
- Soruda geçen kelimelerin bağlamda birebir geçmesi gerekmez. Soruyla konu olarak yakın bilgi bağlamda varsa, ondan cevap üret.

NE ZAMAN CEVAP ÜRETME:
- Bağlam soruyla tamamen alakasızsa "Bu bilgiye sahip değilim." de.
- Bağlamda olmayan detayları ekleme, uydurma yapma.

GÖRSELLERDEN GELEN BİLGİLER:
- Bağlamda <VLM_START ...>...<VLM_END> etiketleri arasında gördüğün içerik, dokümandaki görsellerden (tablo, şema, grafik vb.) bir görsel modeli tarafından çıkarılmış metindir. Doğrudan dokümanın yazılı kısmı değildir.
- Bağlam (hem VLM blokları hem doğrudan dokümandan okunan metin/tablolar) yapısal hata içerebilir: bir sayı yanlış okunmuş, bir etiket atlanmış, sütun başlıkları veri hücreleriyle yanlış eşleşmiş ya da satır/sütun hizalaması kaymış olabilir. Bir tablo veya yapıdaki tutarsızlık fark edersen, içerikteki mantıksal ilişkilere bakarak hangi değerin hangi sütuna/kategoriye ait olduğunu çıkarsa.
- Bir VLM bloğunun ne anlattığı net değilse, aynı bağlam parçasındaki çevresindeki metne (başlık, üst/alt paragraflar) bakarak görselin orada neyi temsil ettiğini çıkarsamayı dene.

Bağlam:
{context}

Soru: {question}<end_of_turn>
<start_of_turn>model
"""

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"], template=prompt_text
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

        log.info(f"LLM Motoru (Gemma) yüklendi ({time.time() - load_start:.2f} sn).")

    def generate_answer(self, context: str, question: str) -> str:
        try:
            return self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            # ERROR seviyesi + exc_info: traceback dosyaya zengin biçimde gider,
            # kullanıcı terminalde kısa mesaj görür, hata kaybolmaz.
            log.error(f"LLM yanıt üretirken hata oluştu: {e}", exc_info=True)
            return f"LLM Yanıt Üretirken Hata Oluştu: {str(e)}"

    def generate_answer_stream(self, context: str, question: str):
        """
        Cevabı token token üreten generator versiyon.

        LangChain'in chain.stream() metodu generator döndürür; her bir
        parça (genelde 1-2 token) ortaya çıkar çıkmaz yield ile dışarı
        aktarılır. Çağıran taraf for döngüsü ile parçaları toplar veya
        HTTP stream'e yazar.
        """
        try:
            for chunk in self.chain.stream({"context": context, "question": question}):
                # chunk her zaman string — StrOutputParser ile çıktı parse edildi.
                # Boş chunk olabilir, frontend'e yollamak anlamsız.
                if chunk:
                    yield chunk
        except Exception as e:
            log.error(f"LLM stream sırasında hata: {e}", exc_info=True)
            yield f"\n\n[HATA] LLM yanıt üretirken sorun oluştu: {e}"

    def unload(self):
        """
        LLM'i VRAM'den serbest bırakır.

        LangChain LlamaCpp arkada llama-cpp-python kullanıyor — C++ tabanlı,
        PyTorch değil. del + gc.collect() yeterli.
        """
        import gc

        log.info("LLM bellekten tahliye ediliyor...")
        if hasattr(self, "chain"):
            del self.chain
        if hasattr(self, "llm"):
            del self.llm
        gc.collect()
        log.info("LLM belleği temizlendi.")
