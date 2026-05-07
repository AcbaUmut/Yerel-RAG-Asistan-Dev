import os

from core.config import AppConfig
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class LLMEngine:
    def __init__(self):
        self.model_path = f"./backend/models/{AppConfig.LLM_MODEL_NAME}"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

        print("[SİSTEM] LLM Motoru (Gemma) başlatılıyor...")

        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=(
                AppConfig.LLM_TEMPERATURE
                if AppConfig.LLM_TEMPERATURE is not None
                else 0.1
            ),
            max_tokens=(
                AppConfig.LLM_MAX_TOKENS
                if AppConfig.LLM_MAX_TOKENS is not None
                else 1024
            ),
            n_ctx=(AppConfig.LLM_N_CTX if AppConfig.LLM_N_CTX is not None else 4096),
            n_gpu_layers=-1,
            n_batch=512,
            f16_kv=True,
            repeat_penalty=1.1,
            verbose=False,
        )

        prompt_text = """<start_of_turn>user
        Sen KESİNLİKLE kendi önceden eğitilmiş bilgini KULLANMAYAN, sadece verilen bağlama sadık kalan bir asistansın.
        Aşağıdaki 'Bağlam' metninde kullanıcının sorusunun cevabı yoksa, "Bu bilgiye sahip değilim." de. Asla uydurma yapma!

        Bağlam:
        {context}

        Soru: {question}<end_of_turn>
        <start_of_turn>model
        """
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"], template=prompt_text
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_answer(self, context: str, question: str) -> str:
        try:
            return self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            return f"LLM Yanıt Üretirken Hata Oluştu: {str(e)}"

    def unload(self):
        print("[SİSTEM] LLM bellekten tahliye ediliyor...")
        del self.llm
        del self.chain
        print("[SİSTEM] LLM belleği temizlendi.")
