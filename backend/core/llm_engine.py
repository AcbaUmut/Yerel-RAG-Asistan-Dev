import os

from core.config import AppConfig  # YENİ: Merkezi Sinir Sistemini içe aktardık
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class LLMEngine:
    # Parametre verilmezse Config'den almasını sağladık
    def __init__(self):
        self.model_path = f"./backend/models/{AppConfig.LLM_MODEL_NAME}"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

        print("[SİSTEM] LLM Motoru (Gemma) başlatılıyor...")

        # 1. MOTOR KURULUMU: TAM GPU HAKİMİYETİ
        self.llm = LlamaCpp(
            model_path=self.model_path,
            temperature=AppConfig.LLM_TEMPERATURE or 0.1,  # Config'den çekildi
            max_tokens=AppConfig.LLM_MAX_TOKENS or 1024,  # Config'den çekildi
            n_ctx=AppConfig.LLM_N_CTX or 4096,  # Config'den çekildi
            n_gpu_layers=-1,  # Modelin TAMAMI 8GB VRAM'e yükleniyor. PCIe darboğazı iptal.
            n_batch=512,  # Config dışı, sabit tutuldu
            f16_kv=True,  # 4096 tokenın VRAM'e sığması için KV önbelleği 16-bit'te tutulmalı.
            repeat_penalty=1.1,  # Cosmos ekibinin önerisi: Sonsuz döngü/tekrar engelleme.
            verbose=False,
        )

        # 2. SİSTEM İSTEMİ (Gemma-2 Formatı)
        # Model mantık yürütmeye optimize edildiği için kısıtlamayı çok dar tutmuyoruz.
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

        # 3. ZİNCİRİN KURULUMU
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_answer(self, context: str, question: str) -> str:
        try:
            return self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            return f"LLM Yanıt Üretirken Hata Oluştu: {str(e)}"
