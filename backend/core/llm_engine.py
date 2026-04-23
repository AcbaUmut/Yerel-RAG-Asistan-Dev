import os

from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class LLMEngine:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        # 1. MOTOR KURULUMU: TAM GPU HAKİMİYETİ
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=2048,  # Düşünme faslı uzun süreceği için maksimum token artırıldı.
            n_ctx=4096,  # Genişletilmiş bağlam ve düşünme penceresi.
            n_gpu_layers=-1,  # Modelin TAMAMI 8GB VRAM'e yükleniyor. PCIe darboğazı iptal.
            n_batch=512,
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
