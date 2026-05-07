from typing import List

import chromadb
import torch
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


class JinaEmbeddings(BaseEmbedding):
    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__(model_name="jina-v5-nano-onnx")
        print("[SİSTEM] Jina V5 Nano ONNX (CPU) yükleniyor...")
        model_dir = "./backend/models/jina-v5-nano"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
        # Resmi dokümantasyondaki kullanım — optimum wrapper
        self._model = ORTModelForFeatureExtraction.from_pretrained(
            model_dir,
            subfolder="onnx",
            file_name="model.onnx",
            provider="CPUExecutionProvider",
            trust_remote_code=True,
            local_files_only=True,
        )
        print("[SİSTEM] Jina V5 Nano ONNX başarıyla yüklendi.")

    def _encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,  # max_lenght = 8192 maksimum.
            return_tensors="pt",
        )
        # Resmi dokümantasyondaki pooling — last-token pooling
        with torch.no_grad():
            outputs = self._model(**inputs)

        last_hidden = outputs.last_hidden_state
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
        embeddings = last_hidden[torch.arange(last_hidden.size(0)), seq_lengths]

        # L2 normalize
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embeddings = (embeddings / norms).numpy()
        return embeddings.tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode([f"Document: {text}"])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode([f"Query: {query}"])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._encode([f"Document: {t}" for t in texts])


class VectorStoreEngine:
    def __init__(
        self,
        persist_dir: str = "./backend/chroma_db",
        collection_name: str = "tez_koleksiyonu",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        print(
            f"[SİSTEM] VectorStoreEngine başlatılıyor... Kayıt dizini: {self.persist_dir}"
        )

        Settings.embed_model = JinaEmbeddings()
        Settings.llm = None

        self.db_client = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.db_client.get_or_create_collection(
            self.collection_name
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def add_nodes(self, nodes, file_name: str):
        if not nodes:
            print("Uyarı: Veritabanına eklenecek düğüm (node) bulunamadı.")
            return None

        try:
            self.chroma_collection.delete(where={"file_name": file_name})
            print(
                f"[SİSTEM] Eski '{file_name}' kayıtları ChromaDB'den başarıyla temizlendi."
            )
        except Exception:
            pass

        print(f"Toplam {len(nodes)} düğüm vektör uzayına gömülüyor...")
        index = VectorStoreIndex(nodes=nodes, storage_context=self.storage_context)
        print("İşlem Başarılı! Düğümler ChromaDB'ye kaydedildi.")
        return index


if __name__ == "__main__":
    print("Test için lütfen ingest.py dosyasını kullanınız.")
