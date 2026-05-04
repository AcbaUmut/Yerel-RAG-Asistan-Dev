from typing import List

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import SentenceTransformer


class JinaEmbeddings(BaseEmbedding):
    """LlamaIndex native embedding — LangchainEmbedding wrapper'ı bypass eder."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__(model_name="jina-v5-nano")
        print("[SİSTEM] Jina V5 Nano (local/CPU) yükleniyor...")
        self._model = SentenceTransformer(
            "./backend/models/jina-v5-nano",
            trust_remote_code=True,
            device="cpu",
            local_files_only=True,
        )
        print("[SİSTEM] Jina V5 Nano başarıyla yüklendi.")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode([query], normalize_embeddings=True)[0].tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


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
