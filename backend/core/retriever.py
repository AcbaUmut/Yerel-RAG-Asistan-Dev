import time

import torch
from core.vector_store import JinaEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModel


class RetrieverEngine:
    def __init__(self, collection_name: str = "tez_koleksiyonu"):
        self.persist_dir = "./backend/chroma_db"
        self.collection_name = collection_name

        print("[SİSTEM] Retriever modelleri belleğe alınıyor (CPU)...")

        jina = JinaEmbeddings()

        class _LCAdapter:
            """JinaEmbeddings'i LangChain embedding interface'ine sarar."""

            def embed_documents(self, texts):
                return jina._get_text_embeddings(texts)

            def embed_query(self, text):
                return jina._get_query_embedding(text)

        print("[SİSTEM] Jina Reranker v3 yükleniyor (CPU)...")
        self.reranker = AutoModel.from_pretrained(
            "./backend/models/jina-reranker-v3",
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.reranker.eval()
        print("[SİSTEM] Jina Reranker v3 başarıyla yüklendi.")

        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=_LCAdapter(),
            collection_name=self.collection_name,
        )

        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

    def get_relevant_context(self, query: str, top_n: int = 3, threshold: float = 0.0):

        raw_docs = self.base_retriever.invoke(query)
        if not raw_docs:
            return ""

        temp_time = time.time()

        documents = [doc.page_content for doc in raw_docs]

        # Listwise reranking — tüm belgeler tek seferde gönderiliyor
        results = self.reranker.rerank(query, documents, top_n=top_n)

        print(f"\nReranker süresi: {time.time() - temp_time:.2f} sn")

        # Threshold filtresi
        filtered = [r for r in results if r["relevance_score"] >= threshold]

        return "\n\n".join([r["document"].strip() for r in filtered])
