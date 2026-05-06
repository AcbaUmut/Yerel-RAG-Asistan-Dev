import time

from core.config import AppConfig
from core.vector_store import JinaEmbeddings
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class RetrieverEngine:
    def __init__(self, collection_name: str = "tez_koleksiyonu"):
        self.persist_dir = "./backend/chroma_db"
        self.collection_name = collection_name

        print("[SİSTEM] Retriever modelleri belleğe alınıyor (CPU)...")

        jina = JinaEmbeddings()

        class _LCAdapter:
            def embed_documents(self, texts):
                return jina._get_text_embeddings(texts)

            def embed_query(self, text):
                return jina._get_query_embedding(text)

        self.bge_model = HuggingFaceCrossEncoder(
            model_name=f"./backend/models/{AppConfig.RERANKER_MODEL_NAME}",
            model_kwargs={"device": "cpu"},
        )

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

        pairs = [[query, doc.page_content] for doc in raw_docs]
        scores = self.bge_model.score(pairs)

        print(f"\nReranker süresi: {time.time() - temp_time:.2f} sn")

        for doc, score in zip(raw_docs, scores):
            doc.metadata["relevance_score"] = float(score)

        sorted_docs = sorted(
            raw_docs, key=lambda x: x.metadata["relevance_score"], reverse=True
        )
        filtered_docs = [
            doc for doc in sorted_docs if doc.metadata["relevance_score"] >= threshold
        ]
        best_docs = filtered_docs[:top_n]
        return "\n\n".join([doc.page_content.strip() for doc in best_docs])
