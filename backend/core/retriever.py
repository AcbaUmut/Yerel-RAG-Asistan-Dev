import time

from core.config import AppConfig
from core.vector_store import JinaEmbeddings
from langchain_chroma import Chroma
from llama_cpp.llama_cpp import LLAMA_POOLING_TYPE_RANK
from llama_cpp.llama_embedding import LlamaEmbedding


class RetrieverEngine:
    def __init__(self, collection_name: str = "tez_koleksiyonu"):
        self.persist_dir = "./backend/chroma_db"
        self.collection_name = collection_name

        print("[SİSTEM] Retriever modelleri belleğe alınıyor...")

        jina = JinaEmbeddings()

        class _LCAdapter:
            def embed_documents(self, texts):
                return jina._get_text_embeddings(texts)

            def embed_query(self, text):
                return jina._get_query_embedding(text)

        print("[SİSTEM] BGE Reranker GGUF yükleniyor...")
        self.reranker = LlamaEmbedding(
            model_path=f"./backend/models/{AppConfig.RERANKER_MODEL_NAME}",
            pooling_type=LLAMA_POOLING_TYPE_RANK,
            n_gpu_layers=0,
            n_ctx=0,
            n_batch=4096,
            n_ubatch=4096,
            verbose=False,
        )
        print("[SİSTEM] BGE Reranker GGUF başarıyla yüklendi.")

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
        scores = self.reranker.rank(query, documents)

        print(f"\nReranker süresi: {time.time() - temp_time:.2f} sn")

        scored_docs = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)

        best_docs = [doc for score, doc in scored_docs[:top_n] if score >= threshold]

        parts = []
        for doc in best_docs:
            content = doc.page_content.strip()
            prefix = doc.metadata.get("context_prefix", "")
            if prefix and doc.metadata.get("node_type") == "vlm":
                parts.append(f"[Bağlam: {prefix}]\n{content}")
            else:
                parts.append(content)

        return "\n\n".join(parts)

    def unload(self):
        print("[SİSTEM] Reranker bellekten tahliye ediliyor...")
        del self.reranker
        print("[SİSTEM] Bellek temizlendi.")
