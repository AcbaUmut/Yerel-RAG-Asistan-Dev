from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import LlamaCppEmbeddings


class RetrieverEngine:
    def __init__(
        self, persist_dir="./backend/chroma_db", collection_name="tez_koleksiyonu"
    ):
        """
        Modelleri sadece bu sınıf (class) çağrıldığında RAM'e yükler. Global israfı önler.
        """
        print("[SİSTEM] Retriever modelleri belleğe alınıyor (CPU)...")

        # Vektörleyici (Jina V5 Nano GGUF - CPU)
        jina_model_path = "./backend/models/jina-embeddings-v5-text-nano-retrieval-f16.gguf"  # Kendi dosya adına göre düzenle

        self.embeddings = LlamaCppEmbeddings(
            model_path=jina_model_path,
            n_ctx=8192,
            n_batch=512,
            device="cpu",
        )

        # Hakem (CPU)
        self.bge_model = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"device": "cpu"}
        )

        # Veritabanı Bağlantısı
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

    def get_relevant_context(self, query: str, top_n: int = 3, threshold: float = 0.0):
        # 1. Kaba Arama
        raw_docs = self.base_retriever.invoke(query)
        if not raw_docs:
            return (
                ""  # LLM string beklediği için boş liste değil, boş string dönüyoruz.
            )

        # 2. Hakeme Puanlat
        pairs = [[query, doc.page_content] for doc in raw_docs]
        scores = self.bge_model.score(pairs)

        # 3. Puanları Ekle ve Sırala
        for doc, score in zip(raw_docs, scores):
            doc.metadata["relevance_score"] = float(score)

        sorted_docs = sorted(
            raw_docs, key=lambda x: x.metadata["relevance_score"], reverse=True
        )
        filtered_docs = [
            doc for doc in sorted_docs if doc.metadata["relevance_score"] >= threshold
        ]

        # 4. LLM İçin Tek Parça Metne Çevir (Kritik Düzeltme)
        # Gemma listeleri değil, düz metni (string) okur. Dökümanları uç uca ekleyerek tek metin yapıyoruz.
        best_docs = filtered_docs[:top_n]
        context_string = "\n\n".join([doc.page_content.strip() for doc in best_docs])

        return context_string
