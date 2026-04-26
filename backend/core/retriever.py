from core.config import AppConfig  # YENİ: Merkezi Sinir Sistemini içe aktardık
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import LlamaCppEmbeddings


class RetrieverEngine:
    def __init__(self, collection_name: str = "tez_koleksiyonu"):
        """
        Modelleri sadece bu sınıf (class) çağrıldığında RAM'e yükler. Global israfı önler.
        """
        # Parametre gelmezse Config'den çek
        self.persist_dir = "./backend/chroma_db"
        self.collection_name = (
            collection_name  # Senin kararınla varsayılan olarak kaldı
        )

        print("[SİSTEM] Retriever modelleri belleğe alınıyor (CPU)...")

        # Vektörleyici (Jina V5 Nano GGUF - CPU)
        self.embeddings = LlamaCppEmbeddings(
            model_path=f"./backend/models/{AppConfig.EMBED_MODEL_NAME}",
            n_ctx=(
                AppConfig.EMBED_N_CTX if AppConfig.EMBED_N_CTX is not None else 8192
            ),  # Config'den çekildi (8192)
            n_batch=512,  # Config dışı, sabit tutuldu
            device="cpu",  # Donanım kısıtı, kesinlikle değişmez
        )

        # Hakem (CPU)
        self.bge_model = HuggingFaceCrossEncoder(
            model_name=AppConfig.RERANKER_MODEL_NAME,  # Config'den çekildi
            model_kwargs={"device": "cpu"},  # Donanım kısıtı, kesinlikle değişmez
        )

        # Veritabanı Bağlantısı
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
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

        # 4. LLM İçin Tek Parça Metne Çevir
        best_docs = filtered_docs[
            :top_n
        ]  # Config'den gelen top_n değeri ile dilimlendi
        context_string = "\n\n".join([doc.page_content.strip() for doc in best_docs])

        return context_string
