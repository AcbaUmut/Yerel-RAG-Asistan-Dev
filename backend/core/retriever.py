from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

# --- GLOBAL MODEL YÜKLEME (Sadece bir kez çalışır) ---
print("Sistem modelleri belleğe alınıyor (CPU)...")

# Vektörleyici (CPU)
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v2-moe",
    model_kwargs={"device": "cpu", "trust_remote_code": True},
)

# Hakem (CPU)
bge_model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"device": "cpu"}
)

# Veritabanı Bağlantısı
vectorstore = Chroma(
    persist_directory="./backend/chroma_db",
    embedding_function=embeddings,
    collection_name="tez_koleksiyonu",
)

# Temel Geri Çağırıcı
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


def get_relevant_context(query: str, top_n: int = 3, threshold: float = 0.0):
    """
    Kullanıcı sorgusuna en yakın, doğrulanmış ve sıralanmış dökümanları döndürür.

    Args:
        query (str): Kullanıcının sorusu.
        top_n (int): Maksimum kaç döküman dönecek?
        threshold (float): Alakalılık puanı bu değerden düşük olanlar elenir.

    Returns:
        list: Sıralanmış ve filtrelenmiş döküman listesi.
    """

    # 1. Adım: Veritabanından kaba arama yap
    raw_docs = base_retriever.invoke(query)

    if not raw_docs:
        return []

    # 2. Adım: Hakeme (Reranker) puanlat
    pairs = [[query, doc.page_content] for doc in raw_docs]
    scores = bge_model.score(pairs)

    # 3. Adım: Puanları işle ve dökümanlara ekle
    for doc, score in zip(raw_docs, scores):
        doc.metadata["relevance_score"] = float(score)

    # 4. Adım: Sırala
    sorted_docs = sorted(
        raw_docs, key=lambda x: x.metadata["relevance_score"], reverse=True
    )

    # 5. Adım: Eşik Değeri (Threshold) Filtrelemesi
    # Belirlediğin puanın altındakiler Gemma'yı kirletmesin diye elenir
    filtered_docs = [
        doc for doc in sorted_docs if doc.metadata["relevance_score"] >= threshold
    ]

    # 6. Adım: En iyi N sonucu döndür
    return filtered_docs[:top_n]


# --- KENDİ BAŞINA ÇALIŞTIRMA TESTİ ---
if __name__ == "__main__":
    test_query = (
        "Charles Babbage'ın tasarladığı makinenin adı nedir ve temel amacı neydi?"
    )
    print(f"\n[TEST] Sorgu: {test_query}")

    # Fonksiyonu çağırıyoruz
    results = get_relevant_context(test_query, top_n=3, threshold=0.0)

    print("\n" + "=" * 50)
    print(" DOĞRULANMIŞ VE FİLTRELENMİŞ SONUÇLAR ")
    print("=" * 50)

    if not results:
        print("Belirtilen eşik değerini geçen bir sonuç bulunamadı.")
    else:
        for i, doc in enumerate(results):
            score = doc.metadata.get("relevance_score", 0.0)
            print(f"\n[Sıra {i + 1}] Skor: {score:.4f}")
            print(f"Kaynak: {doc.metadata.get('file_name', 'Bilinmiyor')}")
            print(f"İçerik: {doc.page_content.strip()}")
            print("-" * 20)
