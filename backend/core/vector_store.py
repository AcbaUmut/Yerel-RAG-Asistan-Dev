import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def create_or_load_vector_store(
    nodes, persist_dir="./backend/chroma_db", collection_name="tez_koleksiyonu"
):
    print(f"Vektör veritabanı başlatılıyor... Kayıt dizini: {persist_dir}")

    # 1. Aşama: Embedding Modelinin Ayarlanması (HuggingFace - %100 Kontrol)
    # Nomic modelini doğrudan HF üzerinden indirip SADECE CPU'ya yüklüyoruz.
    # VRAM'i ana LLM modeline saklamak için device="cpu" ayarı kritik!
    # Not: trust_remote_code=True ayarı Nomic modeli için HuggingFace'te zorunludur.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="nomic-ai/nomic-embed-text-v2-moe",
        device="cpu",
        trust_remote_code=True,
    )

    # LLM'i şimdilik devre dışı bırakıyoruz.
    Settings.llm = None

    # 2. Aşama: ChromaDB İstemcisinin Başlatılması
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Aşama: Düğümleri Vektörlere Çevirip Veritabanına Yazma
    if nodes:
        print(f"Toplam {len(nodes)} düğüm vektör uzayına gömülüyor...")
        print(
            "Model ilk kez çalışıyorsa HuggingFace üzerinden indirilecektir (yaklaşık 500MB)."
        )

        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        print("İşlem Başarılı! Düğümler ChromaDB'ye kaydedildi.")
    else:
        print("Düğüm listesi boş, sadece var olan veritabanı yüklendi.")
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    return index


# --- Test Alanı ---
if __name__ == "__main__":
    from document_parser import parse_pdf_to_nodes

    test_dosyasi = "test.pdf"

    try:
        uretilen_dugumler = parse_pdf_to_nodes(test_dosyasi)
        index = create_or_load_vector_store(uretilen_dugumler)
        print(
            "\nSistem Kontrolü: Lütfen proje klasöründe 'chroma_db' adlı bir klasör oluştuğunu teyit edin."
        )

    except Exception as e:
        print(f"HATA OLUŞTU: {e}")
