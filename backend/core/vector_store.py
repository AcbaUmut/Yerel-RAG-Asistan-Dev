import chromadb
from core.config import AppConfig  # YENİ: Merkezi Sinir Sistemini içe aktardık
from langchain_community.embeddings import LlamaCppEmbeddings
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class VectorStoreEngine:
    def __init__(
        self,
        collection_name: str = "tez_koleksiyonu",  # Senin kararın: Varsayılan olarak kaldı
    ):
        """
        Vektör veritabanı motorunu ve yerleştirme (embedding) modelini CPU üzerinde başlatır.
        """
        self.persist_dir = AppConfig.CHROMA_DB_DIR
        self.collection_name = collection_name

        print(
            f"[SİSTEM] VectorStoreEngine başlatılıyor... Kayıt dizini: {self.persist_dir}"
        )

        print("[SİSTEM] Jina V5 Nano GGUF (CPU) modeli başlatılıyor...")
        lc_embed_model = LlamaCppEmbeddings(
            model_path=AppConfig.EMBED_MODEL_PATH,  # Config'den çekildi
            n_ctx=AppConfig.EMBED_N_CTX,  # Config'den çekildi (8192)
            n_batch=512,  # Config dışı, sabit tutuldu
            device="cpu",  # Donanım kısıtı, değişmez
        )

        # Langchain embedding'ini LlamaIndex'in anlayacağı formata çeviriyoruz
        Settings.embed_model = LangchainEmbedding(lc_embed_model)
        Settings.llm = None

        # 2. Aşama: ChromaDB İstemcisinin Başlatılması
        self.db_client = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.db_client.get_or_create_collection(
            self.collection_name
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def add_nodes(self, nodes, file_name: str):
        """
        Düğümleri Jina ile sayısal vektörlere çevirip ChromaDB'ye yazar.
        Çakışmaları önlemek için önce dosyanın eski kayıtlarını temizler.
        """
        if not nodes:
            print("Uyarı: Veritabanına eklenecek düğüm (node) bulunamadı.")
            return None

        # --- EKLENEN CRUD (SİLME) MANTIĞI ---
        try:
            # Metadata içindeki 'file_name' anahtarına göre eski kayıtları bul ve sil
            self.chroma_collection.delete(where={"file_name": file_name})
            print(
                f"[SİSTEM] Eski '{file_name}' kayıtları ChromaDB'den başarıyla temizlendi."
            )
        except Exception:
            # Eğer koleksiyon yepyeni ise veya bu dosya daha önce hiç yüklenmediyse burası sessizce geçilir.
            pass
        # ------------------------------------

        print(f"Toplam {len(nodes)} düğüm vektör uzayına gömülüyor...")

        # Vektörleme ve diske yazma işlemi burada gerçekleşir
        index = VectorStoreIndex(nodes=nodes, storage_context=self.storage_context)

        print("İşlem Başarılı! Düğümler ChromaDB'ye kaydedildi.")
        return index


if __name__ == "__main__":
    print("Test için lütfen ingest.py dosyasını kullanınız.")
