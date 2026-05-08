import time

from core.config import AppConfig
from core.vector_store import JinaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
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

    # ──────────────────────────────────────────────────────────────────────────
    # Komşu Node Genişletici
    # ──────────────────────────────────────────────────────────────────────────

    def _expand_with_neighbors(self, docs: list) -> list:
        """
        Retriever'ın döndürdüğü node'ları belge sırasına göre genişletir.

        Kural 1 — VLM node bulunduysa:
            Sol komşu (index-1) ve sağ komşu (index+1) da getirilir.
            Hedef: Başlık → VLM veya VLM → metin geçişlerinin korunması.
            Örnek: node_index=6 ("## OSI Referans Modeli") + node_index=7
                   (VLM şema) → ikisi birlikte LLM'e gider.

        Kural 2 — Metin node bulunduysa:
            Sadece sağ komşu (index+1) çekilir.
            Hedef: Metnin hemen ardından gelen VLM node'u da bağlama katmak.

        Sonuç liste node_index'e göre sıralanır ve tekrarlar temizlenir.
        Böylece LLM, belgedeki orijinal sırayı görür.
        """
        if not docs:
            return docs

        # Mevcut indeksleri kayıt altına al
        existing_indices: dict[int, LCDocument] = {}
        for doc in docs:
            idx = doc.metadata.get("node_index")
            if idx is not None:
                existing_indices[int(idx)] = doc

        # Hangi komşuları getireceğimizi belirle
        neighbor_indices: set[int] = set()
        for doc in docs:
            idx = doc.metadata.get("node_index")
            if idx is None:
                continue
            idx = int(idx)
            ntype = doc.metadata.get("node_type", "text")

            if ntype == "vlm":
                # Her iki komşu
                if idx - 1 >= 0:
                    neighbor_indices.add(idx - 1)
                neighbor_indices.add(idx + 1)
            else:
                # Sadece sağ komşu: yanındaki VLM olabilir
                neighbor_indices.add(idx + 1)

        # Zaten elimizde olanları çıkar
        to_fetch = [i for i in neighbor_indices if i not in existing_indices]

        if not to_fetch:
            return sorted(docs, key=lambda d: int(d.metadata.get("node_index", 9999)))

        # ChromaDB'den komşuları çek
        # node_index integer metadata olarak saklandığı için $in operatörü çalışır
        extra_docs: list[LCDocument] = []
        try:
            results = self.vectorstore._collection.get(
                where={"node_index": {"$in": to_fetch}},
                include=["documents", "metadatas"],
            )
            if results and results.get("documents"):
                for text, meta in zip(results["documents"], results["metadatas"]):
                    if text and meta is not None:
                        extra_docs.append(LCDocument(page_content=text, metadata=meta))
        except Exception as e:
            print(f"[UYARI] Komşu node'lar alınamadı: {e}")

        # Birleştir → tekilleştir → belge sırasına göre sırala
        all_docs = docs + extra_docs
        seen: set[int] = set()
        final: list[LCDocument] = []

        for doc in sorted(
            all_docs,
            key=lambda d: int(d.metadata.get("node_index", 9999)),
        ):
            idx = int(doc.metadata.get("node_index", id(doc)))
            if idx not in seen:
                seen.add(idx)
                final.append(doc)

        return final

    # ──────────────────────────────────────────────────────────────────────────
    # Ana Bağlam Getirici
    # ──────────────────────────────────────────────────────────────────────────

    def get_relevant_context(self, query: str, top_n: int = 3, threshold: float = 0.0):
        """
        Sorguya en uygun bağlamı döndürür.

        Akış:
            1. Embedding benzerliğiyle ilk 10 aday çek (base_retriever)
            2. Reranker ile yeniden sırala, top_n al
            3. Komşu node'larla genişlet (_expand_with_neighbors)
            4. VLM node'larına context_prefix ekle
            5. Birleşik metni döndür
        """
        raw_docs = self.base_retriever.invoke(query)
        if not raw_docs:
            return ""

        # ── Reranker ─────────────────────────────────────────────────────────
        temp_time = time.time()
        documents = [doc.page_content for doc in raw_docs]
        scores = self.reranker.rank(query, documents)
        print(f"\nReranker süresi: {time.time() - temp_time:.2f} sn")

        scored_docs = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
        best_docs = [doc for score, doc in scored_docs[:top_n] if score >= threshold]

        if not best_docs:
            return ""

        # ── Komşu genişletme ──────────────────────────────────────────────────
        expanded_docs = self._expand_with_neighbors(best_docs)

        # ── Bağlamı oluştur ───────────────────────────────────────────────────
        # VLM node ise context_prefix öne eklenir; böylece LLM görselin
        # hangi metnin ardından geldiğini de bilir.
        parts = []
        for doc in expanded_docs:
            content = doc.page_content.strip()
            if not content:
                continue

            prefix = doc.metadata.get("context_prefix", "")
            if prefix and doc.metadata.get("node_type") == "vlm":
                parts.append(f"[Önceki Bağlam: {prefix}]\n{content}")
            else:
                parts.append(content)

        return "\n\n---\n\n".join(parts)

    def unload(self):
        print("[SİSTEM] Reranker bellekten tahliye ediliyor...")
        del self.reranker
        print("[SİSTEM] Bellek temizlendi.")
