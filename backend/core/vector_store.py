import gc
import json
import os
from typing import List, Optional

import chromadb
import numpy as np
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


class JinaEmbeddings(BaseEmbedding):
    """
    Jina V5 Nano ONNX embedding sarmalayıcısı (numpy I/O).

    device:
        "cuda" → CUDAExecutionProvider, VRAM'e yüklenir.
        "cpu"  → CPUExecutionProvider, RAM'de kalır.

    PyTorch CUDA build bağımlılığı yok; ORT host↔device kopyayı yönetir.
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="jina-v5-nano-onnx")
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device 'cuda' veya 'cpu' olmalı, aldı: {device}")

        self._device = device
        self._model_dir = "./backend/models/jina-v5-nano"
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[ORTModelForFeatureExtraction] = None
        self._load()

    # ── Yükleme / Tahliye ────────────────────────────────────────────────
    def _load(self) -> None:
        print(f"[SİSTEM] Jina V5 Nano ONNX ({self._device.upper()}) yükleniyor...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_dir, trust_remote_code=True, local_files_only=True
        )

        if self._device == "cuda":
            provider = "CUDAExecutionProvider"
            provider_options = [
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }
            ]
        else:
            provider = "CPUExecutionProvider"
            provider_options = None

        self._model = ORTModelForFeatureExtraction.from_pretrained(
            self._model_dir,
            subfolder="onnx",
            file_name="model.onnx",
            provider=provider,
            provider_options=provider_options,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Sessiz CPU fallback kontrolü — ORT yanlış kuruluysa CUDA ister,
        # uyarı bile vermeden CPU'ya düşebilir.
        active = self._model.model.get_providers()
        if self._device == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"CUDA provider yüklenemedi. Aktif provider'lar: {active}\n"
                "Kontrol listesi:\n"
                "  1) pip uninstall onnxruntime onnxruntime-gpu -y\n"
                "  2) pip install onnxruntime-gpu\n"
                "  3) nvidia-smi → sürücü 525+ olmalı (CUDA 12.x için)\n"
                "  4) Windows'ta cuDNN bin klasörü PATH'e eklenmiş olmalı"
            )

        print(f"[SİSTEM] Jina V5 Nano ONNX hazır. Aktif provider: {active[0]}")

    def unload(self) -> None:
        """ORT InferenceSession'ı serbest bırakır; CUDA buffer'lar destructor'da temizlenir."""
        print(f"[SİSTEM] Jina embedding ({self._device.upper()}) tahliye ediliyor...")
        self._model = None
        self._tokenizer = None
        gc.collect()
        # Not: torch.cuda.empty_cache() ÇAĞIRILMIYOR.
        # ORT'un CUDA bellek arenası torch'tan ayrıdır; ORT session destructor'u
        # kendi buffer'larını serbest bırakır. del + gc yeterli.
        print("[SİSTEM] Jina embedding belleği temizlendi.")

    # ── Encode (numpy I/O) ───────────────────────────────────────────────
    def _encode(self, texts: List[str]) -> List[List[float]]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model tahliye edilmiş; tekrar JinaEmbeddings() oluştur."
            )

        # Tokenizer'dan numpy iste — torch tensor üretimini atla.
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="np",
        )

        # ORT InferenceSession'a doğrudan numpy ver.
        # CUDA provider aktifse host→device kopya ORT içinde yapılır.
        ort_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        outputs = self._model.model.run(None, ort_inputs)
        last_hidden = outputs[0]  # (batch, seq, hidden), numpy

        # Last-token pooling (resmi Jina kullanımı)
        attention_mask = enc["attention_mask"]
        seq_lengths = attention_mask.sum(axis=1) - 1
        batch_idx = np.arange(last_hidden.shape[0])
        embeddings = last_hidden[batch_idx, seq_lengths]

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        return embeddings.tolist()

    # ── BaseEmbedding kontratı ───────────────────────────────────────────
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode([f"Document: {text}"])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode([f"Query: {query}"])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._encode([f"Document: {t}" for t in texts])


class VectorStoreEngine:
    def __init__(
        self,
        persist_dir: str = "./backend/chroma_db",
        collection_name: str = "tez_koleksiyonu",
        embed_device: str = "cuda",  # ← yeni
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        print(f"[SİSTEM] VectorStoreEngine başlatılıyor... ({embed_device.upper()})")

        self.embed_model = JinaEmbeddings(device=embed_device)  # ← referansı tut
        Settings.embed_model = self.embed_model
        Settings.llm = None

        self.db_client = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.db_client.get_or_create_collection(
            self.collection_name
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _save_sections(self, parent_nodes: list, file_name: str) -> None:
        """
        Section parent node'larını JSON dosyasına kaydeder.

        Neden JSON, neden ChromaDB değil?
            Section parent'lar hiçbir zaman similarity araması ile bulunmaz.
            Erişim her zaman section_id ile yapılır: sections_map[section_id].
            Bu ID bazlı erişim için vektör veritabanı gerekmiyor; saf bir
            dict (JSON) aynı işi O(1) ile yapar, embedding maliyeti sıfır,
            HNSW indeksi oluşmaz.

        Dosya yapısı:
            { "<section_id>": {"text": "...", "metadata": {...}}, ... }
        """

        sections_file = os.path.join(self.persist_dir, "sections.json")

        # Mevcut dosyayı yükle (başka PDF'lerden gelen section'lar korunur)
        existing: dict = {}
        if os.path.exists(sections_file):
            with open(sections_file, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # Yeni section'ları ekle
        for node in parent_nodes:
            existing[node.node_id] = {
                "text": node.text,
                "metadata": node.metadata,
            }

        with open(sections_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def add_nodes(self, nodes, file_name: str) -> tuple[int, int]:
        """
        Dönüş: (child_count, section_count)
        """
        if not nodes:
            print("Uyarı: Eklenecek node bulunamadı.")
            return 0, 0

        child_nodes = [n for n in nodes if n.metadata.get("node_type") != "section"]
        parent_nodes = [n for n in nodes if n.metadata.get("node_type") == "section"]

        print(f"  {len(child_nodes)} child node embedding'leniyor...")
        VectorStoreIndex(nodes=child_nodes, storage_context=self.storage_context)

        # Parent'ları JSON'a kaydet — embedding hesabı yok, salt metin deposu
        if parent_nodes:
            print(f"  {len(parent_nodes)} section parent JSON'a kaydediliyor...")
            self._save_sections(parent_nodes, file_name)

        print(
            f"[SİSTEM] Kayıt tamamlandı: {len(child_nodes)} child → '{self.collection_name}'"
            f" | {len(parent_nodes)} section → sections.json"
        )
        return len(child_nodes), len(parent_nodes)

    def unload(self):
        """
        Jina embedding modelini VRAM'den serbest bırakır.

        Settings.embed_model global state olduğu için onu da temizliyoruz,
        yoksa lokal referanslar silinse bile model bellekte kalır.
        """
        print("[SİSTEM] Embedding modeli bellekten tahliye ediliyor...")
        # 1. Önce Jina'nın kendi unload'unu çağır — ORT session ve tokenizer
        # açıkça None'lanıyor, destructor garantili çalışıyor.
        try:
            jina = Settings.embed_model
            if jina is not None and hasattr(jina, "unload"):
                jina.unload()
        except Exception as e:
            print(f"[UYARI] Jina unload sırasında: {e}")

        # 2. LlamaIndex global state'i temizle
        try:
            Settings.embed_model = None
        except Exception:
            pass

        # 3. Diğer ağır referansları sil
        if hasattr(self, "vector_store"):
            del self.vector_store
        if hasattr(self, "storage_context"):
            del self.storage_context
        if hasattr(self, "chroma_collection"):
            del self.chroma_collection
        if hasattr(self, "db_client"):
            del self.db_client

        # Not: torch.cuda.empty_cache() çağrılmıyor — ChromaDB ve Jina ONNX
        # PyTorch GPU kullanmıyor. Jina'nın kendi unload'u zaten ORT
        # session'ı temizliyor.
        import gc

        gc.collect()

        print("[SİSTEM] Embedding belleği temizlendi.")


if __name__ == "__main__":
    print("Test için lütfen ingest.py dosyasını kullanınız.")
