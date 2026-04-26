import os


class AppConfig:
    # --- 1. DİZİN VE YOL (PATH) AYARLARI ---
    # BASE_DIR, her zaman "backend" klasörünü işaret eder.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

    # --- 2. VERİTABANI AYARLARI ---
    COLLECTION_NAME = "tez_koleksiyonu"

    # --- 3. MODEL YOLLARI ---
    LLM_MODEL_PATH = os.path.join(MODELS_DIR, "Turkish-Gemma-9b-T1-Q4_K_M.gguf")
    VLM_MODEL_PATH = os.path.join(MODELS_DIR, "ZwZ-4B-Q4_K_M.gguf")
    VLM_MMPROJ_PATH = os.path.join(MODELS_DIR, "mmproj-ZwZ-4B-F16.gguf")
    EMBED_MODEL_PATH = os.path.join(
        MODELS_DIR, "jina-embeddings-v5-text-nano-retrieval-f16.gguf"
    )

    # Hakem modeli HuggingFace üzerinden çalıştığı için yerel yol değil, ismini tutuyoruz
    RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    # --- 4. DONANIM VE ORKESTRASYON LİMİTLERİ (Tatlı Noktalar) ---
    CHUNK_SIZE = 1100
    CHUNK_OVERLAP = 200
    HF_THRESHOLD = 0.6  # Otonom üstbilgi temizleyici eşiği

    LLM_N_CTX = 4096
    LLM_MAX_TOKENS = 1024

    VLM_N_CTX = 4096

    EMBED_N_CTX = 8192
    BATCH_SIZE = 512
