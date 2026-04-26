import os


class AppConfig:
    # --- DİZİN VE YOL (PATH) AYARLARI ---
    # BASE_DIR, her zaman "backend" klasörünü işaret eder.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

    # --- MODEL YOLLARI VE İSİMLERİ ---
    LLM_MODEL_PATH = os.path.join(MODELS_DIR, "Turkish-Gemma-9b-T1-Q4_K_M.gguf")
    VLM_MODEL_PATH = os.path.join(MODELS_DIR, "ZwZ-4B-Q4_K_M.gguf")
    VLM_MMPROJ_PATH = os.path.join(MODELS_DIR, "mmproj-ZwZ-4B-F16.gguf")
    EMBED_MODEL_PATH = os.path.join(
        MODELS_DIR, "jina-embeddings-v5-text-nano-retrieval-f16.gguf"
    )

    RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    # Chunk Ayarları
    CHUNK_SIZE = 1100
    CHUNK_OVERLAP = 200

    # Embedding/Vektörleme (Jina) Ayarları
    EMBED_N_CTX = 8192

    # Reranker (BGE) Ayarları
    RERANKER_TOP_N = 3

    # Ana Dil Modeli (Gemma) Ayarları
    LLM_N_CTX = 4096
    LLM_MAX_TOKENS = 1024
    LLM_TEMPERATURE = 0.1

    # Görsel Dil Modeli (ZwZ-4B) Ayarları
    VLM_N_CTX = 4096
    VLM_MAX_TOKENS = 1024
    VLM_TEMPERATURE = 0.0
