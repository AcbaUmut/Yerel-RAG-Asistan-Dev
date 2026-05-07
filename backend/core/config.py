class AppConfig:
    LLM_MODEL_NAME = "Turkish-Gemma-9b-T1-Q4_K_M.gguf"
    VLM_MODEL_NAME = "ZwZ-4B-Q4_K_M.gguf"
    VLM_MMPROJ_NAME = "mmproj-ZwZ-4B-Q8_0.gguf"
    # EMBED_MODEL_NAME = "jina-v5-nano"
    # RERANKER_MODEL_NAME = "bge-reranker-v2-m3"
    RERANKER_MODEL_NAME = "bge-reranker-v2-m3-Q8_0.gguf"
    # CPU 0, GPU -1
    # RERANKER_MODE = 0

    CHUNK_SIZE = 1100
    CHUNK_OVERLAP = 200

    RERANKER_TOP_N = 3

    LLM_N_CTX = 4096
    LLM_MAX_TOKENS = 1024
    LLM_TEMPERATURE = 0.1

    VLM_N_CTX = 4096
    VLM_MAX_TOKENS = 1024
    VLM_TEMPERATURE = 0.0
