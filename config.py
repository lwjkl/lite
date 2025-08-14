class Config:
    index_path = "index.faiss"
    faiss_dim = 384
    faiss_m = 32
    batch_size = 64
    log_file_path = "api.log"
    host = "0.0.0.0"
    port = 1234
    reload = False


config = Config()
