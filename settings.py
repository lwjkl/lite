from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    index_path: str = "index.faiss"
    faiss_dim: int = 384
    faiss_m: int = 32
    batch_size: int = 64

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
