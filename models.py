from fastapi import Form
from pydantic import BaseModel


class IndexRequest(BaseModel):
    image_directory: str
    wildcard: str = "*.JPG"
    batch_size: int = 64


class SearchRequest:
    def __init__(
        self,
        num_results: int = Form(9),
        download_ids: bool = Form(False),
        distance_threshold: float = Form(None)
    ):
        self.num_results = num_results
        self.download_ids = download_ids
        self.distance_threshold = distance_threshold