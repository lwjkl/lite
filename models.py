from fastapi import Form
from pydantic import BaseModel
from typing import Optional, List
from typing_extensions import TypedDict


class IndexRequest(BaseModel):
    image_directory: str
    wildcard: str = "*.JPG"
    batch_size: int = 64


class SearchRequest(BaseModel):
    num_results: int
    download_ids: bool
    distance_threshold: Optional[float]
    top_k: Optional[int]
    download_zip: bool 

    @classmethod
    def as_form(
        cls,
        num_results: int = Form(9),
        download_ids: bool = Form(False),
        distance_threshold: Optional[float] = Form(None),
        top_k: Optional[int] = Form(500),
        download_zip: bool = Form(False),
    ):
        return cls(
            num_results=num_results,
            download_ids=download_ids,
            distance_threshold=distance_threshold,
            top_k=top_k,
            download_zip=download_zip
        )
    

class DownloadZipRequest(BaseModel):
    image_ids: List[str]


class RootResponse(TypedDict):
    message: str

