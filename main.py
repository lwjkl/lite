import cv2
import numpy
import os
import time
import torch
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from glob import glob
from pydantic import BaseModel

from index import FaissIndex
from utils import get_embeder, embed
from settings import settings


# === Globals ===
faiss_index = None
resnet_model = None
device_global = None
IMAGE_DIR = "images"
DOWNLOAD_DIR = "downloads"


# Get the directory where main.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))


# === FastAPI App ===
app = FastAPI()
app.mount(
    "/ui", 
    StaticFiles(directory=current_dir, html=True), 
    name="frontend"
)


# === Data Models ===
class IndexRequest(BaseModel):
    image_directory: str
    wildcard: str = "*.JPG"
    batch_size: int = 64


class SearchRequest:
    def __init__(
        self,
        num_results: int = Form(8),
        download_ids: bool = Form(False),
        distance_threshold: float = Form(None)
    ):
        self.num_results = num_results
        self.download_ids = download_ids
        self.distance_threshold = distance_threshold


# === Helpers ===
def save_ids_file(image_ids: list[int]) -> str:
    """
    Save matched image IDs to a temp file and return file path.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", dir=DOWNLOAD_DIR)
    with open(tmp_file.name, "w") as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")
    return tmp_file.name


def send_json_file(data: dict, filename: str, background_tasks: BackgroundTasks) -> FileResponse:
    """
    Save data as a temporary JSON file and return as FileResponse.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json", dir=DOWNLOAD_DIR
    )
    tmp_file.write(json.dumps(data, indent=2))
    tmp_file.close()

    # Schedule deletion of the temp file
    background_tasks.add_task(os.unlink, tmp_file.name)

    return FileResponse(
        tmp_file.name,
        media_type="application/json",
        filename=filename
    )


# === Startup / Shutdown ===
@app.on_event("startup")
async def startup_event():
    global faiss_index, resnet_model, device_global

    print("Initializing FAISS index and ResNet model...")
    device_global = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faiss_index = FaissIndex(
        d=settings.faiss_dim,
        M=settings.faiss_m,
        index_path=settings.index_path
    )
    resnet_model = get_embeder(device_global)
    print("Initialization complete.")


@app.on_event("shutdown")
async def shutdown_event():
    if faiss_index:
        image_dir = getattr(faiss_index, "image_directory", None)
        if image_dir:
            faiss_index.save_index_and_meta(settings.index_path, image_dir)
            print(f"FAISS index and metadata saved ({settings.index_path})")
        else:
            # fallback if no image_directory is set
            faiss_index.save(settings.index_path)
            print(f"FAISS index saved without metadata ({settings.index_path})")


def get_image_from_directory(image_path: str, wildcard: str = "*.JPG"):
    images = glob(f"{image_path}/{wildcard}")
    return images


# === Endpoints ===
@app.get("/index-images")
async def index_images_get(
    image_directory: str = Query(..., description="Path to image directory"),
    wildcard: str = Query("*.JPG", description="File pattern"),
    batch_size: int = Query(64, description="Batch size for indexing")
):
    """
    Index images (GET version with query params for browser/EventSource).
    """
    return StreamingResponse(
        index_images_stream(image_directory, wildcard, batch_size),
        media_type="text/event-stream"
    )


@app.post("/index-images")
async def index_images_post(request: IndexRequest):
    """
    Index images (POST version with JSON body for API calls).
    """
    return StreamingResponse(
        index_images_stream(
            request.image_directory, request.wildcard, request.batch_size
        ),
        media_type="text/event-stream"
    )


def index_images_stream(image_directory: str, wildcard: str, batch_size: int):
    """
    Shared generator for both GET/POST progress streaming.
    """
    paths = glob(f"{image_directory}/{wildcard}")
    total_images = len(paths)
    if not paths:
        yield f"data: error - No images found in {image_directory}\n\n"
        return

    processed_images = 0
    for batch in [paths[i:i + batch_size] for i in range(0, total_images, batch_size)]:
        vectors, ids = [], []
        for path in batch:
            try:
                uid = os.path.basename(path)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                vector = embed(img, resnet_model, device_global)
                vectors.append(vector)
                ids.append(uid)
            except Exception as e:
                print(f"Error indexing {path}: {e}")
            finally:
                processed_images += 1
                progress = int((processed_images / total_images) * 100)
                yield f"data: {progress}\n\n"
                time.sleep(0.01)  # tiny delay

        if vectors:
            faiss_index.insert(vectors, ids)

    # Save the index and metadata at the end
    faiss_index.save_index_and_meta(settings.index_path, image_directory)
    yield "data: done\n\n"


@app.post("/search-similar")
async def search_similar(
    file: UploadFile = File(...),
    request: SearchRequest = Depends(SearchRequest),
):
    """
    Search for similar images.
    """
    contents = await file.read()
    np_img = numpy.frombuffer(contents, numpy.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    vector = embed(img, resnet_model, device_global)

    # Always search with top_n=500 internally
    top_n = 500
    filtered_results = faiss_index.search(vector, k=top_n, distance_threshold=request.distance_threshold)

    count_within_threshold = len(filtered_results)

    num_display = request.num_results or 8
    results = [
        {"image_id": img_id, "distance": dist}
        for img_id, dist in filtered_results[:num_display]
    ]

    return {
        "count_within_threshold": count_within_threshold,
        "results": results,
        "full_results": filtered_results
    }


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    """
    Serve an image by its ID.
    """
    if not faiss_index.image_directory:
        raise HTTPException(status_code=500, detail="Image directory not loaded.")
    
    image_path = os.path.join(faiss_index.image_directory, f"{image_id}")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_path)


@app.get("/index-status")
async def index_status():
    """
    Return current FAISS index status.
    """
    if not faiss_index:
        raise HTTPException(status_code=500, detail="FAISS index is not initialized.")

    total_vectors, unique_ids, has_duplicates = faiss_index.get_index_status()

    return {
        "message": faiss_index.status_message,
        "image_directory": faiss_index.image_directory,
        "total_vectors": total_vectors,
        "unique_ids": unique_ids,
        "has_duplicates": has_duplicates
    }


@app.get("/")
async def root():
    print("Local Image Similarity Service")
    return {"message": "Local Image Similarity Service"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=1234,
        reload=True,
    )
