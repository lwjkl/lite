import cv2
import io
import json
import numpy
import os
import time
import torch
import warnings
import zipfile
from glob import glob
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from starlette.routing import Route

from index import FaissIndex
from logger import logger
from plot import plot_embeddings
from settings import settings
from utils import get_embeder, embed


# === Globals ===
IMAGE_DIR = "images"
DOWNLOAD_DIR = "downloads"
BASE = os.path.dirname(os.path.abspath(__file__))


# --- Lifespan handler ---
@asynccontextmanager
async def lifespan(app):
    logger.info("Initializing FAISS index and ResNet model...")
    device_global = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faiss_index = FaissIndex(
        d=settings.faiss_dim, M=settings.faiss_m, index_path=settings.index_path
    )
    model = get_embeder(device_global)

    app.state.device = device_global
    app.state.faiss_index = faiss_index
    app.state.model = model
    logger.info("Initialization complete.")

    yield

    # Cleanup on shutdown
    if faiss_index:
        image_dir = getattr(faiss_index, "image_directory", None)
        if image_dir:
            faiss_index.save_index_and_meta(settings.index_path, image_dir)
            logger.info(f"FAISS index and metadata saved ({settings.index_path})")
        else:
            faiss_index.save(settings.index_path)
            logger.info(f"FAISS index saved without metadata ({settings.index_path})")


# --- Endpoints ---
async def index_images_get(request):
    query = request.query_params
    image_directory = query.get("image_directory")
    wildcard = query.get("wildcard", "*.JPG")
    batch_size = int(query.get("batch_size", 64))

    if not image_directory:
        return JSONResponse({"detail": "Missing image_directory"}, status_code=400)

    return StreamingResponse(
        index_images_stream(request.app.state, image_directory, wildcard, batch_size),
        media_type="text/event-stream",
    )


async def index_images_post(request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"detail": "Invalid JSON"}, status_code=400)

    image_directory = body.get("image_directory")
    wildcard = body.get("wildcard", "*.JPG")
    batch_size = int(body.get("batch_size", 64))

    return StreamingResponse(
        index_images_stream(request.app.state, image_directory, wildcard, batch_size),
        media_type="text/event-stream",
    )


def index_images_stream(state, image_directory: str, wildcard: str, batch_size: int):
    """
    Shared generator for both GET/POST progress streaming.
    """
    paths = glob(f"{image_directory}/{wildcard}")
    total_images = len(paths)
    if not paths:
        yield f"data: error - No images found in {image_directory}\n\n"
        return

    processed_images = 0
    for batch in [
        paths[i : i + batch_size] for i in range(0, total_images, batch_size)
    ]:
        vectors, ids = [], []
        for path in batch:
            try:
                uid = os.path.basename(path)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                vector = embed(img, state.model, state.device)
                vectors.append(vector)
                ids.append(uid)
            except Exception as e:
                logger.error(f"Error indexing {path}: {e}")
            finally:
                processed_images += 1
                progress = int((processed_images / total_images) * 100)
                yield f"data: {progress}\n\n"
                time.sleep(0.01)

        if vectors:
            state.faiss_index.insert(vectors, ids)

    state.faiss_index.save_index_and_meta(settings.index_path, image_directory)
    yield "data: done\n\n"


async def search_similar(request):
    """
    Search for similar images.
    """
    form = await request.form()
    file = form["file"]
    try:
        num_results = int(form.get("num_results", 9))
        top_k = int(form.get("top_k", 500))
        distance_threshold = float(form.get("distance_threshold", 2000))
    except ValueError:
        return JSONResponse({"detail": "Invalid numeric parameters"}, status_code=400)

    contents = await file.read()
    np_img = numpy.frombuffer(contents, numpy.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"detail": "Invalid image."}, status_code=400)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vector = embed(img, request.app.state.model, request.app.state.device)

    filtered_results = request.app.state.faiss_index.search(
        vector, k=top_k, distance_threshold=distance_threshold
    )

    results = [
        {"image_id": img_id, "distance": dist}
        for img_id, dist in filtered_results[:num_results]
    ]

    return JSONResponse(
        {
            "count_within_threshold": len(filtered_results),
            "results": results,
            "full_results": filtered_results,
        }
    )


async def stream_zip(request, image_ids):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        for image_id in image_ids:
            image_path = os.path.join(request.app.faiss_index.image_directory, image_id)
            if os.path.exists(image_path):
                zipf.write(image_path, arcname=os.path.basename(image_path))
            # Flush buffer for each file
            buffer.seek(0)
            chunk = buffer.read()
            yield chunk
            buffer.seek(0)
            buffer.truncate(0)


async def download_zip(request):
    """
    Download image as a zip file.
    """
    body = await request.json()
    image_ids = body.get("image_ids", [])
    return StreamingResponse(
        stream_zip(request, image_ids),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=matched_images.zip"},
    )


async def get_image(request):
    """
    Serve an image by its ID.
    """
    image_id = request.path_params["image_id"]
    if not request.app.state.faiss_index.image_directory:
        return JSONResponse({"detail": "Image directory not loaded."}, status_code=500)

    image_path = os.path.join(request.app.state.faiss_index.image_directory, image_id)
    if not os.path.exists(image_path):
        return JSONResponse({"detail": "Image not found."}, status_code=404)
    return FileResponse(image_path)


async def index_status(request):
    """
    Return current FAISS index status.
    """
    if not request.app.state.faiss_index:
        return JSONResponse(
            {"detail": "FAISS index is not initialized."}, status_code=500
        )
    return JSONResponse(request.app.state.faiss_index.get_status())


async def gui_page(request):
    file_path = os.path.join(BASE, "gui.html")
    return FileResponse(file_path)


async def plot_page(request):
    file_path = os.path.join(BASE, "plot.html")
    return HTMLResponse(open(file_path, "r").read())


async def root(request):
    logger.info("Local Image Similarity Service")
    return JSONResponse({"message": "Local Image Similarity Service"})


warnings.filterwarnings(
    "ignore",
    message="xFormers is not available",
    category=UserWarning,
    module="dinov2.layers",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# --- Starlette app ---
app = Starlette(
    debug=True,
    lifespan=lifespan,
    routes=[
        Route("/", root, methods=["GET"]),
        Route("/index-status", index_status, methods=["GET"]),
        Route("/image/{image_id}", get_image, methods=["GET"]),
        Route("/download-zip", download_zip, methods=["POST"]),
        Route("/search-similar", search_similar, methods=["POST"]),
        Route("/index-images", index_images_get, methods=["GET"]),
        Route("/index-images", index_images_post, methods=["POST"]),
        Route("/plot-embeddings", plot_embeddings, methods=["POST"]),
        Route("/plot", plot_page, methods=["GET"]),
        Route("/gui", gui_page),
    ],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=1234, reload=True)
