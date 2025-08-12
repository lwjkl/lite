import base64
import cv2
import io
import json
import numpy
import os
import warnings
import zipfile

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import (
    JSONResponse,
    StreamingResponse,
    FileResponse,
    PlainTextResponse,
)
from contextlib import asynccontextmanager

from main import App
from logger import logger

BASE = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app):
    logger.info("Starting app lifespan: initializing app instance.")
    app.state.app = App()
    yield
    app.state.app.save()


async def index_images_get(request):
    query = request.query_params
    image_directory = query.get("image_directory")
    wildcard = query.get("wildcard", "*.JPG")
    batch_size = int(query.get("batch_size", 64))

    if not image_directory:
        return JSONResponse({"detail": "Missing image_directory"}, status_code=400)

    return StreamingResponse(
        request.app.state.app.index_images_stream(
            image_directory, wildcard, batch_size
        ),
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

    if not image_directory:
        return JSONResponse({"detail": "Missing image_directory"}, status_code=400)

    return StreamingResponse(
        request.app.state.app.index_images_stream(
            image_directory, wildcard, batch_size
        ),
        media_type="text/event-stream",
    )


async def search_similar(request):
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

    result = request.app.state.app.search_similar_images(
        img, top_k=top_k, distance_threshold=distance_threshold, num_results=num_results
    )

    return JSONResponse(result)


async def stream_zip(request, image_ids: str):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        for image_id in image_ids:
            image_path = os.path.join(
                request.app.state.app.faiss_index.image_directory, image_id
            )
            if os.path.exists(image_path):
                zipf.write(image_path, arcname=os.path.basename(image_path))
    buffer.seek(0)
    yield buffer.read()


async def download_zip(request):
    body = await request.json()
    image_ids = body.get("image_ids", [])
    return StreamingResponse(
        stream_zip(request, image_ids),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=matched_images.zip"},
    )


async def get_image(request):
    image_id = request.path_params["image_id"]
    if not request.app.state.app.faiss_index.image_directory:
        return JSONResponse({"detail": "Image directory not loaded."}, status_code=500)

    image_path = os.path.join(
        request.app.state.app.faiss_index.image_directory, image_id
    )
    if not os.path.exists(image_path):
        return JSONResponse({"detail": "Image not found."}, status_code=404)
    return FileResponse(image_path)


async def index_status(request):
    if not request.app.state.app.faiss_index:
        return JSONResponse(
            {"detail": "FAISS index is not initialized."}, status_code=500
        )
    return JSONResponse(request.app.state.app.faiss_index.get_status())


async def gui_page(request):
    file_path = os.path.join(BASE, "gui.html")
    return FileResponse(file_path)


async def plot_embeddings(request):
    form = await request.form()
    index_path = form.get("index_path")
    metadata_path = form.get("metadata_path")

    try:
        examples_per_cluster = int(form.get("examples_per_cluster", 5))
    except ValueError:
        return PlainTextResponse("Invalid examples_per_cluster", status_code=400)

    logger.info("Generating plot...")
    app = request.app.state.app
    png_bytes_list = app.plot_embeddings(
        index_path, metadata_path, examples_per_cluster
    )
    logger.info("Complete generating plot...")

    response_data = {
        "plot1": base64.b64encode(png_bytes_list[0]).decode("utf-8"),
        "plot2": base64.b64encode(png_bytes_list[1]).decode("utf-8"),
    }

    return JSONResponse(response_data)


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
        Route("/gui", gui_page, methods=["GET"]),
    ],
)


if __name__ == "__main__":
    import uvicorn
    from settings import settings

    uvicorn.run("api:app", host="0.0.0.0", port=settings.port, reload=True)
