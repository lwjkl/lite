import cv2
import numpy as np
import pytest
from unittest.mock import patch

from lite.app import App
from config import config


@pytest.fixture(scope="module")
def app():
    # Patch get_embeder so it doesn't load a real model 
    # Return a dummy model that outputs a fixed-size random vector
    dummy_dim = config.faiss_dim
    with patch("lite.app.get_embeder", return_value=lambda device=None: None), \
         patch("lite.app.embed", side_effect=lambda img, model, device: np.random.rand(dummy_dim).astype(np.float32)):
        test_app = App()
        yield test_app


def test_embedding(app):
    # Create a dummy RGB image (e.g., 224x224 with random pixels)
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    embedding = app.embed_image(dummy_img)

    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    # Check embedding shape
    assert embedding.shape[0] == app.faiss_index.d


def create_dummy_images(directory, count=5):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        path = directory / f"img_{i}.jpg"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


def test_index_and_search_via_app(app, tmp_path):
    # Create dummy images
    img_dir = tmp_path / "images"
    create_dummy_images(img_dir, count=5)

    # Index images using the streaming generator
    progress_updates = list(app.index_images_stream(str(img_dir), "*.jpg", batch_size=2))
    assert any("data:" in p for p in progress_updates)
    assert any("done" in p for p in progress_updates)

    # Search for similar images using a dummy query
    query_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    results = app.search_similar_images(query_img, top_k=5, distance_threshold=1000, num_results=3)

    assert "results" in results
    assert isinstance(results["results"], list)
    assert len(results["results"]) <= 3
    assert all("image_id" in r and "distance" in r for r in results["results"])

