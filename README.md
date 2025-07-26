# lite

lite is a simple image similarity search application built with [FAISS](https://github.com/facebookresearch/faiss) and [FastAPI](https://github.com/fastapi/fastapi), using the [Dinov2](https://github.com/facebookresearch/dinov2) ViT-S/14 distilled model to generate image embeddings. The main purpose of the application is to quickly sort through a large pool of images for data labeling. A clean UI is provided for both indexing and searching. The current focus is on running the application on a CPU-only instance, particularly for the ann index and feature extraction model.

## behaviour

Currently, the application only supports storing the index using HNSW, specifically the **IndexHNSWFlat**.

Once the index has been created, you will notice two additional files appear in the root project directory:

- index.faiss
- index.faiss.meta.json

These correspond to the serialized FAISS index (saved locally for persistence) and the metadata for the image directory, which encodes mappings between `image_name ↔ id`.

Two optional parameters are available when performing a search — you can adjust them ased on your preferences:

- distance threshold - limits results to those within a certain distance.
- top-k result to search -  the number of top results to return, sorted by distance.

## run (CPU)

To run the application, clone this repository to your machine, navigate into the project directory, create a virtual environment, and install the dependencies with:

```bash
pip install -r requirements.txt
```

You can launch the application by running (server binds to 0.0.0.0:1234 by default):

```bash
python main.py
```

## usage (ui)

To access the UI, open your browser and go to `localhost:1234/ui`.

To index a directory, provide the full path to the image directory in the input field and click **Index Image**. Indexing may take some time depending on the size of your dataset.

Once indexing is complete, you can start searching for similar images by uploading a query image using the **Upload Query Image** input field.

Downloading the results,

- To download results as JSON, check the Download Results as JSON box before searching.
- To download raw images, after clicking the Search button, the top 9 most similar images will be displayed on the right panel. An option will be available below them to download the images as a ZIP file.

## usage (cli)

You can also use the application via the command-line interface.

To index a directory:

```bash
python cli.py index --dir /path/to/your/image/directory
```

To check index status:

```bash
python cli.py status
```

To search for a similar image:

```bash
python cli.py search --image /path/to/query/image.jpg
```

To search with threshold and top_k override:

```bash
python cli.py search --image /path/to/query/image.jpg --threshold 1000 --top_k 100
```
