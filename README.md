# lite

lite is a simple image similarity search application built with [FAISS](https://github.com/facebookresearch/faiss) and [FastAPI](https://github.com/fastapi/fastapi), using the [Dinov2](https://github.com/facebookresearch/dinov2) ViT-S/14 distilled model to generate image embeddings. The main purpose of the application is to quickly sort through a large pool of images for data labeling. A clean UI is provided for both indexing and searching. The current focus is on running the application on a CPU-only instance, particularly for the ann index and feature extraction model.

## quick start (CPU)

To run the application, clone this repository to your machine, navigate into the project directory, create a virtual environment, and install the dependencies with:

```bash
pip install -r requirements.txt
```

You can start the application by running (default on port 1234):

```bash
python main.py
```

## usage

To access the UI, open your browser and go to `localhost:1234/ui`.

First, you need to index your image pool before performing any search. Provide the full path to the image directory in the input field and click **Index Image**. Indexing may take some time depending on the size of your dataset.

Once indexing is complete, you can start searching for similar images by uploading a query image using the **Upload Query Image** input field. Two optional parameters are available—you may experiment with them based on your preferences:

- distance threshold - limits results to those within a certain distance.
- top-k result to search -  the number of top results to return, sorted by distance.

Downloading the Results,

- To download results as JSON, check the Download Results as JSON box before searching.
- To download raw images, after clicking the Search button, the top 9 most similar images will be displayed on the right panel. An option will be available below them to download the images as a ZIP file.
