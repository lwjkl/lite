# lite

lite is a lightweight image similarity search application built with [FAISS](https://github.com/facebookresearch/faiss), using the [Dinov2](https://github.com/facebookresearch/dinov2) ViT-S/14 distilled model to generate image embeddings. The main purpose of the application is to quickly sort through a large pool of images for data labeling. The current focus is on running the application on a CPU-only instance.

## behaviour

Currently, the application only supports storing the index using HNSW, specifically the **IndexHNSWFlat**.

Once the index has been created, you will notice two additional files appear in the root project directory:

- index.faiss
- index.faiss.meta.json

These correspond to the serialized FAISS index (saved locally for persistence) and the metadata for the image directory, which encodes mappings between `image_name ↔ id`.

Two optional parameters are available when performing a search — you can adjust them based on your preferences:

- distance threshold - limits results to those within a certain distance.
- top-k result to search -  the number of top results to return, sorted by distance.

## plot embeddings

Plot the embeddings with [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/). UMAP is helpful on reducing the embeddings to lower dimensional space while preserving local and global structure. HDBSCAN is particularly useful on identifying clusters in irregular shapes and detecting outliers.

## install

To run the application, clone this repository to your machine, navigate into the project directory, create a virtual environment, and install the dependencies with:

```bash
pip install -r requirements.txt
```

## run

You can run the application in several method:

Gradio

```bash
python gui.py
```

API:

```bash
python api.py
```

Cli

```bash
python cli.py
```

or as a python application:

```python
from main import App

app = App()
app.embed_image(...)
app.faiss_index.insert(...)
app.faiss_index.search(...)
app.faiss_index.save_index_and_meta(...)
```

Refer `settings.py` file for configuration.

## usage (gradio)

To index a directory, provide the full path to the image directory in the input field and click **Index Images**. Indexing may take some time depending on the size of your dataset.

Once indexing is complete, you can start searching for similar images by uploading an image at the right hand side panel.

Click the plot embeddings tab for plotting embeddings.

> [!NOTE]  
> You can launch the API and navigate to `/gui` endpoint to access the download result function.

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

To plot embeddings:

```bash
python cli.py plot --index_path /path/to/index.faiss --metadata_path /path/to/index.faiss.meta.json
```
