# lite

lite is a lightweight image similarity search application built with [FAISS](https://github.com/facebookresearch/faiss), using the [Dinov2](https://github.com/facebookresearch/dinov2) ViT-S/14 distilled model to generate image embeddings. The current focus is on running the application on a CPU-only instance.

## behaviour

Currently, the application only supports storing the index using HNSW, specifically the **IndexHNSWFlat**.

Once the index has been created, you will notice two additional files appear in the root project directory:

- index.faiss
- index.faiss.meta.json

These correspond to the serialized FAISS index (saved locally for persistence) and the metadata for the image directory, which encodes mappings between `image_name ↔ id`.

## install

To run the application, clone this repository to your machine, navigate into the project directory, recommnd using [uv](https://github.com/astral-sh/uv) and run:

```bash
uv sync
```

## run

You can run the application with make. Check the makefile for all the option. Check `config.py` file for configuration.

## usage

To index a directory, provide the full path to the image directory. Indexing may take some time depending on the size of your dataset. Once indexing is complete, you can start searching for similar images.

Two optional parameters are available when performing a search — you can adjust them based on your preferences:

- distance threshold - limits results to those within a certain distance.
- top-k result to search - the number of top results to return, sorted by distance.

Plotting the embeddings by providing the path to the index and metadata.
