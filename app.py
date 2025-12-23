import io
import json
import os
import time
import zipfile
from datetime import datetime

import cv2
import faiss
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D
from PIL import Image

from config import config
from index import FaissIndex
from logger import logger
from utils import embed, get_embeder


class App:
    def __init__(self):
        logger.info("Initializing...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.faiss_index = FaissIndex(
            d=config.faiss_dim,
            M=config.faiss_m,
            index_path=config.index_path,
        )
        self.model = get_embeder(self.device)
        self.results_dir = os.path.abspath(getattr(config, "results_dir", "results"))
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info("Application initialized.")

    def embed_image(self, img: np.ndarray):
        """
        Embed a single image.
        """
        return embed(img, self.model, self.device)

    def index_images_stream(self, image_directory: str, wildcard: str, batch_size: int):
        """
        Generator yielding progress for image indexing.
        """
        from glob import glob

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
                    vector = self.embed_image(img)
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
                self.faiss_index.insert(vectors, ids)

        self.faiss_index.save_index_and_meta(config.index_path, image_directory)
        yield "data: done\n\n"

    def search_similar_images(
        self,
        img: np.ndarray,
        top_k: int = 500,
        distance_threshold: int | float = 2000,
        num_results: int = 9,
    ):
        """
        Search for similar images given an input image.
        """
        vector = self.embed_image(img)
        results = self.faiss_index.search(
            vector, k=top_k, distance_threshold=distance_threshold
        )
        filtered_results = [
            {"image_id": img_id, "distance": dist}
            for img_id, dist in results[:num_results]
        ]

        # Add metadata to the results
        search_metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "top_k": top_k,
                "distance_threshold": distance_threshold,
                "num_results": num_results,
            },
            "index_info": {
                "image_directory": getattr(self.faiss_index, "image_directory", None),
                "total_indexed": self.faiss_index.faiss_index.ntotal
                if hasattr(self.faiss_index, "faiss_index")
                else 0,
            },
        }

        return {
            "count_within_threshold": len(results),
            "results": filtered_results,
            "full_results": results,
            "metadata": search_metadata,
        }

    def save_search_results(
        self, search_results: dict, results_name: str | None = None
    ):
        """
        Save search results to a JSON file.
        """
        if results_name is None:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{timestamp}.json"
        else:
            # Use custom name but ensure .json extension
            if not results_name.endswith(".json"):
                results_name += ".json"
            filename = results_name

        filepath = os.path.join(self.results_dir, filename)

        # Add save metadata
        save_data = {
            "saved_at": datetime.now().isoformat(),
            "filename": filename,
            "search_results": search_results,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Search results saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving search results: {e}")
            raise

    def get_image_path_from_id(self, image_id: str) -> str:
        """
        Helper function to get the full image path from its ID.
        """
        image_directory = getattr(self.faiss_index, "image_directory", None)
        if not image_directory:
            logger.error("Image directory not found in index metadata.")
            return ""
        return os.path.join(image_directory, image_id)

    def save_search_images_to_zip(
        self, search_results: dict, zip_name: str | None = None
    ):
        """
        Bundles and saves the images from search results into a zip file.
        """
        if not search_results or "results" not in search_results:
            logger.error("Invalid search results provided.")
            return None

        if zip_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"search_images_{timestamp}.zip"
        else:
            if not zip_name.endswith(".zip"):
                zip_filename += ".zip"
            zip_filename = zip_name

        zip_filepath = os.path.join(self.results_dir, zip_filename)

        # Check if the indexed images directory exists
        image_directory = getattr(self.faiss_index, "image_directory", None)
        if not image_directory or not os.path.isdir(image_directory):
            logger.error(
                f"Image directory '{image_directory}' not found. Cannot create zip file."
            )
            return None

        try:
            with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
                for result in search_results["full_results"]:
                    image_id = result[0]
                    source_path = self.get_image_path_from_id(image_id)

                    if os.path.exists(source_path):
                        # Add file to the zip archive
                        zipf.write(source_path, arcname=os.path.basename(source_path))
                        logger.info(f"Added {image_id} to {zip_filename}")
                    else:
                        logger.warning(f"Image file not found at {source_path}")

            logger.info(f"Successfully created zip file at {zip_filepath}")
            return zip_filepath

        except Exception as e:
            logger.error(f"Error creating zip file: {e}")
            raise

    def save(self):
        """
        Save the index and metadata.
        """
        image_dir = getattr(self.faiss_index, "image_directory", None)
        if image_dir:
            self.faiss_index.save_index_and_meta(config.index_path, image_dir)
            logger.info(f"FAISS index and metadata saved ({config.index_path})")
        else:
            self.faiss_index.save(config.index_path)
            logger.info(f"FAISS index saved without metadata ({config.index_path})")

    def load_faiss_and_metadata(
        self, index_path: str, metadata_path: str, num_embeddings: int = -1
    ):
        index = faiss.read_index(index_path)
        metadata = json.load(open(metadata_path))
        image_root_path = metadata.get("image_directory")
        id_to_path = metadata.get("int_to_str")

        total_vecs = index.ntotal
        if num_embeddings == -1 or num_embeddings > total_vecs:
            num_embeddings = total_vecs
        embeddings = index.index.reconstruct_n(0, num_embeddings)

        external_ids = faiss.vector_to_array(index.id_map)[:num_embeddings]
        image_paths = [
            os.path.join(image_root_path, id_to_path.get(str(i))) for i in external_ids
        ]

        return embeddings, external_ids, image_paths

    def compute_umap_and_hdbscan(
        self,
        embeddings: np.ndarray,
        cluster_umap_neighbors: int = 15,
        cluster_umap_min_dist: int | float = 0.0,
        cluster_umap_n_components: int = 2,
        cluster_umap_random_state: int = 42,
        hdbscan_min_samples: int = 10,
        hdbscan_min_cluster_size: int = 30,
        plot_umap_neighbors: int | None = None,
        plot_umap_min_dist: int | float | None = None,
        plot_umap_n_components: int = 2,
        plot_umap_random_state: int = 42,
        umap_n_jobs: int = 1,
    ):
        """
        Reduce embeddings with UMAP for clustering then cluster with HDBSCAN.
        """
        clusterable_embedding = umap.UMAP(
            n_neighbors=cluster_umap_neighbors,
            min_dist=cluster_umap_min_dist,
            n_components=cluster_umap_n_components,
            random_state=cluster_umap_random_state,
            n_jobs=umap_n_jobs,
        ).fit_transform(embeddings)

        labels = hdbscan.HDBSCAN(
            min_samples=hdbscan_min_samples, min_cluster_size=hdbscan_min_cluster_size
        ).fit_predict(clusterable_embedding)

        plot_embedding = umap.UMAP(
            n_neighbors=plot_umap_neighbors
            if plot_umap_neighbors is not None
            else cluster_umap_neighbors,
            min_dist=plot_umap_min_dist
            if plot_umap_min_dist is not None
            else cluster_umap_min_dist,
            n_components=plot_umap_n_components,
            random_state=plot_umap_random_state,
            n_jobs=umap_n_jobs,
        ).fit_transform(embeddings)

        return plot_embedding.toarray(), labels

    def set_style(self):
        plt.rcParams.update(
            {
                # Fonts
                "font.family": "serif",
                "font.serif": [
                    "DejaVu Serif",
                    "Times New Roman",
                    "Georgia",
                    "Palatino",
                    "serif",
                ],
                "font.size": 12,
                # Figure
                "figure.figsize": (8, 6),
                "figure.dpi": 100,
                "figure.titlesize": "large",
                # Axes
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "axes.grid": True,
                "axes.grid.which": "major",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 1,
                # Grid
                "grid.color": "0.85",
                "grid.linestyle": "-",
                # Ticks
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.major.size": 5,
                "ytick.major.size": 5,
                "xtick.minor.size": 2.5,
                "ytick.minor.size": 2.5,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                # Lines
                "lines.linewidth": 1.5,
                "lines.markersize": 6,
                # Legend
                "legend.fontsize": 11,
                "legend.frameon": False,
                # Savefig
                "savefig.dpi": 300,
                "savefig.format": "png",
                # Colors
                "axes.prop_cycle": plt.cycler(
                    color=[
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                        "#e377c2",
                        "#7f7f7f",
                        "#bcbd22",
                        "#17becf",
                    ]
                ),
            }
        )

    def plot_matplotlib(
        self,
        standard_embedding: np.ndarray,
        labels: np.ndarray,
        image_paths: list[str],
        examples_per_cluster: int = 5,
    ):
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("Spectral", len(unique_labels))
        noise_color = (0.3, 0.3, 0.3, 0.4)

        # Assign colors to clusters
        label_to_color = {}
        color_index = 0
        for label in unique_labels:
            if label == -1:
                label_to_color[label] = noise_color
            else:
                label_to_color[label] = cmap(color_index)
                color_index += 1
        colors = [label_to_color[label] for label in labels]

        # --- Scatter plot ---
        self.set_style()
        fig_scatter, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            standard_embedding[:, 0],
            standard_embedding[:, 1],
            c=colors,
            s=10,
            edgecolor="white",
            alpha=0.8,
            linewidth=0.2,
        )

        # Cluster IDs in scatter plot
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            cx = standard_embedding[mask, 0].mean()
            cy = standard_embedding[mask, 1].mean()
            ax.text(
                cx,
                cy,
                str(cluster_id),
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6),
            )

        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title("UMAP Projection of Embeddings with HDBSCAN Clusters")

        # Legend
        handles = []
        for label in unique_labels:
            color = label_to_color[label]
            if label == -1:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=(0.3, 0.3, 0.3, 1.0),
                        markersize=5,
                        label="Noise",
                    )
                )
            else:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=5,
                        label=f"Cluster {label}",
                    )
                )

        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            markerscale=2,
        )

        ax.set_aspect("equal", adjustable="datalim")
        fig_scatter.tight_layout()

        # --- Cluster examples plot ---
        sorted_clusters = sorted(unique_labels, key=lambda x: (x == -1, x))
        n_clusters = len(sorted_clusters)

        fig_examples, axes = plt.subplots(
            n_clusters,
            examples_per_cluster,
            figsize=(examples_per_cluster * 2, n_clusters * 2),
        )

        if n_clusters == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, cluster_id in enumerate(sorted_clusters):
            mask = labels == cluster_id
            cluster_points = standard_embedding[mask]
            cluster_paths = np.array(image_paths)[mask]

            if cluster_points.shape[0] == 0:
                for col_idx in range(examples_per_cluster):
                    axes[row_idx, col_idx].axis("off")
                continue

            # Find centroid in UMAP space
            centroid = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)

            # Closest images to centroid
            closest_indices = np.argsort(distances)[:examples_per_cluster]

            for col_idx, idx in enumerate(closest_indices):
                cur_image_path = str(cluster_paths[idx])
                ax_img = axes[row_idx, col_idx]
                img = Image.open(cur_image_path).convert("RGB")
                ax_img.imshow(img)
                ax_img.axis("off")

            # Cluster label
            label_text = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            fig_examples.text(
                0.005,
                1 - (row_idx + 0.5) / n_clusters,
                label_text,
                fontsize=10,
                weight="bold",
                va="center",
            )

        fig_examples.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))

        return fig_scatter, fig_examples

    def figures_to_png_bytes(self, figures: list, dpi: int = 300):
        """
        Convert one or multiple Matplotlib figures to PNG bytes.
        """
        if not isinstance(figures, (list, tuple)):
            figures = [figures]

        png_bytes_list = []
        for fig in figures:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            png_bytes_list.append(buf.read())
            plt.close(fig)
        return png_bytes_list

    def plot_embeddings(
        self,
        index_path: str,
        metadata_path: str,
        examples_per_cluster: int = 5,
    ):
        embeddings, _, image_paths = self.load_faiss_and_metadata(
            index_path, metadata_path
        )

        standard_embedding, labels = self.compute_umap_and_hdbscan(embeddings)

        fig_scatter, fig_examples = self.plot_matplotlib(
            standard_embedding, labels, image_paths, examples_per_cluster
        )

        png_bytes_list = self.figures_to_png_bytes([fig_scatter, fig_examples])
        return png_bytes_list
