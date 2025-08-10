import cv2
import faiss
import hdbscan
import io
import json
import matplotlib.pyplot as plt
import numpy
import os
import time
import torch
import umap

from matplotlib.lines import Line2D
from PIL import Image

from index import FaissIndex
from settings import settings
from utils import get_embeder, embed
from logger import logger


class App:
    def __init__(self):
        logger.info("Initializing...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.faiss_index = FaissIndex(
            d=settings.faiss_dim,
            M=settings.faiss_m,
            index_path=settings.index_path,
        )
        self.model = get_embeder(self.device)
        logger.info("Application initialized.")

    def embed_image(self, img):
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
        for batch in [paths[i : i + batch_size] for i in range(0, total_images, batch_size)]:
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

        self.faiss_index.save_index_and_meta(settings.index_path, image_directory)
        yield "data: done\n\n"

    def search_similar_images(self, img, top_k=500, distance_threshold=2000, num_results=9):
        """
        Search for similar images given an input image.
        """
        vector = self.embed_image(img)
        filtered_results = self.faiss_index.search(
            vector, k=top_k, distance_threshold=distance_threshold
        )
        results = [
            {"image_id": img_id, "distance": dist}
            for img_id, dist in filtered_results[:num_results]
        ]
        return {
            "count_within_threshold": len(filtered_results),
            "results": results,
            "full_results": filtered_results,
        }

    def save(self):
        """
        Save the index and metadata.
        """
        image_dir = getattr(self.faiss_index, "image_directory", None)
        if image_dir:
            self.faiss_index.save_index_and_meta(settings.index_path, image_dir)
            logger.info(f"FAISS index and metadata saved ({settings.index_path})")
        else:
            self.faiss_index.save(settings.index_path)
            logger.info(f"FAISS index saved without metadata ({settings.index_path})")

    
    def load_faiss_and_metadata(self, index_path, metadata_path, num_embeddings=-1):
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
        embeddings,
        cluster_umap_neighbors=15,
        cluster_umap_min_dist=0.0,
        cluster_umap_n_components=2,
        cluster_umap_random_state=42,
        hdbscan_min_samples=10,
        hdbscan_min_cluster_size=30,
        plot_umap_neighbors=None,
        plot_umap_min_dist=None,
        plot_umap_n_components=2,
        plot_umap_random_state=42,
        umap_n_jobs=1,
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

        return plot_embedding, labels
    
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

    def plot_matplotlib(self, standard_embedding, labels, image_paths, examples_per_cluster=5):
        unique_labels = numpy.unique(labels)
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
            axes = numpy.expand_dims(axes, axis=0)

        for row_idx, cluster_id in enumerate(sorted_clusters):
            mask = labels == cluster_id
            cluster_points = standard_embedding[mask]
            cluster_paths = numpy.array(image_paths)[mask]

            if cluster_points.shape[0] == 0:
                for col_idx in range(examples_per_cluster):
                    axes[row_idx, col_idx].axis("off")
                continue

            # Find centroid in UMAP space
            centroid = cluster_points.mean(axis=0)
            distances = numpy.linalg.norm(cluster_points - centroid, axis=1)

            # Closest images to centroid
            closest_indices = numpy.argsort(distances)[:examples_per_cluster]

            for col_idx, idx in enumerate(closest_indices):
                ax_img = axes[row_idx, col_idx]
                img = Image.open(cluster_paths[idx]).convert("RGB")
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

        fig_examples.tight_layout(rect=[0.08, 0, 1, 1])

        return fig_scatter, fig_examples
    
    def figures_to_png_bytes(self, figures, dpi=300):
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
        embeddings, _, image_paths = self.load_faiss_and_metadata(index_path, metadata_path)
        standard_embedding, labels = self.compute_umap_and_hdbscan(embeddings)
        fig_scatter, fig_examples = self.plot_matplotlib(
            standard_embedding, labels, image_paths, examples_per_cluster
        )
        png_bytes_list = self.figures_to_png_bytes([fig_scatter, fig_examples])
        return png_bytes_list