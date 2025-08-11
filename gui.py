import io
import gradio as gr
import numpy as np
from PIL import Image

from main import App


class View:
    def __init__(self):
        self.app = App()

    def refresh_status_handler(self):
        """Fetch and display FAISS index status."""
        try:
            status = self.app.faiss_index.get_status()
            return (
                f"Message: {status['message']}\n"
                f"Image Directory: {status['image_directory']}\n"
                f"Total Vectors: {status['total_vectors']}\n"
                f"Unique IDs: {status['unique_ids']}\n"
                f"Has Duplicates: {status['has_duplicates']}"
            )
        except Exception as e:
            return f"Error retrieving status: {e}"

    def index_images_handler(self, image_dir, wildcard, batch_size):
        """
        Generator handler to run indexing and stream progress into Index Status box.
        Yields progress updates as strings so Gradio will update the textbox live.
        """
        if not image_dir:
            yield "Error: image directory is empty."
            return
        try:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ValueError()
        except Exception:
            yield "Error: batch size must be a positive integer."
            return

        # Start indexing generator
        try:
            gen = self.app.index_images_stream(image_dir, wildcard, batch_size)
        except Exception as e:
            yield f"Error starting indexing: {e}"
            return

        # Iterate and yield progress lines
        try:
            for item in gen:
                # generator yields strings like "data: 23\n\n" or "data: done\n\n"
                if isinstance(item, bytes):
                    item = item.decode("utf-8", errors="ignore")
                text = str(item).strip()
                # Remove any "data:" prefix and extra whitespace/newlines
                if text.startswith("data:"):
                    text = text[len("data:") :].strip()
                # Normalize common messages
                if text.lower().startswith("error"):
                    yield f"Error during indexing: {text}"
                elif text.lower() == "done":
                    yield "Indexing: done (finalizing...)"
                else:
                    # If numeric, show percent
                    try:
                        pct = int(text)
                        yield f"Indexing progress: {pct}%"
                    except Exception:
                        yield f"Indexing: {text}"
            # after generator completes, refresh status
            yield "Indexing finished. Fetching index status..."
            yield self.refresh_status_handler()
        except Exception as e:
            yield f"Indexing failed: {e}"

    def search_handler(self, image, distance_threshold, top_k):
        """Search similar images."""
        if image is None:
            return None
        try:
            distance_threshold = float(distance_threshold)
            top_k = int(top_k)
        except ValueError:
            return None

        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        results = self.app.search_similar_images(
            img_np, top_k=top_k, distance_threshold=distance_threshold, num_results=9
        )

        images_to_display = [
            (
                f"{self.app.faiss_index.image_directory}/{r['image_id']}",
                f"{r['distance']:.2f}",
            )
            for r in results["results"]
        ]

        return images_to_display

    def plot_embeddings_handler(self, index_path, metadata_path, examples_per_cluster):
        """Generate embeddings plots and return as list for gallery."""
        try:
            examples_per_cluster = int(examples_per_cluster)
        except ValueError:
            examples_per_cluster = 5

        png_bytes_list = self.app.plot_embeddings(
            index_path=index_path,
            metadata_path=metadata_path,
            examples_per_cluster=examples_per_cluster,
        )

        if not png_bytes_list:
            print("No images returned from plot_embeddings()")
            return []

        images = [
            Image.open(io.BytesIO(png_bytes)).copy() for png_bytes in png_bytes_list
        ]

        return images

    def launch(self):
        with gr.Blocks(
            css="""
            .main-container {
                max-width: 900px;
                margin: auto;
                padding: 20px;
            }
        """
        ) as demo:
            # === PAGE 1: Search Tab ===
            with gr.Tab("Search"):
                with gr.Row(elem_classes="main-container", equal_height=True):
                    # Left Column
                    with gr.Column():
                        index_status = gr.Textbox(
                            label="Index Status", lines=6, interactive=False
                        )
                        refresh_status_btn = gr.Button("Refresh Status")

                        gr.Markdown("### Indexing Parameters")
                        image_dir_input = gr.Textbox(
                            label="Image Directory Path", placeholder="/path/to/images"
                        )
                        wildcard_input = gr.Textbox(label="Wildcard", value="*.JPG")
                        batch_size_input = gr.Number(
                            label="Batch Size", value=16, precision=0
                        )
                        index_button = gr.Button("Index Images")

                        gr.Markdown("### Search Parameters")
                        distance_threshold_input = gr.Number(
                            label="Distance Threshold", value=2000, precision=4
                        )
                        topk_results_input = gr.Number(
                            label="Top-K Results", value=500, precision=0
                        )
                        search_button = gr.Button("Search")

                    # Right Column
                    with gr.Column():
                        upload_image_input = gr.Image(
                            label="Upload Image for Search", type="pil"
                        )
                        top_results_gallery = gr.Gallery(
                            label="Top Results", columns=3, height="auto"
                        )

            # === PAGE 2: Visualization Tab ===
            with gr.Tab("Visualize Embeddings"):
                with gr.Row(elem_classes="main-container"):
                    with gr.Column():
                        index_path_input = gr.Textbox(
                            label="Path to FAISS Index",
                            placeholder="/path/to/index.faiss",
                        )
                        metadata_path_input = gr.Textbox(
                            label="Path to Metadata JSON",
                            placeholder="/path/to/metadata.json",
                        )
                        examples_per_cluster_input = gr.Number(
                            label="Examples Per Cluster", value=5, precision=0
                        )
                        plot_button = gr.Button("Plot Embeddings")
                        embeddings_gallery = gr.Gallery(
                            label="Embedding Plots",
                            columns=2,
                            height="auto",
                            format="png",
                        )

            # === Event Bindings ===
            refresh_status_btn.click(
                self.refresh_status_handler, inputs=[], outputs=[index_status]
            )

            index_button.click(
                self.index_images_handler,
                inputs=[image_dir_input, wildcard_input, batch_size_input],
                outputs=[index_status],
            )

            search_button.click(
                self.search_handler,
                inputs=[
                    upload_image_input,
                    distance_threshold_input,
                    topk_results_input,
                ],
                outputs=[top_results_gallery],
            )

            plot_button.click(
                self.plot_embeddings_handler,
                inputs=[
                    index_path_input,
                    metadata_path_input,
                    examples_per_cluster_input,
                ],
                outputs=[embeddings_gallery],
            )

            demo.load(self.refresh_status_handler, inputs=[], outputs=[index_status])

        demo.launch(allowed_paths=["/"])


if __name__ == "__main__":
    View().launch()
