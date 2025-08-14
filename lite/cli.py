import argparse
import cv2
import os
import sys
import zipfile
from tqdm import tqdm

from lite.app import App
from lite.logger import logger


def create_app():
    """Initialize the App instance"""
    try:
        return App()
    except Exception as e:
        print(f"Failed to initialize app: {e}")
        sys.exit(1)


def index_images(app: App, image_dir: str):
    """Index images with progress bar"""
    print(f"Indexing images from: {image_dir}")

    # Get total count for progress bar
    from glob import glob

    paths = (
        glob(f"{image_dir}/*.JPG")
        + glob(f"{image_dir}/*.jpg")
        + glob(f"{image_dir}/*.png")
        + glob(f"{image_dir}/*.PNG")
    )
    total_images = len(paths)

    if not paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {total_images} images to index")

    # Use the stream generator
    pbar = tqdm(total=100, desc="Indexing Progress", unit="%")
    last_progress = 0

    for progress_data in app.index_images_stream(image_dir, "*.JPG", 64):
        if progress_data.startswith("data:"):
            data = progress_data[5:].strip()

            if data == "done":
                pbar.n = 100
                pbar.refresh()
                break
            elif data.startswith("error"):
                print(f"\nError: {data}")
                break
            else:
                try:
                    current = int(data)
                    delta = current - last_progress
                    if delta > 0:
                        pbar.update(delta)
                        last_progress = current
                except ValueError:
                    pass

    pbar.close()
    print("Indexing completed!")


def search_and_save(
    app: App,
    image_path: str,
    num_results: int = 9,
    top_k: int = 500,
    threshold: float = None,
    save_results: bool = False,
    results_name: str = None,
    download_zip: bool = False,
):
    """Search for similar images and optionally save results"""
    print(f"Searching for similar images using: {image_path}")

    try:
        # Load and preprocess query image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Search for similar images
        search_params = {
            "top_k": top_k,
            "num_results": num_results,
        }
        if threshold is not None:
            search_params["distance_threshold"] = threshold

        result = app.search_similar_images(img, **search_params)

        print(
            f"Found {result['count_within_threshold']} similar images within threshold"
        )
        print(f"Showing top {len(result['results'])} results:")

        # Display results
        for i, res in enumerate(result["results"], 1):
            print(f"  {i}. {res['image_id']} (distance: {res['distance']:.2f})")

        # Save results if requested
        if save_results:
            filename = app.save_search_results(result, results_name)
            print(f"Results saved to: {filename}")

        # Download ZIP if requested
        if download_zip:
            download_similar_images_zip(app, result)

    except Exception as e:
        print(f"Search failed: {e}")


def download_similar_images_zip(
    app: App, search_result: dict, zip_filename: str = "matched_images.zip"
):
    """Download similar images as ZIP file"""
    image_ids = [r[0] for r in search_result["full_results"]]

    if not image_ids:
        print("No images to download")
        return

    print(f"Downloading {len(image_ids)} similar images...")

    try:
        # Get image directory from the app's faiss_index
        image_directory = getattr(app.faiss_index, "image_directory", None)
        if not image_directory:
            print("Image directory not available")
            return

        # Create ZIP file
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for image_id in tqdm(image_ids, desc="Adding to ZIP"):
                image_path = os.path.join(image_directory, image_id)
                if os.path.exists(image_path):
                    zipf.write(image_path, arcname=os.path.basename(image_path))
                else:
                    print(f"Warning: Image not found: {image_path}")

        print(f"Downloaded ZIP file: {zip_filename}")

    except Exception as e:
        print(f"Failed to create ZIP: {e}")


def check_status(app: App):
    """Check index status"""
    try:
        if not app.faiss_index:
            print("FAISS index is not initialized")
            return

        status = app.faiss_index.get_status()
        print("Index Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Failed to get status: {e}")


def plot_embedding(
    app: App,
    index_path: str,
    metadata_path: str,
    examples_per_cluster: int = 5,
    output_file: str = "embeddings_plot.png",
):
    """Generate embedding visualization plots"""
    print("Generating visualization...")

    try:
        png_bytes_list = app.plot_embeddings(
            index_path, metadata_path, examples_per_cluster
        )

        # Save plots
        plot1_filename = output_file.replace(".png", "_scatter.png")
        plot2_filename = output_file.replace(".png", "_examples.png")

        with open(plot1_filename, "wb") as f:
            f.write(png_bytes_list[0])

        with open(plot2_filename, "wb") as f:
            f.write(png_bytes_list[1])

        print(f"Scatter plot saved to: {plot1_filename}")
        print(f"Examples plot saved to: {plot2_filename}")

    except Exception as e:
        print(f"Failed to generate visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description="Image Similarity CLI (Direct)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    parser_index = subparsers.add_parser("index", help="Index directory of images")
    parser_index.add_argument("--dir", required=True, help="Path to image directory")

    # Search command
    parser_search = subparsers.add_parser("search", help="Search with a query image")
    parser_search.add_argument("--image", required=True, help="Path to query image")
    parser_search.add_argument(
        "--num_results", type=int, default=9, help="Number of results to show"
    )
    parser_search.add_argument(
        "--top_k", type=int, default=500, help="Top K to search in index"
    )
    parser_search.add_argument("--threshold", type=float, help="Distance threshold")
    parser_search.add_argument(
        "--save", action="store_true", help="Save search results"
    )
    parser_search.add_argument("--name", help="Custom name for saved results")
    parser_search.add_argument(
        "--download", action="store_true", help="Download results as ZIP"
    )

    # Status command
    subparsers.add_parser("status", help="Check index status")

    # Plot command
    parser_plot = subparsers.add_parser("plot", help="Generate embedding plot")
    parser_plot.add_argument(
        "--index_path", required=True, help="Path to FAISS index file"
    )
    parser_plot.add_argument(
        "--metadata_path", required=True, help="Path to metadata JSON file"
    )
    parser_plot.add_argument(
        "--examples_per_cluster", type=int, default=5, help="Examples per cluster"
    )
    parser_plot.add_argument(
        "--output", default="embeddings_plot.png", help="Output PNG filename"
    )

    args = parser.parse_args()

    # Initialize app once
    print("Initializing application...")
    app = create_app()

    # Execute commands
    try:
        if args.command == "index":
            index_images(app, args.dir)

        elif args.command == "search":
            search_and_save(
                app,
                args.image,
                args.num_results,
                args.top_k,
                args.threshold,
                save_results=args.save,
                results_name=args.name,
                download_zip=args.download,
            )

        elif args.command == "status":
            check_status(app)

        elif args.command == "plot":
            plot_embedding(
                app,
                args.index_path,
                args.metadata_path,
                args.examples_per_cluster,
                args.output,
            )

        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Save any changes
        app.save()


if __name__ == "__main__":
    main()
