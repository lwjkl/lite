import argparse
import requests
import sys
from tqdm import tqdm


API_BASE = "http://localhost:1234"


def index_images(image_dir: str):
    response = requests.post(
        f"{API_BASE}/index-images", json={"image_directory": image_dir}, stream=True
    )
    pbar = tqdm(total=100, desc="Indexing Progress", unit="%")
    last_progress = 0

    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data:"):
                data = decoded[5:].strip()

                if data == "done":
                    pbar.n = 100
                    pbar.refresh()
                    break

                try:
                    current = int(data)
                    delta = current - last_progress
                    if delta > 0:
                        pbar.update(delta)
                        last_progress = current
                except ValueError:
                    print(f"Unexpected stream data: {data}")

    pbar.close()


def search_and_download(image_path: str, num_results: int = 9, top_k: int = 500, threshold: float = None):
    print("Searching for similar images...")

    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "top_k": top_k,
            "num_results": num_results,
        }
        if threshold is not None:
            data["distance_threshold"] = threshold

        response = requests.post(f"{API_BASE}/search-similar", files=files, data=data)
        response.raise_for_status()
        result = response.json()

    image_ids = [r[0] for r in result["full_results"]]
    print(f"Found {len(image_ids)} similar images. Downloading...")

    zip_response = requests.post(
        f"{API_BASE}/download-zip", json={"image_ids": image_ids}, stream=True
    )

    if zip_response.status_code == 200:
        with open("matched_images.zip", "wb") as f:
            for chunk in zip_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Downloaded ZIP file: matched_images.zip")
    else:
        print("Failed to download ZIP:", zip_response.status_code, zip_response.text)


def check_status():
    response = requests.get(f"{API_BASE}/index-status")
    print(response.json())


def check_app_status():
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code != 200:
            print("Application is not running or incorrect URL.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Failed to connect to API.")
        sys.exit(1)


def plot_embedding(
    index_path: str, metadata_path: str, examples_per_cluster: int = 5, output_file: str = "embeddings_plot.png"
):
    print("Requesting visualization generation...")

    data = {
        "index_path": index_path,
        "metadata_path": metadata_path,
        "examples_per_cluster": str(examples_per_cluster),
    }

    response = requests.post(f"{API_BASE}/plot-embeddings", data=data)
    if (
        response.status_code == 200
        and response.headers.get("content-type") == "image/png"
    ):
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Visualization saved to {output_file}")
    else:
        print(f"Failed to generate visualization. Status: {response.status_code}")
        print("Response:", response.text)


def main():
    check_app_status()

    parser = argparse.ArgumentParser(description="lite CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_index = subparsers.add_parser("index", help="Directory of images to index")
    parser_index.add_argument("--dir", required=True, help="Path to image directory")

    parser_search = subparsers.add_parser(
        "search", help="Search with a query image and download results as ZIP"
    )
    parser_search.add_argument("--image", required=True, help="Path to query image")
    parser_search.add_argument(
        "--num_results", type=int, default=9, help="Useless at the moment"
    )
    parser_search.add_argument(
        "--top_k", type=int, default=500, help="Top K to search in index"
    )
    parser_search.add_argument("--threshold", type=float, help="Distance threshold")

    subparsers.add_parser("status", help="Check index status")

    parser_plot = subparsers.add_parser("plot", help="Generate embedding plot")
    parser_plot.add_argument(
        "--index_path", required=True, help="Path to FAISS index file"
    )
    parser_plot.add_argument(
        "--metadata_path", required=True, help="Path to metadata JSON file"
    )
    parser_plot.add_argument(
        "--examples_per_cluster",
        type=int,
        default=5,
        help="Examples per cluster to show",
    )
    parser_plot.add_argument(
        "--output", default="embeddings_plot.png", help="Output PNG filename"
    )

    args = parser.parse_args()

    if args.command == "index":
        index_images(args.dir)
    elif args.command == "search":
        search_and_download(args.image, args.num_results, args.top_k, args.threshold)
    elif args.command == "status":
        check_status()
    elif args.command == "plot":
        plot_embedding(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            examples_per_cluster=args.examples_per_cluster,
            output_file=args.output,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
