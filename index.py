import faiss
import json
import numpy
import os


class FaissIndex:
    def __init__(self, d: int = 2048, M: int = 32, index_path: str = None):
        self.status_message = ""

        if index_path is None or not os.path.exists(index_path):
            print(f"Creating new FAISS index (d={d}, M={M})...")
            base_index = faiss.IndexHNSWFlat(d, M)
            self.index = faiss.IndexIDMap(base_index)
            self.status_message = "New FAISS index created (empty)."
        else:
            self.index: faiss.IndexHNSWFlat = faiss.read_index(index_path)
            print(f"Loaded FAISS index successfully from {index_path}!")
            self.status_message = f"FAISS index loaded from {index_path}."

            meta_path = index_path + ".meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as meta_file:
                    meta = json.load(meta_file)
                    self.image_directory = meta.get("image_directory")
                print(f"Loaded metadata: image_directory={self.image_directory}")
                # self.status_message += f"\nMetadata loaded: image_directory={self.image_directory}."
            else:
                print(f"No metadata found for {index_path}.")
                # self.status_message += " No metadata found."

    def insert(self, vectors, ids):
        """
        Insert vectors into the index.
        """
        if not isinstance(vectors, numpy.ndarray):
            vectors = numpy.array(vectors, dtype=numpy.float32)

        if not isinstance(ids, numpy.ndarray):
            ids = numpy.array(ids)

        if len(vectors) == 0:
            print("No vectors to insert.")
            return

        # Get existing IDs in FAISS
        existing_ids = set(faiss.vector_to_array(self.index.id_map))

        filtered_vectors = []
        filtered_ids = []
        for v, i in zip(vectors, ids):
            if i in existing_ids:
                print(f"Skipping duplicate ID: {i}")
                continue
            filtered_vectors.append(v)
            filtered_ids.append(i)

        if filtered_vectors:
            self.index.add_with_ids(numpy.array(filtered_vectors, dtype=numpy.float32),
                                    numpy.array(filtered_ids))

    def search(self, query_vector: numpy.ndarray, k: int = 5, distance_threshold: float = None):
        if not isinstance(query_vector, numpy.ndarray):
            query_vector = numpy.array(query_vector, dtype=numpy.float32)
       
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                # This happens if FAISS returns an invalid index
                continue
            if distance_threshold is not None and dist > distance_threshold:
                continue
            results.append((int(idx), float(dist)))

        return results

    def save(self, index_path: str):
        """
        Save the index to a local directory.
        """
        faiss.write_index(self.index, index_path)
        print(f"Successfully saved index to {index_path}.")

    def save_index_and_meta(self, index_path: str, image_dir: str):
        """
        Save the index and the image directory metadata.
        """
        faiss.write_index(self.index, index_path)
        print(f"FAISS index saved to {index_path}.")

        meta_path = index_path + ".meta.json"
        with open(meta_path, "w") as meta_file:
            json.dump({"image_directory": image_dir}, meta_file)
        print(f"Metadata saved to {meta_path}.")
        self.image_directory = image_dir  # Update current instance

    def get_index_status(self):
        """
        Get the status of index.
        """
        total_vectors = self.index.ntotal
        print("Total vectors in FAISS:", total_vectors)

        ids = faiss.vector_to_array(self.index.id_map)
        num_ids = len(ids)
        num_uids = len(set(ids))
        is_duplicates = num_ids != num_uids

        # print("IDs in FAISS:", ids)
        # print("Unique IDs:", num_uids)
        # print("Duplicates?", is_duplicates)

        return total_vectors, num_uids, is_duplicates
    
        



