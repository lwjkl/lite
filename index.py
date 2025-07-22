import faiss
import json
import numpy
import os


class FaissIndex:
    def __init__(self, d: int = 2048, M: int = 32, index_path: str = None):
        self.status_message = "Uninitialized FAISS index."
        self.image_directory = None
        self._str_to_int = {}
        self._int_to_str = {}
        self._next_id = 1

        if index_path is None or not os.path.exists(index_path):
            print(f"Creating new FAISS index (d={d}, M={M})...")
            base_index = faiss.IndexHNSWFlat(d, M)
            self.index = faiss.IndexIDMap(base_index)
            self.status_message = "New FAISS index created (empty)."
        else:
            try:
                self.index: faiss.IndexHNSWFlat = faiss.read_index(index_path)
                print(f"Loaded FAISS index successfully from {index_path}!")
                self.status_message = f"FAISS index loaded from {index_path}."

                # Load metadata (image directory + ID mappings)
                meta_path = index_path + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as meta_file:
                        meta = json.load(meta_file)
                        self.image_directory = meta.get("image_directory")
                        print(self.image_directory)
                        self._str_to_int = meta.get("str_to_int", {})
                        self._int_to_str = {
                            int(k): v for k, v in meta.get("int_to_str", {}).items()
                        }
                        self._next_id = max(self._int_to_str.keys(), default=0) + 1
                    print(f"Loaded metadata: image_directory={self.image_directory}")
                else:
                    print(f"No metadata found for {index_path}.")
            except Exception as exp:
                print(f"Failed to load FAISS index: {exp}")
                base_index = faiss.IndexHNSWFlat(d, M)
                self.index = faiss.IndexIDMap(base_index)
                self.status_message = f"Failed to load FAISS index. Created new empty index. Error: {exp}"


    def _get_or_create_int_id(self, string_id: str) -> int:
        """
        Get an existing int ID for a string ID, or create a new one.
        """
        if string_id in self._str_to_int:
            return self._str_to_int[string_id]
        int_id = self._next_id
        self._next_id += 1
        self._str_to_int[string_id] = int_id
        self._int_to_str[int_id] = string_id
        return int_id

    def insert(self, vectors, string_ids):
        """
        Insert vectors into the index, using string IDs.
        """
        if not isinstance(vectors, numpy.ndarray):
            vectors = numpy.array(vectors, dtype=numpy.float32)

        # Map string IDs to integer IDs
        int_ids = [self._get_or_create_int_id(sid) for sid in string_ids]

        if len(vectors) == 0:
            print("No vectors to insert.")
            return

        existing_ids = set(faiss.vector_to_array(self.index.id_map))

        filtered_vectors = []
        filtered_ids = []
        for v, int_id in zip(vectors, int_ids):
            if int_id in existing_ids:
                print(f"Skipping duplicate ID: {self._int_to_str[int_id]}")
                continue
            filtered_vectors.append(v)
            filtered_ids.append(int_id)

        if filtered_vectors:
            self.index.add_with_ids(
                numpy.array(filtered_vectors, dtype=numpy.float32),
                numpy.array(filtered_ids, dtype=numpy.int64)
            )

    def search(self, query_vector: numpy.ndarray, k: int = 5, distance_threshold: float = None):
        """
        Search vectors and return (string_id, distance).
        """
        if not isinstance(query_vector, numpy.ndarray):
            query_vector = numpy.array(query_vector, dtype=numpy.float32)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, int_ids = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], int_ids[0]):
            if idx == -1:
                continue
            if distance_threshold is not None and dist > distance_threshold:
                continue
            string_id = self._int_to_str.get(idx, f"unknown_{idx}")
            results.append((string_id, float(dist)))
        return results

    def save(self, index_path: str):
        """
        Save the index to a local directory.
        """
        faiss.write_index(self.index, index_path)
        print(f"Successfully saved index to {index_path}.")

    def save_index_and_meta(self, index_path: str, image_dir: str):
        """
        Save the index and associated metadata (directory + ID mappings).
        """
        faiss.write_index(self.index, index_path)
        print(f"FAISS index saved to {index_path}.")

        meta_path = index_path + ".meta.json"
        with open(meta_path, "w") as meta_file:
            json.dump(
                {
                    "image_directory": image_dir,
                    "str_to_int": self._str_to_int,
                    "int_to_str": self._int_to_str
                }, meta_file, indent=4)
        print(f"Metadata saved to {meta_path}.")
        self.image_directory = image_dir

    # def get_index_status(self):
    #     """
    #     Get the status of index.
    #     """
    #     total_vectors = self.index.ntotal
    #     print("Total vectors in FAISS:", total_vectors)

    #     ids = faiss.vector_to_array(self.index.id_map)
    #     num_ids = len(ids)
    #     num_uids = len(set(ids))
    #     is_duplicates = num_ids != num_uids

    #     return total_vectors, num_uids, is_duplicates
    
    def get_status(self) -> dict:
        """
        Get the status of index.
        """
        total_vectors = self.index.ntotal
        print("Total vectors in FAISS:", total_vectors)

        ids = faiss.vector_to_array(self.index.id_map)
        num_ids = len(ids)
        num_uids = len(set(ids))
        is_duplicates = num_ids != num_uids

        return {
            "message": self.status_message,
            "image_directory": self.image_directory,
            "total_vectors": total_vectors,
            "unique_ids": num_uids,
            "has_duplicates": is_duplicates
        }

