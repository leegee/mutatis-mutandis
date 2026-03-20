# from slice_embedding_pipeline import (
#     SLICES,
#     faiss_slice_path,
#     vocab_slice_path,
#     load_aligned_vectors,
#     load_unaligned_vectors,
#     load_macberth_vectors,
# )
# import faiss
# import numpy as np

# def search_token_across_slices(
#     query: str,
#     top_k: int = 10,
#     aligned: bool = True,
#     backend: str = "macberth"
# ):
#     """
#     Search for a token or phrase across all slices.

#     Parameters
#     ----------
#     query : str
#         The token or phrase to search.
#     top_k : int
#         Number of top matches to return.
#     aligned : bool
#         Whether to use aligned embeddings (comparable across slices).
#     backend : str
#         "fasttext" or "macberth" embeddings.

#     Returns
#     -------
#     list of tuples: [(slice_range, token, score), ...] sorted by descending similarity
#     """
#     all_results = []

#     # Loop over slices
#     for slice_range in SLICES:
#         slice_id = f"{slice_range[0]}-{slice_range[1]}"

#         # Load embeddings
#         if backend == "fasttext":
#             embeddings = (
#                 load_aligned_vectors(slice_id, 'macberth')
#                 if aligned
#                 else load_unaligned_vectors(slice_id, 'macberth')
#             )

#         elif backend == "macberth":
#             embeddings = load_macberth_vectors(slice_id)

#         else:
#             raise ValueError(f"Unknown backend: {backend}")

#         # Compute query vector
#         tokens = [t for t in query.split() if t in embeddings]
#         if not tokens:
#             continue

#         query_vec = np.mean([embeddings[t] for t in tokens], axis=0)
#         query_vec = query_vec / np.linalg.norm(query_vec)
#         query_vec = np.expand_dims(query_vec.astype(np.float32), axis=0)

#         # Load FAISS index
#         index = faiss.read_index(str(faiss_slice_path(slice_range, aligned, 'macberth')))

#         # Search
#         D, Index = index.search(query_vec, top_k)

#         # Map indices to tokens
#         vocab = open(vocab_slice_path(slice_range, aligned, 'macberth'), encoding="utf-8").read().splitlines()
#         results = [(slice_range, vocab[i], D[0][idx]) for idx, i in enumerate(Index[0]) if i >= 0]
#         all_results.extend(results)

#     # Sort across all slices
#     all_results.sort(key=lambda x: -x[2])
#     return all_results[:top_k]


# # Example usage
# if __name__ == "__main__":
#     top_matches = search_token_across_slices("king", top_k=15, aligned=True, backend='macberth')
#     for s, tok, score in top_matches:
#         print(f"Slice {s}: {tok} ({score:.3f})")
