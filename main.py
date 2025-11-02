from sentence_transformers import SentenceTransformer
import faiss
import numpy

def main():
    texts = ["An extremely fast Python package and project manager, written in Rust.", "Faiss is a library for efficient similarity search and clustering of dense vectors."]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(numpy.array(embeddings).astype("float32"))

    query = "package manager"
    q_vec = model.encode([query])

    distances, indices = index.search(q_vec, k=2)

    for i, idx in enumerate(indices[0]):
        print(f"结果 {i+1}: {texts[idx]}（距离={distances[0][i]:.4f}）")

    # print(embeddings)
    # print("Hello from github-stars-search!")


if __name__ == "__main__":
    main()
