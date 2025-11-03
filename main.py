import os
import json
from dotenv import load_dotenv
import numpy
from sentence_transformers import SentenceTransformer
import voyageai
import faiss

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if os.getenv("VOYAGE_API_KEY") is not None:

    vo = voyageai.Client()


def texts_to_embeddings(texts):
    embedding_method = os.getenv("EMBEDDING_METHOD")

    if embedding_method == "sentence-transformers":

        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings_np = numpy.array(embeddings).astype("float32")

        return {
            "embeddings": embeddings_np,
            "embeddings_np": embeddings_np,
        }

    if embedding_method == "voyageai" and vo is not None:
        embeddings = vo.embed(texts, model="voyage-3.5").embeddings
        embeddings_np = numpy.array(embeddings).astype("float32")

        return {
            "embeddings": embeddings_np,
            "embeddings_np": embeddings_np,
        }

    raise ValueError(f"Unknown embedding method: {embedding_method}")


def main():

    load_dotenv()

    index_file = "data/index.faiss"
    texts_file = "data/texts.json"

    vo = voyageai.Client()

    if os.path.exists(index_file) and os.path.exists(texts_file):
        print(f"Loading index from {index_file}")
        index = faiss.read_index(index_file)
        print(f"Loading texts from {texts_file}")
        with open(texts_file, "r") as f:
            texts = json.load(f)
    else:
        print("Index not found, creating new one.")
        texts = ["An extremely fast Python package and project manager, written in Rust.", "Faiss is a library for efficient similarity search and clustering of dense vectors."]
        embeddings = texts_to_embeddings(texts)

        d = embeddings["embeddings"].shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings["embeddings_np"])

        print(f"Saving index to {index_file}")
        faiss.write_index(index, index_file)
        print(f"Saving texts to {texts_file}")
        with open(texts_file, "w") as f:
            json.dump(texts, f)

    query = "package manager"
    query_embeddings = texts_to_embeddings([query])

    distances, indices = index.search(query_embeddings["embeddings_np"], k=2)

    for i, idx in enumerate(indices[0]):
        print(f"结果 {i+1}: {texts[idx]}（距离={distances[0][i]:.4f}）")


if __name__ == "__main__":
    main()
