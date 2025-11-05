import os
import faiss
import click
from ..core import get_db_conn, texts_to_embeddings, EMBEDDING_IMPLEMENTATION


@click.command()
@click.argument("query")
@click.option("--k", default=5, help="Number of results to return.")
def search(query, k):
    """Searches the FAISS index for the given query."""
    index_file = f"data/index_{EMBEDDING_IMPLEMENTATION}.faiss"

    if not os.path.exists(index_file):
        print("Index not found. Please run 'create-index' first.")
        return

    print(f"Loading index from {index_file}")
    index = faiss.read_index(index_file)

    print(f"Searching for: '{query}'")
    query_embedding = texts_to_embeddings([query])

    distances, ids = index.search(query_embedding, k)

    # The search returns IDs, so we need to query the database for the actual content.
    result_ids = [int(id) for id in ids[0] if id != -1]  # id can be -1 if not found
    if not result_ids:
        print("No results found.")
        return

    conn = get_db_conn()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in result_ids)
    query_str = f"SELECT id, name, url, summary FROM star WHERE id IN ({placeholders})"
    cursor.execute(query_str, result_ids)
    results = cursor.fetchall()
    conn.close()

    # Create a dictionary for quick lookup
    results_map = {res[0]: res for res in results}

    print("\nSearch Results:")
    for i, id_val in enumerate(result_ids):
        if id_val in results_map:
            res = results_map[id_val]
            print(f"{i+1}. {res[1]} - {res[2]}")
            print(f"   Summary: {res[3]}")
            # Find the corresponding distance
            original_index = list(ids[0]).index(id_val)
            print(f"   (Distance: {distances[0][original_index]:.4f})\n")
