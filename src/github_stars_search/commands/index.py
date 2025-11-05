import os
import faiss
import numpy
import click
from ..core import get_db_conn, texts_to_embeddings, EMBEDDING_IMPLEMENTATION


@click.command()
@click.option('--reset', is_flag=True, help='Reset the index and start from scratch.')
def create_index(reset):
    """Creates or updates a FAISS index for the summaries of the starred repositories."""
    index_file = f"data/index_{EMBEDDING_IMPLEMENTATION}.faiss"
    indexed_star_table = f"indexed_star_{EMBEDDING_IMPLEMENTATION}".replace("-", "_")

    conn = get_db_conn()
    cursor = conn.cursor()

    cursor.execute(f"CREATE TABLE IF NOT EXISTS {indexed_star_table} (star_id INTEGER PRIMARY KEY)")

    if reset:
        print("Resetting index...")
        if os.path.exists(index_file):
            os.remove(index_file)
            print(f"Removed {index_file}")
        cursor.execute(f"DROP TABLE IF EXISTS {indexed_star_table}")
        print(f"Dropped table {indexed_star_table}")
        conn.commit()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {indexed_star_table} (star_id INTEGER PRIMARY KEY)")

    # Load existing index if it exists
    if os.path.exists(index_file):
        print(f"Loading existing index from {index_file}")
        index = faiss.read_index(index_file)
    else:
        print("No existing index found, creating a new one.")
        index = None

    # Get already indexed star IDs from the database
    cursor.execute(f"SELECT star_id FROM {indexed_star_table}")
    indexed_ids = {row[0] for row in cursor.fetchall()}
    print(f"Found {len(indexed_ids)} indexed items in the database.")

    # Query for stars not yet indexed
    if indexed_ids:
        placeholders = ",".join("?" for _ in indexed_ids)
        query = f"SELECT id, summary FROM star WHERE summary != '' AND id NOT IN ({placeholders})"
        cursor.execute(query, list(indexed_ids))
    else:
        cursor.execute("SELECT id, summary FROM star WHERE summary != ''")

    rows = cursor.fetchall()

    new_ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]

    if not texts:
        print("No new summaries to add to the index.")
        if not os.path.exists(index_file):
            print("No data to create an index.")
        conn.close()
        return

    print(f"Creating embeddings for {len(texts)} new summaries...")
    embeddings_np = texts_to_embeddings(texts)
    faiss.normalize_L2(embeddings_np)
    new_ids_np = numpy.array(new_ids).astype("int64")

    if index is None:
        d = embeddings_np.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
        index.add_with_ids(embeddings_np, new_ids_np)
        print("Created a new index.")
    else:
        index.add_with_ids(embeddings_np, new_ids_np)
        print(f"Added {len(new_ids)} new items to the index.")

    print(f"Saving index to {index_file}")
    faiss.write_index(index, index_file)

    # Save new IDs to the database
    if new_ids:
        cursor.executemany(f"INSERT INTO {indexed_star_table} (star_id) VALUES (?)", [(id,) for id in new_ids])
        conn.commit()
        print(f"Saved {len(new_ids)} new IDs to the database.")

    conn.close()
    print("Index creation/update complete.")