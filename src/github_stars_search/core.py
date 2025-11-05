import os
from dotenv import load_dotenv
import sqlite3
import numpy
from sentence_transformers import SentenceTransformer
import voyageai
import openai
import base64

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
EMBEDDING_IMPLEMENTATION = os.getenv("EMBEDDING_IMPLEMENTATION", "sentence-transformers")
DB_PATH = "data/default.db"

# Initialize clients conditionally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vo = voyageai.Client() if os.getenv("VOYAGE_API_KEY") else None

# Use AsyncClient for openai
aclient = (
    openai.AsyncClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    if os.getenv("DEEPSEEK_API_KEY")
    else None
)


def get_db_conn():
    """Creates and returns a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS star (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            url TEXT NOT NULL,
            summary TEXT NOT NULL
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_status (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
        """
    )
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM sync_status WHERE key = 'last_synced_page'")
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO sync_status (key, value) VALUES ('last_synced_page', 0)")
        conn.commit()
    return conn


async def get_readme_content(session, repo_full_name, headers):
    """Fetches and decodes the README.md content for a given repository asynchronously."""
    readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
    try:
        async with session.get(readme_url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                content = data.get("content")
                if content:
                    return base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error fetching README for {repo_full_name}: {e}")
    return None


async def generate_summary(text):
    """Generates a summary for the given text asynchronously."""
    if not aclient or not text:
        return ""
    max_tokens = 1000
    try:
        response = await aclient.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a technology and programming expert renowned for your ability to summarize and distill GitHub repository README.md files, producing highly accurate and comprehensive abstracts.",
                },
                {
                    "role": "user",
                    "content": f"Please summarize the README.md text as follows, removing non-essential information such as how to contribute or donate. Focus on what this GitHub repository is and what capabilities it provides. Keep the summary within 50 words: {text[:max_tokens * 4]}",
                },
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""


def texts_to_embeddings(texts):
    """Converts a list of texts to embeddings using the configured implementation."""
    if EMBEDDING_IMPLEMENTATION == "sentence-transformers":
        embeddings = model.encode(texts, show_progress_bar=True)
    elif EMBEDDING_IMPLEMENTATION == "voyageai" and vo:
        result = vo.embed(texts, model="voyage-3.5")
        embeddings = result.embeddings
    else:
        raise ValueError(f"Unknown or unconfigured embedding implementation: {EMBEDDING_IMPLEMENTATION}")

    return numpy.array(embeddings).astype("float32")


async def process_star(session, item, headers):
    """Processes a single starred repository."""
    description = item.get("description") or ""
    readme_content = await get_readme_content(session, item["full_name"], headers)
    content_for_summary = readme_content if readme_content else description
    summary = await generate_summary(content_for_summary)

    return (
        item["id"],
        item["full_name"],
        description,
        item["html_url"],
        summary,
    )
