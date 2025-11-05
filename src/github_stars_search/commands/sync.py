import asyncio
import aiohttp
import click
from ..core import get_db_conn, process_star, GITHUB_TOKEN


@click.command()
@click.option("--to-page", default=None, type=int, help="The page number to sync up to.")
def sync_github_stars(to_page):
    """Fetches starred repositories from GitHub and stores them in the database."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = "https://api.github.com/user/starred"

    async def main():
        conn = get_db_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM sync_status WHERE key = 'last_synced_page'")
        row = cursor.fetchone()
        page = (row[0] + 1) if row and row[0] else 1

        async with aiohttp.ClientSession() as session:
            while True:
                if to_page is not None and page > to_page:
                    print(f"Reached target page {to_page}, stopping sync.")
                    break

                print(f"Fetching page {page} of starred repositories...")
                response = await session.get(
                    url,
                    headers=headers,
                    params={
                        "per_page": 30,
                        "page": page,
                        "sort": "created",
                        "direction": "asc",
                    },
                )
                if response.status != 200:
                    print(f"Failed to fetch stars: {await response.text()}")
                    break
                data = await response.json()

                if not data:
                    print("No more new stars to fetch.")
                    break

                star_ids = [item["id"] for item in data]
                placeholders = ",".join("?" for _ in star_ids)
                cursor.execute(f"SELECT id FROM star WHERE id IN ({placeholders})", star_ids)
                existing_ids = {row[0] for row in cursor.fetchall()}

                new_items = [item for item in data if item["id"] not in existing_ids]

                if new_items:
                    print(f"Found {len(new_items)} new stars to process on page {page}.")
                    tasks = [process_star(session, item, headers) for item in new_items]
                    results = await asyncio.gather(*tasks)
                    stars_to_insert = [r for r in results if r is not None]

                    if stars_to_insert:
                        cursor.executemany(
                            "INSERT OR IGNORE INTO star (id, name, description, url, summary) VALUES (?, ?, ?, ?, ?)",
                            stars_to_insert,
                        )
                        conn.commit()
                        print(f"Successfully inserted {len(stars_to_insert)} new stars.")
                else:
                    print(f"No new stars to process on page {page}.")

                cursor.execute(
                    "UPDATE sync_status SET value = ? WHERE key = 'last_synced_page'",
                    (page,),
                )
                conn.commit()
                print(f"Successfully synced page {page}.")

                page += 1
        conn.close()

    asyncio.run(main())
