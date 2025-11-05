import click
import faiss
from .commands.sync import sync_github_stars
from .commands.index import create_index
from .commands.search import search

faiss.omp_set_num_threads(1)


@click.group()
def cli():
    """A command line tool for managing and searching GitHub starred repositories."""
    pass


cli.add_command(sync_github_stars)
cli.add_command(create_index)
cli.add_command(search)

if __name__ == "__main__":
    cli()
