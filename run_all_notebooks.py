import papermill as pm
from pathlib import Path
import os
import click


@click.command()
@click.option(
    "--root_dir",
    "-d",
    default="./",
    help="Root directory to find and run all Jupyter notebooks.",
)
@click.option(
    "--ignore_sub_dirs",
    "-i",
    default=["env"],
    multiple=True,
    help="Ignore these sub directories when traversing the root directory. Can pass multiple args.",
)
def main(root_dir, ignore_sub_dirs):
    """Traverse root directory and run all Jupyter notebooks"""

    print("-----------------------------------------------------------------")
    print(f"Executing all Jupyter notebooks in root directory: {root_dir}")
    print("-----------------------------------------------------------------")

    # traverse root directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ipynb"):

                # path to Jupyter notebook
                notebook_path = Path(os.path.join(root, file))

                # execute notebooks
                if not (
                    notebook_path.parts[0]
                    in ignore_sub_dirs  # ignore notebook if it is in the list of sub directories
                ):

                    # execute notebook with papermill
                    print(f"Executing Jupyter notebook: {notebook_path}")
                    pm.execute_notebook(
                        input_path=notebook_path,
                        output_path=notebook_path,
                    )


if __name__ == "__main__":
    main()
