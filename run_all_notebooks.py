import papermill as pm
from pathlib import Path
import os
import click


@click.command()
@click.option(
    "--root_dir",
    "-d",
    default="./",
    help="Root directory to run all Jupyter notebooks.",
)
def main(root_dir):
    """Run all Jupyter notebooks in the root directory"""

    print("-----------------------------------------------------------------")
    print(f"Executing all Jupyter notebooks in root directory: {root_dir}")
    print("-----------------------------------------------------------------")

    # list of notebook folders to ignore
    ignore_folders = [
        # ignoring checkpoints, git folders and v1 notebooks
        ".ipynb_checkpoints",
        ".git",
        "contrib",
        # insert notebooks to ignore below
        "6_cifar10_cnn_coteaching",
        "7_mnist_cnn",
        "8_amazon_reviews_fasttext",
    ]

    folders = [
        f for f in filter(os.path.isdir, os.listdir("./")) if f not in ignore_folders
    ]

    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root_dir, folder, file)

                # execute notebook with papermill
                print(f"Executing Jupyter notebook: {notebook_path}")
                pm.execute_notebook(
                    input_path=notebook_path,
                    output_path=notebook_path,
                )


if __name__ == "__main__":
    main()
