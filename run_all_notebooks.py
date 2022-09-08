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

    # list of notebooks for latest examples
    notebooks = [
        "iris_simple_example.ipynb",
        "classifier_comparison.ipynb",
        "hyperparameter_optimization.ipynb",
        "simplifying_confident_learning.ipynb",
        "visualizing_confident_learning.ipynb",
        "outlier-detection-cifar10/outlier_detection.ipynb",
    ]

    for notebook in notebooks:

        notebook_path = f"{root_dir}{notebook}"

        # execute notebook with papermill
        print(f"Executing Jupyter notebook: {notebook_path}")
        pm.execute_notebook(
            input_path=notebook_path,
            output_path=notebook_path,
        )


if __name__ == "__main__":
    main()
