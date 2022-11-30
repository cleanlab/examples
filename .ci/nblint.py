#!/usr/bin/env python

"""
Lint Jupyter notebooks being checked in to this repo.
This linter check if the kernel used in the notebooks are correct. 
"""

import argparse
import json
import os
import sys


def main():
    opts = get_opts()
    notebooks = find_notebooks(opts.dir)
    for notebook in notebooks:
        check(notebook)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir", help="Directories to search for notebooks", type=str, nargs="+"
    )
    return parser.parse_args()


def find_notebooks(dirs):
    notebooks = set()
    for d in dirs:
        for dirname, _, filenames in os.walk(d):
            for filename in filenames:
                if not filename.endswith(".ipynb"):
                    continue
                full_path = os.path.join(dirname, filename)
                notebooks.add(full_path)
    return notebooks


def check(notebook):
    with open(notebook) as f:
        contents = json.load(f)
    check_correct_kernel(notebook, contents)


def check_correct_kernel(path, contents):
    if contents["metadata"]["kernelspec"]["display_name"] != "Python 3 (ipykernel)":
        fail(
            path,
            "notebook kernel is incorrect, ensure it is set to 'Python 3 (ipykernel)'",
        )


def check_notebook_badge(path, contents):
    first_cell = contents["cells"][1]["source"][0]
    if not ("<!--<badge>-->" in first_cell or "{{ badge }}" in first_cell):
        fail(path, "missing colab badge")


def fail(path, message, cell=None):
    cell_msg = f" [cell {cell}]" if cell is not None else ""
    print(f"{path}{cell_msg}: {message}")
    sys.exit(1)


if __name__ == "__main__":
    main()
