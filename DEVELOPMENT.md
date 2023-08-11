# Development

This guide provides a checklist for contributing new cleanlab examples.

- Add your example's notebook and any relevant files in a folder.

- Ensure that the notebook contains cell outputs and that they look as expected in Jupyter notebook and **on GitHub**. Note this is different than our tutorials in the main cleanlab repository (where notebook cells should not be executed)! Unlike the tutorials, we want examples notebooks to also look good in GitHub's viewer (which has limited rendering functionality, so avoid things like `<div>` that GitHub's viewer does not render properly). 

- Ensure that the jupyter notebook cells are executed in order. Additionally clear any cell blocks that are too large (eg. model training code that specifies accuracy for each epoch), it is ok if these do not have an execution number after being cleared.

- The second cell of the notebook (right after the `<h1>` title block, and right before the text introduction of the notebook) should be a markdown block containing the text:
    ```
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cleanlab/examples/blob/master/{ relative path to notebook }.ipynb)
    ``` 

    Replace the `{ relative path to notebook }` portion with the path to the notebook relative to the root folder. 

    > eg. the [find_label_errors_iris](find_label_errors_iris/find_label_errors_iris.ipynb) notebook will have a relative path of `find_label_errors_iris/find_label_errors_iris.ipynb` and will have the badge:
    > 
    > ```
    > [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cleanlab/examples/blob/master/find_label_errors_iris/find_label_errors_iris.ipynb.ipynb)
    > ```

    This will create a badge that will link to a Google Colab version of the notebook. 
    
    Note that the Colab badge links to the notebook in the master branch, so at the time of making the PR, the link will be invalid. Please remember to check that the Colab link works after the PR has been approved and merged to `master`.
    
    The Colab badge must also be in its own notebook cell, not with other content.
    
- If your notebook cannot be fully executed and does not have a Colab badge, you can specify the folder name containing your notebook (not the notebook name) to the `ignore_paths` argument in the [ci.yml file](./.github/workflows/ci.yml). This input should be a comma-separated list of folders. Adding your folder to the `ignore_paths` list will make the notebook linter skip all notebooks in that folder, and hence will not throw an error about a missing Colab badge during CI.

- Use `pip freeze` to determine the package versions that are used, then

    1. create a `requirements.txt` file in the specific example's folder, and add the notebook's dependency specifications there.
    
    2. add the dependency specifications in the main `requirements.txt` file, ensuring that the new dependencies do not conflict with the existing ones.

    3. add a markdown block right above the code cell that imports the dependencies that reads:
    
        > Please install the dependencies specified in this requirements.txt file before running the notebook.

        where the requirements.txt text hyperlinks to the `requirements.txt` file located in that specific example's folder (the file created in step 1). Be sure to use the absolute github link (instead of relative path) as Google Colab will have no access to other files. 

        > eg. for the [find_label_errors_iris](find_label_errors_iris/find_label_errors_iris.ipynb) notebook, we would hyperlink the following url:
        > 
        > ```
        > https://github.com/cleanlab/examples/blob/master/find_label_errors_iris/requirements.txt
        > ```

- Ensure that your notebook is using the correct kernel. In jupyter notebook, you can check the notebook's metadata by navigating to `Edit` > `Edit Notebook Metadata` and check if the follow fields match this:

```json
"kernelspec": {
    "name": "python3",
    "display_name": "Python 3 (ipykernel)",
    "language": "python"
  }
```

This is especially important so that papermill can run smoothly.

- If the notebook takes a long time to run or is hard to auto-execute, add its folder name to the `ignore_folders` list in [run_all_notebooks.py](run_all_notebooks.py).

- Add the notebook to the [Table of Contents](https://github.com/cleanlab/examples#table-of-contents)
 table in the README, ideally grouping the newly added example with any other related examples.

- After a new notebook has been added and pushed to `master` branch, AVOID changing the notebook and folder names, as this may break existing links referencing the example notebook throughout cleanlab documentation, blog posts, and more.


### Running all examples

You may run the notebooks individually or run the bash script below which will execute and save each notebook (at least the first few examples notebooks, later notebooks may be skipped because they do not run quickly). Note that before executing the script to run all notebooks for the first time you will need to create a jupyter kernel named `cleanlab-examples`. Be sure that you have already created and activated the virtual environment before running the following command to create the jupyter kernel.

```console$ 
python -m pip install virtualenv
$ python -m venv cleanlab-examples  # creates a new venv named cleanlab-examples
$ source cleanlab-examples/bin/activate
$ python -m pip install -r requirements-dev.txt
$ python -m ipykernel install --user --name=cleanlab-examples
```

Bash script to run all notebooks:

```console
$ bash ./run_all_notebooks.sh
```

Note that the `run_all_notebooks` script will only work on `python<=3.9` as some of the dependencies do not support later versions.