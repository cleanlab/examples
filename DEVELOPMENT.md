# Development

This guide provides a checklist for contributing new cleanlab examples.

- Add your example's notebook and any relevant files in a folder, prefixing the name of the folder with a number (use the next number from the [Table of Contents](https://github.com/cleanlab/examples#table-of-contents))

- Ensure that the notebook contains cell outputs and that they look as expected on GitHub, additionally clear any cell blocks that are too large (eg. model training code that specifies accuracy for each epoch). This is different than our tutorials in the main cleanlab repository (where notebook cells should not be executed)!

- Ensure that the jupyter notebook cells are executed in order.

- If the notebook takes a long time to run or is hard to auto-execute, add its folder name to the `ignore_folders` list in [run_all_notebooks.py](run_all_notebooks.py)

- When adding a new example, use `pip freeze` to determine the package versions that are used, then

    1. create a `requirements.txt` file in the specific example's folder, and add the notebook's dependency specifications there
    
    2. add the dependency specifications in the main `requirements.txt` file, ensuring that the new dependencies do not conflict with the existing ones

- Add the notebook to the [Table of Contents](https://github.com/cleanlab/examples#table-of-contents)
 table in the README, ensuring that the folder prefix number matches the index of the table
