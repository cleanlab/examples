# `cleanlab` Examples

This repo contains code examples that demonstrate how to use [cleanlab](https://github.com/cleanlab) and how [confident learning](https://arxiv.org/abs/1911.00068) works to find label errors.

## Latest Examples

Recommended order of examples to try:

1. [iris_simple_example.ipynb](https://github.com/cleanlab/examples/blob/master/iris_simple_example.ipynb)

   Use `cleanlab` to find synthetic label errors in the Iris dataset.

2. [classifier_comparison.ipynb](https://github.com/cleanlab/examples/blob/master/classifier_comparison.ipynb)

   Demonstrate how `cleanlab` can be used to train 10 different classifiers on 4 dataset distributions with label errors.

3. [model_selection_demo.ipynb](https://github.com/cleanlab/examples/blob/master/model_selection_demo.ipynb)

   Perform hyperparameter optimization with `cleanlab`'s hyperparameters.

4. [simplifying_confident_learning_tutorial.ipynb](https://github.com/cleanlab/examples/blob/master/simplifying_confident_learning_tutorial.ipynb)

   Implement `cleanlab` as raw numpy code.

5. [visualizing_confident_learning.ipynb](https://github.com/cleanlab/examples/blob/master/visualizing_confident_learning.ipynb)

   Demonstrate how `cleanlab` performs noise matrix estimation.

## Instructions

To run the latest example scripts and notebooks, execute the commands below which will install the required libraries in a virtual environment.

```console
python3 -m pip install virtualenv
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

Run the notebooks individually or run the scripts below which will execute and save each notebook.

Bash script:

```console
./run_all_notebooks.sh
```

Python script (can pass optional arguments):

```console
python run_all_notebooks.py --ignore_sub_dirs env
```

## Old Examples

See the `contrib` folder for examples from v1 of `cleanlab`.

## License

Copyright (c) 2017-2022 Cleanlab Inc.

All files listed above and contained in this folder (<https://github.com/cleanlab/examples>) are part of cleanlab.

cleanlab is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cleanlab is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License in [LICENSE](LICENSE).
