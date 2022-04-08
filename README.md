# `cleanlab` Examples

This repo contains code examples that demonstrate how to use [cleanlab](https://github.com/cleanlab) and how [confident learning](https://arxiv.org/abs/1911.00068) works to find label errors.

## Latest Examples

Recommended order of examples to try:

| Example | Notebook / Folder                             | Description                                                                                                                                                                                                                                                                    |
| ------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1       | iris_simple_example.ipynb                     | Use `cleanlab` to find synthetic label errors in the Iris dataset.                                                                                                                                                                                                             |
| 2       | classifier_comparison.ipynb                   | Demonstrate how `cleanlab` can be used to train 10 different classifiers on 4 dataset distributions with label errors.                                                                                                                                                         |
| 3       | model_selection_demo.ipynb                    | Perform hyperparameter optimization to find the best settings of `cleanlab`'s optional parameters.                                                                                                                                                                             |
| 4       | simplifying_confident_learning_tutorial.ipynb | Implement `cleanlab` as raw numpy code.                                                                                                                                                                                                                                        |
| 5       | visualizing_confident_learning.ipynb          | Demonstrate how `cleanlab` performs noise matrix estimation.                                                                                                                                                                                                                   |
| 6       | cifar10-cnn-coteaching                        | Demonstrate the use of two experimental modules from `cleanlab`: [cifar_cnn.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/cifar_cnn.py) and [coteaching.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/coteaching.py) |

## Instructions

To run the latest example notebooks, execute the commands below which will install the required libraries in a virtual environment.

To run the examples with your own version of cleanlab, simply edit the first line of requirements.txt, or delete it and install cleanlab separately.

```console
python -m pip install virtualenv
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

For examples 1-5, you may run the notebooks individually or run the bash script below which will execute and save each notebook.

Bash script:

```console
./run_all_notebooks.sh
```

For example 6, please follow the instructions in the readme of the folder.

## Older Examples

See the `contrib` folder for examples from v1 of `cleanlab` which may be helpful for reproducing results from the [Confident Learning paper](https://arxiv.org/abs/1911.00068).

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
