# cleanlab Examples

This repo contains code examples that demonstrate how to use [cleanlab](https://github.com/cleanlab) and how [confident learning](https://arxiv.org/abs/1911.00068) works to find label errors.

To quickly learn the basics of running cleanlab on your own data, we recommend first starting [here](https://docs.cleanlab.ai/v1.0.1/index.html#quickstart) before diving into the examples below.

## Table of Contents

Recommended order of examples to try:

|     | Example                                                                                        | Description                                                                                                                                                                                                                                                                  |
| --- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [iris_simple_example.ipynb](iris_simple_example.ipynb)                                         | Use cleanlab to find synthetic label errors in the Iris dataset.                                                                                                                                                                                                             |
| 2   | [classifier_comparison.ipynb](classifier_comparison.ipynb)                                     | Demonstrate how cleanlab can be used to train 10 different classifiers on 4 dataset distributions with label errors.                                                                                                                                                         |
| 3   | [model_selection_demo.ipynb](model_selection_demo.ipynb)                                       | Perform hyperparameter optimization to find the best settings of cleanlab's optional parameters.                                                                                                                                                                             |
| 4   | [simplifying_confident_learning_tutorial.ipynb](simplifying_confident_learning_tutorial.ipynb) | Implement cleanlab as raw numpy code.                                                                                                                                                                                                                                        |
| 5   | [visualizing_confident_learning.ipynb](visualizing_confident_learning.ipynb)                   | Demonstrate how cleanlab performs noise matrix estimation.                                                                                                                                                                                                                   |
| 6   | [cifar10-cnn-coteaching](cifar10-cnn-coteaching)                                               | Demonstrate the use of two experimental modules from cleanlab: [cifar_cnn.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/cifar_cnn.py) and [coteaching.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/coteaching.py) |
| 7   | [mnist-cnn](mnist-cnn)                                                                         | Demonstrate the use of the following experimental module from cleanlab: [mnist_pytorch.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py)                                                                                          |
| 8   | [amazon-reviews-fasttext](amazon-reviews-fasttext)                                             | Demonstrate the use of the following experimental module from cleanlab: [fasttext.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/fasttext.py)                                                                                                    |

## Instructions

To run the latest example notebooks, execute the commands below which will install the required libraries in a virtual environment.

It is recommended to run the examples with the latest stable cleanlab release. See `requirements.txt` file.

```console
$ python -m pip install virtualenv
$ python -m venv env
$ source env/bin/activate
$ python -m pip install -r requirements.txt
```

For examples 1-5, you may run the notebooks individually or run the bash script below which will execute and save each notebook.

Bash script:

```console
$ bash ./run_all_notebooks.sh
```

For examples 6-8, please follow the instructions in the `README` of each folder.

## Older Examples

See the `contrib` folder for examples from v1 of cleanlab which may be helpful for reproducing results from the [Confident Learning paper](https://arxiv.org/abs/1911.00068).

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
