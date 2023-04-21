# cleanlab Examples

This repo contains code examples that demonstrate how to use [cleanlab](https://github.com/cleanlab/cleanlab) with specific real-world models/datasets, how its underlying algorithms work, how to get better results via advanced functionality, and how to train certain models used in some cleanlab tutorials.  

To quickly learn how to run cleanlab on your own data, first check out the [quickstart tutorials](https://docs.cleanlab.ai/) before diving into the examples below.

## Table of Contents

|     | Example                                                                                        | Description                                                                                                                                                                                                                                                                  |
| --- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | [datalab](datalab_image_classification)                              | Use Datalab to audit (a subset of) the Caltech-256 image classification dataset for various issues.                        |
| 1   | [find_label_errors_iris](find_label_errors_iris/find_label_errors_iris.ipynb)                                         | Find label errors introduced into the Iris classification dataset.                                                 |
| 2   | [classifier_comparison](classifier_comparison/classifier_comparison.ipynb)                                     | Use CleanLearning to train 10 different classifiers on 4 dataset distributions with label errors.                     |
| 3   | [hyperparameter_optimization](hyperparameter_optimization/hyperparameter_optimization.ipynb)                                       | Hyperparameter optimization to find the best settings of CleanLearning's optional parameters.                                          |
| 4   | [simplifying_confident_learning](simplifying_confident_learning/simplifying_confident_learning.ipynb) | Straightforward implementation of Confident Learning algorithm with raw numpy code.              |
| 5   | [visualizing_confident_learning](visualizing_confident_learning/visualizing_confident_learning.ipynb)                   | See how cleanlab estimates parameters of the label error distribution (noise matrix).             |
| 6   | [find_tabular_errors](find_tabular_errors/find_tabular_errors.ipynb) | Handle mislabeled [tabular data](https://github.com/cleanlab/s/blob/master/student-grades-demo.csv) to improve a XGBoost classifier.                         |
| 7   | [cnn_mnist](cnn_mnist/find_label_errors_cnn_mnist.ipynb)                                                                         | Finding label errors in MNIST image data with a [Convolutional Neural Network](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py).                                                                                          |
| 8   | [huggingface_keras_imdb](huggingface_keras_imdb/huggingface_keras_imdb.ipynb)                                             |  CleanLearning for text classification with Keras Model + pretrained BERT backbone and Tensorflow Dataset.         |
| 9   | [fasttext_amazon_reviews](fasttext_amazon_reviews/fasttext_amazon_reviews.ipynb)                         | Finding label errors in Amazon Reviews text dataset using a cleanlab-compatible [FastText model](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/models/fasttext.py).                                                                                                    |
| 10   | [multiannotator_cifar10](multiannotator_cifar10/multiannotator_cifar10.ipynb)                                             | Iteratively improve consensus labels and trained classifier from data labeled by multiple annotators.                                                            |
| 11  | [active_learning_multiannotator](active_learning_multiannotator/active_learning.ipynb)                                             | Improve a classifier model by iteratively collecting additional labels from data annotators. This active learning pipeline considers data labeled in batches by multiple (imperfect) annotators.                                                             |
| 12  | [active_learning_single_annotator](active_learning_single_annotator/active_learning_single_annotator.ipynb)                                             | Improve a classifier model by iteratively labeling batches of currently-unlabeled data.  This demonstrates a standard active learning pipeline with *at most one label* collected for each example (unlike our multi-annotator active learning notebook which allows re-labeling).                                                            |
| 13  | [outlier_detection_cifar10](outlier_detection_cifar10/outlier_detection_cifar10.ipynb)                                             | Train AutoML for image classification and use it to detect out-of-distribution images.                                                                                                 |
| 14  | [multilabel_classification](multilabel_classification/image_tagging.ipynb)                                               | Find label errors in an image tagging dataset ([CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) using a [Pytorch model](multilabel_classification/pytorch_network_training.ipynb) you can easily train for multi-label classification. |
| 15  | [entity_recognition](entity_recognition/entity_recognition_training.ipynb)                                             | Train Transformer model  for Named Entity Recognition and produce out-of-sample `pred_probs` for **cleanlab.token_classification**.      |
| 16  | [transformer_sklearn](transformer_sklearn/transformer_sklearn.ipynb)                                             | How to use `KerasWrapperModel` to make any Keras model sklearn-compatible, demonstrated here for a BERT Transformer.      |
| 17  | [cnn_coteaching_cifar10](cnn_coteaching_cifar10)                                               | Train a [Convolutional Neural Network](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/cifar_cnn.py) on noisily labeled Cifar10 image data using cleanlab with [coteaching](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/coteaching.py).  |


## Instructions

To run the latest example notebooks, execute the commands below which will install the required libraries in a virtual environment.

```console
$ python -m pip install virtualenv
$ python -m venv cleanlab-examples  # creates a new venv named cleanlab-examples
$ source cleanlab-examples/bin/activate
$ python -m pip install -r requirements.txt
```

Alternatively you can only install those dependencies required for a specific example by calling `pip install -r requirements.txt` inside the subfolder for that example (each example's subfolder contains a separate `requirements.txt` file).

It is recommended to run the examples with the latest stable cleanlab release (`pip install cleanlab`). 
However be aware that notebooks in the master branch of this repository are assumed to correspond to master branch version of cleanlab, hence some very-recently added examples may require you to instead install the developer version of cleanlab (`pip install git+https://github.com/cleanlab/cleanlab.git`). 
To see the examples corresponding to specific version of cleanlab, check out the [Tagged Releases](https://github.com/cleanlab/examples/releases) of this repository (e.g. the examples for cleanlab v2.1.0 are [here](https://github.com/cleanlab/examples/tree/v2.1.0)).

### Running all examples

You may run the notebooks individually or run the bash script below which will execute and save each notebook (for examples: 1-7). Note that before executing the script to run all notebooks for the first time you will need to create a jupyter kernel named `cleanlab-examples`. Be sure that you have already created and activated the virtual environment (steps provided above) before running the following command to create the jupyter kernel.

```console
$ python -m ipykernel install --user --name=cleanlab-examples
```

Bash script to run all notebooks:

```console
$ bash ./run_all_notebooks.sh
```

### Older Examples

For running older versions of cleanlab, look at the [Tagged Releases](https://github.com/cleanlab/examples/releases) of this repository to see the corresponding older versions of the example notebooks (e.g. the examples for cleanlab v2.0.0 are [here](https://github.com/cleanlab/examples/tree/v2.0.0)). 

See the `contrib` folder for examples from v1 of cleanlab which may be helpful for reproducing results from the [Confident Learning paper](https://arxiv.org/abs/1911.00068).

## License

Copyright (c) 2017 Cleanlab Inc.

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
