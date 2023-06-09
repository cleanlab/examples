# Detecting label issues in an object detection dataset

This example demonstrates how to use cleanlab to detect label errors in an object detection dataset.
We train a neural network with the Detectron2 library with a model from the COCO-Detection model zoo.

There are two notebooks:

| Notebook | Description |
| --- | --- |
| [detectron2_training.ipynb](detectron2_training.ipynb) | Trains an object detection model on a training set of images and produces predictions on a held-out validation set. |
| [detectron2_training-kfold.ipynb](detectron2_training-kfold.ipynb) | Trains an object detection model on a training set of images via k-fold cross-validation to produce predictions on that same training set. |


## Setup

Before running the notebooks, make sure you install dependencies (ideally in a fresh virtual environment) with:

```bash
# # Set up a virtual environment
# python -m venv venv
# source venv/bin/activate

pip install -r requirements.txt
```

Also, make sure you have `wget` installed.
