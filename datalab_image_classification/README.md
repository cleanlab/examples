# Detecting issues in an image classification dataset (Caltech-256) with Datalab

This example uses cleanlab's Datalab class to audit an image dataset. Here we run Datalab with a Swin Transformer model trained for classification.

There are two notebooks:
- [train_image_classifier.ipynb](train_image_classifier.ipynb) - Trains a Swin Transformer classifier model on a subset of Caltech-256


  - Install dependencies with:

    ```
    pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu116
    ```

- [datalab.ipynb](datalab.ipynb) - Audits the dataset using Datalab applied to outputs from the trained model.
  - Install dependencies with
    ```
    pip install -r requirements-datalab.txt
    ```
