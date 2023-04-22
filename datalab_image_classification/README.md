# Detecting issues in an image classification dataset (Caltech-256) with Datalab and a Swin Transformer model.

This example demonstrates cleanlab's Datalab class for auditing an image dataset.

It is split into two notebooks:
- [train_image_classifier.ipynb](train_image_classifier.ipynb) - Trains a Swin Transformer classifier model on a subset of Caltech-256


  - Install dependencies with:

    ```
    pip install -r requirements-train.txt
    ```

- [datalab.ipynb](datalab.ipynb) - Audits the dataset using Datalab applied to outputs from the trained model.
  - Install dependencies with
    ```
    pip install -r requirements-datalab.txt
    ```
