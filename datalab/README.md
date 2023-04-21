# Training a Swin Transformer model with PyTorch on a subset of Caltech-256 for auditing with Datalab
This example demonstrates the use of the following module below from cleanlab:

It is split into two notebooks:
- [train_image_classifier.ipynb](train_image_classifier.ipynb) - Trains a Swin Transformer model on a subset of Caltech-256


  - Install dependencies with:

    ```
    pip install -r requirements-train.txt
    ```

- [datalab_subset.ipynb](datalab_subset.ipynb) - Runs the dataset through Datalab with the trained model artifacts.
  - Install dependencies with
    ```
    pip install -r requirements-datalab.txt
    ```
