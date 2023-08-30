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

<br>

You can also audit your dataset for the same issues detected by Datalab (and more) without having to: write code, train your own machine learning model, or set up your own interface to the data/results. [Cleanlab Studio](https://cleanlab.ai/studio/?utm_source=github&utm_medium=readme&utm_campaign=clostostudio) does all this for you automatically. For image/text/tabular datasets, [most users](https://cleanlab.ai/love/) obtain better results with [Cleanlab Studio](https://cleanlab.ai/studio/?utm_source=github&utm_medium=readme&utm_campaign=clostostudio) vs. implementing their own solution (and achieve these results [100x faster](https://cleanlab.ai/blog/data-centric-ai/)).

<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/imagenet-cleanlab-studio.png" width=70% height=70%>
</p>
