# Training a Fasttext model on the amazon reviews dataset

This example demonstrates how to wrap a Fasttext model to be sklearn-compatible for direct compatibility with `cleanlab.classification`.  The wrapper code is in **fasttext_wrapper.py** and the notebook demonstrates how to apply it to a text classification dataset.

## Instructions

Install [pigz](https://zlib.net/pigz/) if you system doesn't already have it. This can be done with brew on macOS:

```
$ brew install pigz
```

Run bash script below to download all the data.

```console
$ bash ./download_data.sh
```

Start Jupyter and run the notebook: `fasttext_amazon_reviews.ipynb`
