# Training a Fasttext model on the amazon reviews dataset

This example demonstrates the use of the following module below from cleanlab:

- [cleanlab.models.fasttext.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/models/fasttext.py)

The code is adapted from cleanlab v1 examples (see `contrib/v1` folder).

## Instructions

Install [pigz](https://zlib.net/pigz/) if you system doesn't already have it. This can be done with brew on macOS:

```
$ brew install pigz
```

Run bash script below to download all the data.

```console
$ bash ./download_data.sh
```

Start Jupyter Lab and run the notebook: `fasttext_amazon_reviews.ipynb`
