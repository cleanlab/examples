# Training a PyTorch CNN model to find label errors in MNIST

This example demonstrates the use of the following module below from cleanlab:

- [cleanlab.experimental.mnist_pytorch.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py)

The code is adapted from cleanlab v1 examples (see `contrib/v1` folder).

## Instructions

Install PyTorch with CUDA. If needed, change the CUDA version in the `cuda_requirements.txt` file and the link below.

```console
$ pip install -r cuda_requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Start Jupyter Lab and run the notebook: `label_errors_mnist_train_cnn.ipynb`
