## Instructions

```console
$ pip install -r requirements.txt
```

Change the version of `torch` and `torchvision` if necessary.

Start Jupyter Lab and run the notebook: `detecting_spurious_correlations.ipynb`

In this tutorial, we demonstrate the impact of training a model on a dataset with spurious correlations, focusing on a scenario where one class consists predominantly of dark images. We then compare the model's performance on a dataset free from such spurious correlations. Finally, the tutorial shows how these spurious correlation issues can be easily detected using `Datalab`.