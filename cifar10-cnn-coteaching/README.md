# Learning with noisy labels on CIFAR-10

This example demonstrates the use of two experimental modules below from Cleanlab:

- [cleanlab.experimental.cifar_cnn.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/cifar_cnn.py)
- [cleanlab.experimental.coteaching.py](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/coteaching.py)

The code and data for this example are taken from the repo below:

- [cgnorthcutt/confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10)

## Instructions

Run bash script below to download all the data.

The following will be saved in the `data` folder:

- [CIFAR-10 train and test images](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10/dataset) (png files)
- [Noisy labels](https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json) (json file)
  - [20% Noise | 40% Sparsity](https://github.com/cgnorthcutt/confidentlearning-reproduce) as defined in the [Confident Learning](https://github.com/cgnorthcutt/confidentlearning-reproduce) paper
- [True labels](https://github.com/cleanlab/examples/raw/master/cifar10/cifar10_true_uncorrupted_labels.npy) (npy file)
- [Pre-computed predicted probabilities from cross-validation](https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy) (npy file)
- [Pre-computed noisy label mask for training dataset](https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/confidentlearning_and_coteaching/results/4_2/train_pruned_conf_joint_only/train_mask.npy) (npy file)

```console
./download_data.sh
```

Run below to train a CNN model with coteaching.

This script stores the output in a log file (`out_4_2.log`) so we can see the resulting test accuracy for each epoch.

```console
# run Confident Learning training with Co-Teaching on labels with 20% label noise
{ time python3 cifar10_train_crossval.py \
	--coteaching \
    	--seed 1 \
	--batch-size 128 \
	--lr 0.001 \
	--epochs 250 \
	--turn-off-save-checkpoint \
	--train-labels data/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json \
	--gpu 1 \
	--dir-train-mask data/confidentlearning_and_coteaching/results/4_2/train_pruned_conf_joint_only/train_mask.npy \
	data/ ; \
} &> out_4_2.log &
tail -f out_4_2.log;
```

## License

Copyright (c) 2017-2022 Curtis Northcutt. Released under the MIT License. See [LICENSE](https://github.com/cgnorthcutt/cleanlab/blob/master/LICENSE) for details.
