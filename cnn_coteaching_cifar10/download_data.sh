#!/bin/bash

# path to save the data
DATA_PATH=data/

# create data folder if it doesn't exist
mkdir -p $DATA_PATH

# download data
wget -P $DATA_PATH https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/dataset/train_part_1_of_2.tar.gz;
wget -P $DATA_PATH https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/dataset/train_part_2_of_2.tar.gz;
wget -P $DATA_PATH https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/dataset/test.tar.gz;

# extract files
tar -xzvf "${DATA_PATH}train_part_1_of_2.tar.gz"  -C $DATA_PATH
tar -xzvf "${DATA_PATH}train_part_2_of_2.tar.gz"  -C $DATA_PATH
tar -zxvf "${DATA_PATH}test.tar.gz" -C $DATA_PATH

# download noisy labels for training dataset
wget -P $DATA_PATH https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json;
wget -P $DATA_PATH https://github.com/cleanlab/examples/raw/master/cifar10/cifar10_true_uncorrupted_labels.npy;

# download pre-computed pred_probs for training dataset (trained with noisy labels)
PRED_PROBS_PATH=${DATA_PATH}cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_2/
mkdir -p $PRED_PROBS_PATH
wget -P $PRED_PROBS_PATH https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy;

# download pre-computed training mask
TRAIN_MASK_PATH=${DATA_PATH}confidentlearning_and_coteaching/results/4_2/train_pruned_conf_joint_only/
mkdir -p $TRAIN_MASK_PATH
wget -P $TRAIN_MASK_PATH https://github.com/cgnorthcutt/confidentlearning-reproduce/raw/master/cifar10/confidentlearning_and_coteaching/results/4_2/train_pruned_conf_joint_only/train_mask.npy;