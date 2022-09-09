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