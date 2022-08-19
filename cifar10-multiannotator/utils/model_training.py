import sys

sys.path.insert(0, "../")

import numpy as np
import pandas as pd
import pickle
import datetime
from pathlib import Path
import cleanlab
from .cross_validation_autogluon import cross_val_predict_autogluon_image_dataset


def train_model(model_type, data, model_results_folder, *, num_cv_folds=5, verbose=1, epochs=1, holdout_frac=0.2, time_limit=60, random_state=123):
    # run xvalidation
    print("----")
    print(f"Running cross-validation for model: {model_type}")

    MODEL_PARAMS = {
        "model": model_type,
        "epochs": epochs,
        "holdout_frac": holdout_frac,
    }

    # results of cross-validation will be saved to pickle files for each model/fold
    _ = \
        cross_val_predict_autogluon_image_dataset(
            dataset=data,
            out_folder=f"{model_results_folder}_{model_type}/", # save results of cross-validation in pickle files for each fold
            n_splits=num_cv_folds,
            model_params=MODEL_PARAMS,
            time_limit=time_limit,
            random_state=random_state,
        )

# load pickle file util
def _load_pickle(pickle_file_name, verbose=1):
    """Load pickle file"""
    if verbose:
        print(f"Loading {pickle_file_name}")
    with open(pickle_file_name, 'rb') as handle:
        out = pickle.load(handle)
    return out


def sum_xval_folds(model, model_results_folder, num_cv_folds=5, verbose=1, **kwargs):
    # get original label name to idx mapping
    label_name_to_idx_map = {'airplane': 0,
                         'automobile': 1,
                         'bird': 2,
                         'cat': 3,
                         'deer': 4,
                         'dog': 5,
                         'frog': 6,
                         'horse': 7,
                         'ship': 8,
                         'truck': 9}
    results_list = []
    
    # get shapes of arrays (this is dumb way to do it what is better?)
    pred_probs_shape = []
    labels_shape = []
    for split_num in range(num_cv_folds):

        out_subfolder = f"{model_results_folder}_{model}/split_{split_num}/"

        # pickle file name to read
        get_pickle_file_name = (
            lambda object_name: f"{out_subfolder}_{object_name}_split_{split_num}"
        )

        # NOTE: the "test_" prefix in the pickle name correspond to the "test" split during cross-validation.
        pred_probs_split = _load_pickle(get_pickle_file_name("test_pred_probs"), verbose=verbose)
        labels_split = _load_pickle(get_pickle_file_name("test_labels"), verbose=verbose)

        pred_probs_shape.append(pred_probs_split)
        labels_shape.append(labels_split)

    pred_probs_shape = np.vstack(pred_probs_shape)
    labels_shape = np.hstack(labels_shape)
        
    pred_probs = np.zeros_like(pred_probs_shape)
    labels = np.zeros_like(labels_shape)
    images = np.empty((labels_shape.shape[0],) ,dtype=object)

    for split_num in range(num_cv_folds):

        out_subfolder = f"{model_results_folder}_{model}/split_{split_num}/"

        # pickle file name to read
        get_pickle_file_name = (
            lambda object_name: f"{out_subfolder}_{object_name}_split_{split_num}"
        )

        # NOTE: the "test_" prefix in the pickle name correspond to the "test" split during cross-validation.
        pred_probs_split = _load_pickle(get_pickle_file_name("test_pred_probs"), verbose=verbose)
        labels_split = _load_pickle(get_pickle_file_name("test_labels"), verbose=verbose)
        images_split = _load_pickle(get_pickle_file_name("test_image_files"), verbose=verbose)
        indices_split = _load_pickle(get_pickle_file_name("test_indices"), verbose=verbose)
        indices_split = np.array(indices_split)
        
        pred_probs[indices_split] = pred_probs_split
        labels[indices_split] = labels_split
        images[indices_split] = np.array(images_split)

    return pred_probs, labels, images