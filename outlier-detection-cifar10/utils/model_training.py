""" Helper functions for training image classification models with AutoGluon and using cross-validation. """

import sys

sys.path.insert(0, "../")

import numpy as np
import pandas as pd
import pickle
import datetime
import os
from pathlib import Path
from typing import Dict, Tuple

import cleanlab
from autogluon.vision import ImagePredictor, ImageDataset
from sklearn.model_selection import StratifiedKFold


def cross_val_predict_autogluon_image_dataset(
    dataset: ImageDataset,
    out_folder: str = "./cross_val_predict_run/",
    *,
    n_splits: int = 5,
    model_params: Dict = {"epochs": 1, "holdout_frac": 0.2},
    ngpus_per_trial: int = 1,
    time_limit: int = 7200,
    random_state: int = 123,
    verbose: int = 0,
) -> Tuple:
    """Run stratified K-folds cross-validation with AutoGluon image model.

    Parameters
    ----------
    dataset : gluoncv.auto.data.dataset.ImageClassificationDataset
      AutoGluon dataset for image classification.

    out_folder : str, default="./cross_val_predict_run/"
      Folder to save cross-validation results. Save results after each split (each K in K-fold).

    n_splits : int, default=3
      Number of splits for stratified K-folds cross-validation.

    model_params : Dict, default={"epochs": 1, "holdout_frac": 0.2}
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    ngpus_per_trial : int, default=1
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    time_limit : int, default=7200
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    random_state : int, default=123
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    Returns
    -------
    None

    """

    # stratified K-folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    skf_splits = [
        [train_index, test_index]
        for train_index, test_index in skf.split(X=dataset, y=dataset.label)
    ]
    
    for split_num, split in enumerate(skf_splits):

        print("----")
        print(f"Running Cross-Validation on Split: {split_num}")

        # split from stratified K-folds
        train_index, test_index = split

        # init model
        predictor = ImagePredictor(verbosity=0)

        # train model on train indices in this split
        predictor.fit(
            train_data=dataset.iloc[train_index],
            ngpus_per_trial=ngpus_per_trial,
            hyperparameters=model_params,
            time_limit=time_limit,
            random_state=random_state,
        );

        # predict on test indices in this split

        # predicted probabilities for test split
        pred_probs = predictor.predict_proba(
            data=dataset.iloc[test_index], as_pandas=False
        )

        # predicted features (aka embeddings) for test split
        # why does autogluon predict_feature return array of array for the features?
        # need to use stack to convert to 2d array (https://stackoverflow.com/questions/50971123/converty-numpy-array-of-arrays-to-2d-array)
        pred_features = np.stack(
            predictor.predict_feature(data=dataset.iloc[test_index], as_pandas=False)[
                :, 0
            ]
        )

        # save output of model + split in pickle file

        out_subfolder = f"{out_folder}split_{split_num}/"

        try:
            os.makedirs(out_subfolder, exist_ok=False)
        except OSError:
            print(f"Folder {out_subfolder} already exists!")
        finally:

            # save to pickle files

            get_pickle_file_name = (
                lambda object_name: f"{out_subfolder}_{object_name}_split_{split_num}"
            )

            _save_to_pickle(pred_probs, get_pickle_file_name("test_pred_probs"))
            _save_to_pickle(pred_features, get_pickle_file_name("test_pred_features"))
            _save_to_pickle(
                dataset.iloc[test_index].label.values,
                get_pickle_file_name("test_labels"),
            )
            _save_to_pickle(
                dataset.iloc[test_index].image.values,
                get_pickle_file_name("test_image_files"),
            )
            _save_to_pickle(test_index, get_pickle_file_name("test_indices"))

        # save model trained on this split
        predictor.save(f"{out_subfolder}predictor.ag")

    return predictor


def _save_to_pickle(object, pickle_file_name):
    """Save object to pickle file"""

    print(f"Saving {pickle_file_name}")

    # save to pickle file
    with open(pickle_file_name, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_model(model_type, data, model_results_folder, *, num_cv_folds=5, verbose=0, epochs=1, holdout_frac=0.2, time_limit=60, random_state=123):
    """Trains AutoGluon image model with stratified K-folds cross-validation and saves data in model_results_folder.

    Parameters
    ----------
    model_type: str
      Type of backend architecture for Autogluon
      
    data : gluoncv.auto.data.dataset.ImageClassificationDataset
      AutoGluon dataset for image classification.

    model_results_folder : str
      Folder to save cross-validation results. Save results after each split (each K in K-fold).

    num_cv_folds : int, default=5
      Number of splits for stratified K-folds cross-validation.

    model_params : Dict, default={"epochs": 1, "holdout_frac": 0.2, "verbose": 1}
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    time_limit : int, default=7200
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    random_state : int, default=123
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    Returns
    -------
    None

    """
    
    # run xvalidation
    print("----")
    print(f"Running cross-validation for model: {model_type}")

    MODEL_PARAMS = {
        "model": model_type,
        "epochs": epochs,
        "holdout_frac": holdout_frac,
    }

    # results of cross-validation will be saved to pickle files for each model/fold
    predictor = \
        cross_val_predict_autogluon_image_dataset(
            dataset=data,
            out_folder=f"{model_results_folder}_{model_type}/", # save results of cross-validation in pickle files for each fold
            n_splits=num_cv_folds,
            model_params=MODEL_PARAMS,
            time_limit=time_limit,
            random_state=random_state,
            verbose=verbose,
        )
    return predictor

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
    features_shape = []
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
        test_pred_features_split = _load_pickle(get_pickle_file_name("test_pred_features"), verbose=verbose)

        pred_probs_shape.append(pred_probs_split)
        features_shape.append(test_pred_features_split)
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
