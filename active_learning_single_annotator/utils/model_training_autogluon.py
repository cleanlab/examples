""" Helper methods to train AutoML model for image classification. """

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
from autogluon.multimodal import MultiModalPredictor

from cleanlab.internal.util import get_num_classes


def predict_autogluon_classification(
    dataset: ImageDataset,
    out_folder: str = "./model_training_run/",
    *,
    X_predict=None,
    df_predict=None,
    hypterparameters={},
    time_limit=30,
):
    
    if X_predict is None:
        X_predict = np.array([])

    num_classes = get_num_classes(labels=dataset.label.values)
    
    save_path = None if out_folder is None else f'{out_folder}'
    predictor = MultiModalPredictor(label="label", 
                                    path=save_path, 
                                    problem_type="classification", 
                                    warn_if_exist=False)

    # train model on train indices in this split
    predictor.fit(
        train_data=dataset,
        time_limit=time_limit, # seconds
        hyperparameters=hypterparameters,
    ) # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    pred_probs_unlabeled = None
    
    if len(X_predict) > 0:
        pred_probs_unlabeled = predictor.predict_proba(X_predict)
    elif df_predict is not None:
        pred_probs_unlabeled = predictor.predict_proba(df_predict, as_pandas=False)
    
    return predictor, pred_probs_unlabeled


def cross_val_predict_autogluon_classification(
    dataset: ImageDataset,
    out_folder: str = "./cross_val_model_training_run/",
    *,
    cv_n_folds: int = 5,
    df_predict=None,
    hypterparameters={},
    time_limit=30,
):

    if X_predict is None:
        X_predict = np.array([])

    num_classes = get_num_classes(labels=dataset.label.values)
    
    # stratified K-folds
    skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=False)
    skf_splits = [
        [train_index, test_index]
        for train_index, test_index in skf.split(X=dataset, y=dataset.label)
    ]

    # Initialize pred_probs array
    pred_probs_full = []
    labels_full = []
    images_full = []
    n = df_predict.shape[0] if df_predict is not None else len(X_predict)
    pred_probs_unlabeled = np.full((cv_n_folds, n, num_classes), np.NaN)
    
    # run cross-validation
    for split_num, split in enumerate(skf_splits):
        print("----")
        print(f"Running Cross-Validation on Split: {split_num}")

        # split from stratified K-folds
        train_index, test_index = split

        # init model
        save_path = None if out_folder is None else f'{out_folder}/_split{split_num}'
        predictor = MultiModalPredictor(label="label", 
                                        path=save_path, 
                                        problem_type="classification", 
                                        warn_if_exist=False)

        # train model on train indices in this split
        predictor.fit(
            train_data=dataset.iloc[train_index], # you can use train_data_byte as well
            time_limit=time_limit, # seconds
            hyperparameters=hypterparameters,
        ) # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

        # predicted probabilities for test split
        pred_probs = predictor.predict_proba(
            data=dataset.iloc[test_index], as_pandas=False
        )
        
        labels = dataset.iloc[test_index].label.values
        images = dataset.iloc[test_index].label.values
        
        pred_probs_full.append(pred_probs)
        labels_full.extend(labels)
        images_full.extend(images)
        
        if len(X_predict) > 0:
            curr_pred_probs_unlabeled = predictor.predict_proba(X_predict)
            pred_probs_unlabeled[split_num] = curr_pred_probs_unlabeled
        elif df_predict is not None:
            curr_pred_probs_unlabeled = predictor.predict_proba(df_predict, as_pandas=False)
            pred_probs_unlabeled[split_num] = curr_pred_probs_unlabeled
    
    pred_probs_unlabeled = np.mean(np.array(pred_probs_unlabeled), axis=0)
    pred_probs = np.vstack(pred_probs_full)
    labels = np.array(labels_full)
    images = np.array(images_full)
    return pred_probs, pred_probs_unlabeled, labels, images