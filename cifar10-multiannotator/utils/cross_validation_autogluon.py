import pandas as pd
import numpy as np
from autogluon.vision import ImagePredictor, ImageDataset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Tuple
import pickle
import os


def cross_val_predict_autogluon_image_dataset(
    dataset: ImageDataset,
    out_folder: str = "./cross_val_predict_run/",
    *,
    n_splits: int = 5,
    model_params: Dict = {"epochs": 1, "holdout_frac": 0.2},
    ngpus_per_trial: int = 1,
    time_limit: int = 7200,
    random_state: int = 123,
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

    # run cross-validation
    for split_num, split in enumerate(skf_splits):

        print("----")
        print(f"Running Cross-Validation on Split: {split_num}")

        # TODO: add logic to skip if results from model/fold already exists from a previous run

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
        )

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

    return None


def _save_to_pickle(object, pickle_file_name):
    """Save object to pickle file"""

    print(f"Saving {pickle_file_name}")

    # save to pickle file
    with open(pickle_file_name, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
