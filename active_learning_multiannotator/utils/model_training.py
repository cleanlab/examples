"""
This file defines a function to fit a cross-validation model and produce out-of-sample
predicted probabilites for examples in active learning.
"""
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from cleanlab.internal.util import (
    get_num_classes,
    append_extra_datapoint,
    train_val_split,
)


def fit_predict_proba(
    model,
    X,
    labels,
    cv_n_folds=5,
    X_unlabeled=None,
):
    """In your applications, replace this with your code to train model on (X, labels) and
    produce held-out predictions for X and X_unlabeled.
    `labels` here is an array of shape (num_examples,) which contains consensus labels for
    each examples which are derived by aggregating all collected labels from multiple annotators
    for that given example into one.
    Held-out predictions for X can be produced by training via cross-validation."""
    if X_unlabeled is None:
        X_unlabeled = np.array([])

    num_classes = get_num_classes(labels=labels)
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True)

    # Initialize pred_probs array
    pred_probs = np.full((len(labels), num_classes), np.NaN)
    pred_probs_unlabeled = np.full((cv_n_folds, len(X_unlabeled), num_classes), np.NaN)

    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X=X, y=labels)):
        # fresh untrained copy of the model
        model_copy = sklearn.base.clone(model)

        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv, s_train_cv, s_holdout_cv = train_val_split(
            X, labels, cv_train_idx, cv_holdout_idx
        )

        # Fit classifier clf to training set, predict on holdout set, and update pred_probs.
        model_copy.fit(X_train_cv, s_train_cv)
        pred_probs_cv = model_copy.predict_proba(X_holdout_cv)

        pred_probs[cv_holdout_idx] = pred_probs_cv

        # compute pred_probs for unlabeled examples
        if len(X_unlabeled) > 0:
            curr_pred_probs_unlabeled = model_copy.predict_proba(X_unlabeled)
            pred_probs_unlabeled[k] = curr_pred_probs_unlabeled

    pred_probs_unlabeled = np.mean(np.array(pred_probs_unlabeled), axis=0)

    return pred_probs, pred_probs_unlabeled
