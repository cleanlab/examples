import numpy as np
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from cleanlab.internal.util import (
    get_num_classes,
    append_extra_datapoint,
    train_val_split,
)


def train_model(
    model,
    X,
    labels,
    cv_n_folds=5,
    X_unlabeled=None,
    X_test=None,
    labels_test=None,
):
    if X_unlabeled is None:
        X_unlabeled = np.array([])

    num_classes = get_num_classes(labels=labels)
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True)

    # Initialize pred_probs array
    pred_probs = np.full((len(labels), num_classes), np.NaN)
    pred_probs_unlabeled = np.full((cv_n_folds, len(X_unlabeled), num_classes), np.NaN)
    model_accuracy = np.full(cv_n_folds, np.NaN)

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

        # compute test accuracy if test set is provided
        if X_test is not None and labels_test is not None:
            curr_model_pred_labels = model_copy.predict(X_test)
            curr_model_accuracy = np.mean(curr_model_pred_labels == labels_test)
            model_accuracy[k] = curr_model_accuracy

    model_accuacy = np.mean(np.array(model_accuracy))
    pred_probs_unlabeled = np.mean(np.array(pred_probs_unlabeled), axis=0)

    return pred_probs, pred_probs_unlabeled, model_accuacy
