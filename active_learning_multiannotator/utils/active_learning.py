import numpy as np
import pandas as pd


def get_idx_to_relabel(
    active_learning_score,
    num_examples_to_relabel,
    active_learning_score_unlabeled=None,
):
    if active_learning_score_unlabeled is None:
        active_learning_score_unlabeled = np.array([])

    num_labeled = len(active_learning_score)
    active_learning_score_combined = np.concatenate(
        (active_learning_score, active_learning_score_unlabeled)
    )

    if num_examples_to_relabel > len(active_learning_score_combined):
        raise ValueError(
            "num_examples_to_relabel is larger than the total number of examples available"
        )

    relabel_idx_combined = np.argsort(active_learning_score_combined)[
        :num_examples_to_relabel
    ]
    relabel_idx = relabel_idx_combined[relabel_idx_combined < num_labeled]
    relabel_idx_unlabeled = (
        relabel_idx_combined[relabel_idx_combined >= num_labeled] - num_labeled
    )

    return relabel_idx, relabel_idx_unlabeled


def setup_next_iter_data(
    multiannotator_labels,
    relabel_idx,
    relabel_idx_unlabeled,
    X,
    X_unlabeled,
    pred_probs,
    pred_probs_unlabeled,
    extra_labels=None,
    extra_labels_unlabeled=None,
):

    multiannotator_labels = pd.concat(
        (
            multiannotator_labels,
            pd.DataFrame(
                np.full(
                    (len(relabel_idx_unlabeled), multiannotator_labels.shape[1]), np.NaN
                )
            ),
        ),
        ignore_index=True,
    )

    relabel_idx_combined = np.concatenate(
        (
            relabel_idx,
            np.array(range(len(X), len(X) + len(relabel_idx_unlabeled))),
        )
    ).astype(int)

    X_new = X_unlabeled[relabel_idx_unlabeled, :]
    X = np.concatenate((X, X_new))
    X_unlabeled = np.delete(X_unlabeled, relabel_idx_unlabeled, axis=0)

    pred_probs_new = pred_probs_unlabeled[relabel_idx_unlabeled, :]
    pred_probs = np.concatenate((pred_probs, pred_probs_new))
    pred_probs_unlabeled = np.delete(
        pred_probs_unlabeled, relabel_idx_unlabeled, axis=0
    )

    if extra_labels is not None:
        extra_labels_new = extra_labels_unlabeled[relabel_idx_unlabeled, :]
        extra_labels = np.concatenate((extra_labels, extra_labels_new))
        extra_labels_unlabeled = np.delete(
            extra_labels_unlabeled, relabel_idx_unlabeled, axis=0
        )

    return (
        multiannotator_labels,
        relabel_idx_combined,
        X,
        X_unlabeled,
        pred_probs,
        pred_probs_unlabeled,
        extra_labels,
        extra_labels_unlabeled,
    )


def add_new_annotator(multiannotator_labels, extra_labels, relabel_idx):
    def get_random_label(annotator_labels):
        annotator_labels = annotator_labels[~np.isnan(annotator_labels)]
        return np.random.choice(annotator_labels)

    complete_labels_subset = extra_labels[relabel_idx]
    new_annotator_labels = np.apply_along_axis(
        get_random_label, axis=1, arr=complete_labels_subset
    )

    # create new column
    new_annotator = np.full(len(multiannotator_labels), np.nan)
    new_annotator[relabel_idx] = new_annotator_labels

    new_idx = np.max(list(multiannotator_labels.columns)) + 1
    multiannotator_labels[new_idx] = new_annotator

    return multiannotator_labels
