""" Helper methods to train AutoML model for image classification. """

import sys

sys.path.insert(0, "../")

from pathlib import Path
from typing import Dict, Tuple

import cleanlab
from gluoncv.auto.data.dataset import ImageClassificationDataset
from autogluon.multimodal import MultiModalPredictor


def train(dataset: ImageClassificationDataset,
    out_folder: str = "./model_training_run/",
    hypterparameters={},
    time_limit=30,
):
    """ Takes in an image dataset and fits a MultiModalPredictor to it. Returns fitted predictor."""    
    save_path = None if out_folder is None else f'{out_folder}'
    predictor = MultiModalPredictor(label="label", 
                                    path=save_path, 
                                    problem_type="classification", 
                                    warn_if_exist=False)

    predictor.fit(
        train_data=dataset,
        time_limit=time_limit, # seconds
        hyperparameters=hypterparameters,
    )
  
    return predictor