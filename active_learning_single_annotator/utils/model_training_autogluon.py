""" Helper method to train AutoML model for image classification. """
from typing import Dict

import cleanlab
from gluoncv.auto.data.dataset import ImageClassificationDataset
from autogluon.multimodal import MultiModalPredictor


def train(
    dataset,
    out_folder: str = "./model_training_run/",
    hyperparameters: Dict[Any, Any] ={},
    time_limit=30,
):
    """ Takes in an image dataset and fits a MultiModalPredictor to it. Returns fitted predictor.
    
    Parameters
    ----------
    dataset: ImageClassificationDataset or pd.DataFrame
        Takes in a dataset object with "label" and data columns where for every row 'i', label[i] correspods to data columns[i]
    
    out_folder: str
        Location where to save the trained predictor. If "None" autogluon still saves trained predictor in preset folder.
        
    hypterparameters: Dict
        Model training hypterparameters for `fit()` function.
        
    time_limit: int
        Amount of time in seconds to train the model.
        
    """    
    save_path = None if out_folder is None else f'{out_folder}'
    predictor = MultiModalPredictor(label="label", 
                                    path=save_path, 
                                    problem_type="classification", 
                                    warn_if_exist=False)

    predictor.fit(
        train_data=dataset,
        time_limit=time_limit, # seconds
        hyperparameters=hyperparameters,
    )
  
    return predictor
