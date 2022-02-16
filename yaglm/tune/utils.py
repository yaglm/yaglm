import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


def train_validation_idxs(n_samples, test_size=0.2, shuffle=True,
                          random_state=None,
                          y=None, classifier=False):
    """
    Returns the indices for a train/validation split using sklearn.model_selection.train_test_split. Unlike the sklearn's function, shuffle works with classifiers.
    

    Parameters
    ----------
    n_samples: int
        Number of samples.
        
    test_size: float, int or None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If None, will be set to 0.25.

    shuffle: bool
        Whether or not to randomly shuffle the sample indices. If False, the training indices will be the first set of indices.
    
    random_state: None, int
        The seed for shuffling the indices.

    y: None, array-like
        (Optional) The response data. Used to for stratified sampling for classifiers.
        
    classifier: bool
        Whether or not we are fitting a classifier an the sample splitting should be done in a stratified manner.

    Output
    ------
    train, test
    
    train: array-like
        The indices of the training samples.
        
    test: array-like
        The indices of the test samples.
    """
    sample_idxs = np.arange(n_samples)
    
    # pre-shulffle the sample indices
    if shuffle:
        rng = check_random_state(random_state)
        rng.shuffle(sample_idxs)
    
    # stratify samples for classifiers
    stratify = None
    if classifier:
        assert y is not None, \
            'Must provide y for stratifying a classifier'
        stratify = y[sample_idxs]

    return train_test_split(sample_idxs,
                            test_size=test_size,
                            stratify=stratify)


def get_tune_param_df(tune_results):
    """
    Gets a pd.DataFrame of the tuning parameter settings.

    Parameters
    ----------
    tune_results: dict
        The tune_results_ attribute.

    Output
    ------
    tune_params: pd.DataFrame
        The tuning parameter settings
    """
    return pd.DataFrame(list(pd.DataFrame(tune_results['params']).values))
