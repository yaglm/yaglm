import numpy as np


def get_class_weights_binary(y):
    """
    Gets the class weights for binary classes. The class weights are inversely proportional to the number of samples in each class. We follow sklearn's convention of n_samples / (n_classes * np.bincount(y)).
    
    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The binary class labels.
    
    Output
    ------
    class_weights: array-like, (n_samples, )
    """
    assert y.ndim == 1

    prop_1 = y.mean()

    # weights are 1 / class_proportion
    weight_1 = 1 / prop_1
    weight_0 = 1 / (1 - prop_1)
    
    class_weights = np.ones(len(y))
    one_mask = y == 1
    class_weights[~one_mask] = weight_0
    class_weights[one_mask] = weight_1
    
    # normalize to agree with sklean
    class_weights /= 2
    
    return class_weights


def class_weights_indicators(y):
    """
    Gets the class weights for a matrix of indicator vectors.
    The class weights are inversely proportional to the number of samples in each class. We follow sklearn's convention of n_samples / (n_classes * np.bincount(y)).
    
    Parameters
    ----------
    y: array-like, shape (n_samples, n_responses)
        The indicator vectors of the classes.
    
    Output
    ------
    weights: array-like, (n_samples, )
    """
    assert y.ndim == 2

    counts = y.sum(axis=0).A1
    props = counts / counts.sum()
    weights = 1 / props
    
    class_weights = y @ weights
    
    class_weights /= len(counts)
    return class_weights
    

def get_sample_weight_balanced_classes(y, sample_weight=None):
    if y.ndim == 1:
        class_weights = get_class_weights_binary(y)
    elif y.ndim == 2:
        class_weights = class_weights_indicators(y)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    return sample_weight * class_weights
