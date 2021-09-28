import numpy as np

from ya_glm.opt.from_config.lla_structure import get_transform
from ya_glm.config.base_penalty import get_flavor_info, ElasticNetConfig
from ya_glm.config.penalty import OverlappingSum, SeparableSum, InfimalSum


def get_adaptive_weights(coef_init, expon=1,
                         pertub_init=None, n_samples=None,
                         transform=None):
    """
    Computes the adaptive weights from an initial coefficient.

    Parameters
    ----------
    coef_init: array-like
        The initial cofficient used to set the adative weights.

    perturb_init: str, float, None
        (Optional) Perturbation to add to initial coefficeint e.g. to avoid dividing by zero. If pertub_init='n_samples', will use 1/n_samples.

    n_samples: None, int
        (Optional) The number of training samples used to estimatae the initial coefficient.

    transform: None, callable(coef) -> array-like
        (Optional) Transformation to apply to the coefficient before computing the adatpive weights. E.g. for the adaptive nuclear norm transform should output the singular values.

    Output
    ------
    adpt_weights: array-like
        The adaptive weights.
    """
    # check arguments
    if pertub_init == 'n_samples':
        assert n_samples is not None, \
            "Must provide an value for n_samples"

        pertub_init = 1 / n_samples

    elif pertub_init is None:
        pertub_init = 0.0

    # possibly transform coef_init
    if transform is None:
        values = coef_init
    else:
        values = transform(coef_init)

    # compute adaptiv weights
    return 1 / (_abs(values) + pertub_init) ** expon


def _abs(x):
    """
    Ensures the input is positive and a numpy array.
    """
    return abs(np.array(x, copy=False))


def set_adaptive_weights(penalty, init_data):
    """
    Sets the adaptive weights for a penalty config from the initializer data.

    Parameters
    ----------
    penalty: PenaltyConfig
        The adaptive penalty config object whose weight we want to set. Note this config will be modified in place.

    init_data: dict
        The initializer data; must have keys 'coef_init', 'n_samples'.

    Output
    ------
    penalty: PenaltyConfig
        The penalty with the weights set to the adaptive weights.
    """

    # double check penalty is adaptive
    flavor_type = get_flavor_info(penalty)

    if flavor_type != 'adaptive':
        return penalty

    elif isinstance(penalty, OverlappingSum):
        # set adaptive weights for each penalty in the sum
        new_penalties = {}
        for name, pen in penalty.get_penalties().items():
            pen = set_adaptive_weights(penalty=pen, init_data=init_data)
            new_penalties[name] = pen
        penalty.set_params(**new_penalties)

    elif isinstance(penalty, SeparableSum):
        # set adaptive weights for each penalty in the sum

        groups = penalty.get_groups()
        new_penalties = {}
        for name, pen in penalty.get_penalties().items():

            # subset init data coef
            grp_idxs = groups[name]
            _init_data = {**init_data}
            _init_data['coef'] = init_data['coef'][grp_idxs]

            pen = set_adaptive_weights(penalty=pen,
                                       init_data=_init_data)
            new_penalties[name] = pen

        penalty.set_params(**new_penalties)

    elif isinstance(penalty, InfimalSum):
        raise NotImplementedError("TODO: add!")

    elif isinstance(penalty, ElasticNetConfig):
        raise NotImplementedError("TODO: add!")

    else:

        # get the transformation
        transform = get_transform(penalty)

        adpt_weights = get_adaptive_weights(coef_init=init_data['coef'],
                                            expon=penalty.flavor.expon,
                                            pertub_init=penalty.flavor.pertub_init,
                                            n_samples=init_data['n_samples'],
                                            transform=transform)

        penalty.set_params(weights=adpt_weights)

    return penalty
