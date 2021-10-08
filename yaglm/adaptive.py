import numpy as np
from ya_glm.opt.from_config.transforms import \
    get_flavored_transforms, get_parent_key


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

    coef_init = init_data['coef']
    n_samples = init_data['n_samples']

    # get transform functions for all adaptively flavored penalties
    transforms, flavors, flavor_keys = get_flavored_transforms(penalty,
                                                               kind='adaptive')

    # set adaptive weights
    for idx, flav_key in enumerate(flavor_keys):
        pen_key = get_parent_key(flav_key)
        flav_config = flavors[idx]

        # Compute adaptive weights
        adpt_weights = get_adaptive_weights(coef_init=coef_init,
                                            expon=flav_config.expon,
                                            pertub_init=flav_config.pertub_init,
                                            n_samples=n_samples,
                                            transform=transforms[idx])

        # Set penalty's weights
        flav_name = flav_key.split('__')[-1]
        if pen_key == '':
            stub = ''
        else:
            stub = pen_key + '__'

        if flav_name == 'flavor':
            weights_key = stub + 'weights'
        else:
            # ElasticNet case
            sum_name = flav_name.split('_')[0]
            weights_key = stub + sum_name + '_weights'

        penalty.set_params(**{weights_key: adpt_weights})

    return penalty
