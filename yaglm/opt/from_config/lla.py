from ya_glm.config.penalty_utils import build_penalty_tree, extract_penalties,\
     extract_flavors_and_pens, get_parent_key, get_enet_sum_name
from ya_glm.config.base_penalty import ElasticNetConfig, WithFlavorPenSeqConfig
from ya_glm.opt.from_config.penalty import get_outer_nonconvex_func
from ya_glm.opt.from_config.transforms import get_flavored_transforms
from ya_glm.autoassign import autoassign
from ya_glm.opt.base import Func


def get_lla_transformer(penalty):
    """
    Gets the transformation function to be used by the LLA algorithm.

    Parameters
    ----------
    penalty: PenaltyConfig
        The penalty config.

    Output
    ------
    transformer: callable(coef) -> dict
        A transformation function that outputs a dict. The entries of the dict are the transformation for each non-convex subpenalty.
    """
    transforms, _, flavor_keys = get_flavored_transforms(penalty,
                                                         kind='non_convex')
    return LLATransform(transforms, flavor_keys)


class LLATransform:
    @autoassign
    def __init__(self, transforms, flavor_keys): pass

    def __call__(self, coef):
        """
        Parameters
        ----------
        coef: array-like
            The coefficient to be transformed.

        Output
        ------
        transforms: dict of array-like
            All the tranforms for the givnen penalty. The keys corresond to the flavor configs.

        """
        return {key: t(coef) for key, t in
                zip(self.flavor_keys, self.transforms)}


def get_lla_nonconvex_func(penalty):
    """
    Gets the non-convex function(s) linearized by the LLA algorithm.

    Parameters
    ----------
    penalty: PenaltyConfig
        The penalty config.

    Output
    ------
    non_convex_fucns: ya_glm.opt.base.Fucn
        The non-convex function. Formatted to accept the dicts output by LLATransform() and to return a dict whose keys are the weight keys.
    """
    ncvx_funcs, flavor_keys = _get_lla_nonconvex_func_data(penalty)
    return LAANonConvexFunc(funcs=ncvx_funcs, flavor_keys=flavor_keys)


class LAANonConvexFunc(Func):

    def __init__(self, funcs, flavor_keys):
        self.funcs = funcs
        self.flavor_keys = flavor_keys

        # build the map of flavor keys to weight keys
        # lasso__flavor -> lasso__weights
        # sparse_flavor -> sparse_weights
        self.flavor_to_weight = {}
        for flavor_key in flavor_keys:

            # pull 'flavor' off the end
            stub = flavor_key[:-6]
            weight_key = stub + 'weights'

            self.flavor_to_weight[flavor_key] = weight_key

    def grad(self, x):
        """
        Assumes x is the transformation output by a call to LLATransform()

        Parameters
        ----------
        x: dict of array-like
            The transformation dict output by LLATransform() whose keys are the flavor_keys.

        Output
        ------
        grad: dict of array-like
            The gradient of the non-convex function applied to the transforms.
            The keys are the weight keys
        """
        grad = {}
        for idx, flavor_key in enumerate(self.flavor_keys):
            # compute gradient of transformed value
            _x = x[flavor_key]
            g = self.funcs[idx].grad(_x)

            # store via the weight key
            weight_key = self.flavor_to_weight[flavor_key]
            grad[weight_key] = g

        return grad


def _get_lla_nonconvex_func_data(penalty):
    """
    Gets the non-convex functions that are linearized in the LLA algorithm.

    Parameters
    ----------
    penalty: PenaltyConfig
        The penalty.

    Output
    ------
    funcs, flavor_keys

    funcs: list of callable
        The non-convex functions for each term in the penalty.

    flavor_keys: list of str
        The keys for the flavor config that specifies each non-convex function.
    """

    tree = build_penalty_tree(penalty)
    penalties, penalty_keys = extract_penalties(tree)

    # get the non-convex flavors/penalties
    flavors, flavor_keys, parent_pens = \
        extract_flavors_and_pens(penalties=penalties,
                                 penalty_keys=penalty_keys,
                                 restrict='non_convex')

    # get the non-convex function for each term
    funcs = []
    for flav_key, parent_pen in zip(flavor_keys, parent_pens):

        if isinstance(parent_pen, ElasticNetConfig):

            # pull out the config for this summand
            sum_name = get_enet_sum_name(flav_key)
            config = parent_pen.get_sum_configs(sum_name)

        else:
            config = parent_pen

        # get the corresponding non-convex function
        funcs.append(get_outer_nonconvex_func(config))

    return funcs, flavor_keys


def get_lla_subproblem_penalty(penalty):
    """
    Returns the penalty config object that will be used by the LLA subproblems. This unflavors the penalties and sets the multiplicative pen_vals to 1 for all non-convex penalties.

    Parameters
    ----------
    config: PenaltyConfig
        The non-convex flavored penalty config that will be modified in place to create the LLA subproblem penalty config.

    Output
    ------
    penalty: PenaltyConfig
        The penalty config to be used by the LLA subproblem solvers.
    """

    # This base case is very easy

    # Unflavor and set pen_val to 1
    if isinstance(penalty, WithFlavorPenSeqConfig):
        penalty.set_params(flavor=None, pen_val=1)
        return penalty
        # TODO: the last line of the code below breaks if we dont take
        # care of this simple case here

    tree = build_penalty_tree(penalty)
    penalties, penalty_keys = extract_penalties(tree)

    # get the non-convex flavors/penalties
    flavors, flavor_keys, parent_pens = \
        extract_flavors_and_pens(penalties=penalties,
                                 penalty_keys=penalty_keys,
                                 restrict='non_convex')

    # create the subproblem penalties for each non-convex penalty
    sp_penalties = []
    for pen in parent_pens:

        # Unflavor and set pen_val to 1
        if isinstance(pen, WithFlavorPenSeqConfig):
            pen.set_params(flavor=None, pen_val=1)

        # handle ElasticNet-like case
        elif isinstance(pen, ElasticNetConfig):
            pen = _get_lla_sub_prob_for_enet(pen)

        else:
            raise RuntimeError("could not unflavor {}".format(pen))

        sp_penalties.append(pen)

    # create keys for updating the penalties
    # TODO: this may be uncessary since the penalties are modified in place
    # For now we do this in case something crazy happend
    parent_pen_keys = [get_parent_key(k) for k in flavor_keys]
    penalty.set_params(**{k: sp_penalties[idx]
                          for idx, k in enumerate(parent_pen_keys)})

    return penalty


def _get_lla_sub_prob_for_enet(penalty):
    """
    Returns LLA subproblem config object for an ElasticNet like penalty.

    For non-convex subpenalties we want the multiplicative penalty value to be one. For non-non-convex subpenalties we want the multiplicated penalty value to be the original penalty value.

    Parameters
    ----------
    penalty: PenaltyConfig
        The elastic net penalty config. Note this is modified in place

    Output
    ------
    penalty: PenaltyConfig
        The updated elastic net penalty config.

    """
    sum_pens = penalty.get_sum_configs()
    sum_names = penalty.get_sum_names()

    # handle the first summand
    if sum_pens[0].flavor == 'non_convex':
        pen_val_1 = 1

        # unflavor
        flavor_key = sum_names[0] + '_flavor'
        penalty.set_params(**{flavor_key: None})
    else:
        # keep the original penalty value
        pen_val_1 = sum_pens[0].pen_val

    # handle the second summand
    if sum_pens[1].flavor == 'non_convex':
        pen_val_2 = 1

        # unflavor
        flavor_key = sum_names[1] + '_flavor'
        penalty.set_params(**{flavor_key: None})
    else:
        pen_val_2 = sum_pens[1].pen_val

    pen_val, mix_val = _enet_params_from_sum(pen_val_1, pen_val_2)
    penalty.set_params(pen_val=pen_val, mix_val=mix_val)

    return penalty


def _enet_params_from_sum(pen_val_1, pen_val_2):
    """
    Computes the elastic net pen_val and mix_val from the two penalty values.

    pen_val_1 = pen_val * mix_val
    pen_val_2 = pen_val * (1 - mix_val )

    Parameters
    ----------
    pen_val_1, pen_val_2: float
        The two penalty values in the sum.

    Output
    ------
    pen_val, mix_val: float
        The elastic net parameters.
    """
    pen_val = pen_val_1 + pen_val_2
    mix_val = pen_val_1 / (pen_val_1 + pen_val_2)
    return pen_val, mix_val
