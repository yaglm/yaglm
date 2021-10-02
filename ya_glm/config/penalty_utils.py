from ya_glm.config.base_params import get_base_config
from ya_glm.config.flavor import FlavorConfig
from ya_glm.config.base_penalty import PenaltyConfig, PenaltyTuner


def build_penalty_tree(penalty):
    """
    Think of a penalty as a rooted tree where the child nodes are the sub-penalties. This function recursively builds the flavor config tree for a multi-penalty using a depth first search.

    Note a single penalty or elastic-net penalty is a leaf node; Additive penalties are internal nodes.

    Output
    ------
    node: dict
        Each level of the tree has the following keys.

        node['children']: dict
            The subtree rooted at this node.

        node['children_keys']: list of str
            The keys for the child node parameters.

        node['penalty']: PenaltyConfig or TunerConfig
            The penalty at this node.
    """

    node = {'children_keys': [],
            'children': []}

    # store the penalty
    node['penalty'] = penalty

    # iterate over all of the nodes parameters
    # pull out information about flavor and penalty parameters.
    for (key, value) in penalty.get_params(deep=False).items():

        # if this parameter is a TunerConfig pull out the base value
        base_value = get_base_config(value)

        # pull out sub-penalties
        if isinstance(base_value, PenaltyConfig):
            node['children_keys'].append(key)

            # recursively bulid the tree!
            node['children'].append(build_penalty_tree(value))

    return node


def extract_penalties(tree, drop_tuners=False):
    """
    Pulls out a list of the penalty configs from a penalty tree.

    Parameters
    ----------
    tree: dict
        The penatly tree output by build_penalty_tree().

    drop_tuners: bool
        Drop all referecnes to tuner objects e.g. pretends we called detune_config() on the penalty.

    Output
    ------
    penalties, keys

    penalties: list of Configs and TunerConfigs
        The penalties.

    keys: list of str
        The keys for each penalty in the root penalty's set_params() method.
    """
    return _walk_pen_tree(node=tree, drop_tuners=drop_tuners)


def _walk_pen_tree(node, penalties=None, keys=None,
                   key_stub=None, drop_tuners=False):
    """
    Recursive function that does the work for extract_penalties() via a depth first seach.
    """

    # if this is the first call initialize the data we need
    if penalties is None:
        penalties = []
        keys = []
        key_stub = ''

    # store this penalty and key
    this_penalty = node['penalty']

    # skip this config if we are dropping the tuners and this is a tuner
    skip_this = drop_tuners and isinstance(this_penalty, PenaltyTuner)

    # store data if we are not skipping this
    if not skip_this:
        penalties.append(this_penalty)
        keys.append(key_stub)

    # walk the child penalties
    for i, child in enumerate(node['children']):

        # create the key for this child
        if skip_this:
            # if we are skipping this node then we keep the key stub the same
            new_key_stub = key_stub
        else:
            if key_stub == '':
                # if this is the top node then the new key doesnt get a __
                new_key_stub = node['children_keys'][i]
            else:
                new_key_stub = key_stub + '__' + node['children_keys'][i]

        # walk this child's tree
        _walk_pen_tree(node=child,
                       penalties=penalties, keys=keys,
                       key_stub=new_key_stub,
                       drop_tuners=drop_tuners)

    return penalties, keys


def get_flavor_kind(penalty):
    """
    Gets the flavor kind of a penalty.

    Parameters
    ----------
    penalty: PenaltyConfig, PenaltyTuner
        The penalty whose flavor we want to know.

    Output
    ------
    flavor_kind: None, str
        The flavor; one of [None, 'adaptive', 'non_convex']
    """
    if penalty is None:
        return None

    tree = build_penalty_tree(penalty)
    penalties, keys = extract_penalties(tree, drop_tuners=True)
    flavors, _ = extract_flavors(penalties=penalties, keys=keys,
                                 force_base=True)

    # pull out unique flavor names
    flavor_names = set(f.name for f in flavors)

    # pull out the flavor
    if len(flavor_names) == 0:
        return None
    elif len(flavor_names) == 1:
        return list(flavor_names)[0]
    else:
        return 'mixed'


def extract_flavors(penalties, keys, force_base=False, restrict=None):
    """
    Extracts the flavors for the penalties. The input to this function should be the output of extract_penalties()

    Parameters
    ----------
    penalties: list of PenaltyConfigs/PenaltyTuners
        The penalties.

    keys: list of str
        The keys for the penalties.

    force_base: bool
        Always pull out the base flavor config if a flavor is a tuner object.

    restrict: None, str
        Restrict to only this kind of flavor. If provided, must be one of ['adaptive', 'non_convex'].

    Output
    ------
    flavors: list of FlavorConfigs/TunerConfigs
        The flavor configs.

    flavor_keys: list of str
        The keys for the flavor configs.
    """
    flavors = []
    flavor_keys = []

    # go through each penalty
    for pen, pen_key in zip(penalties, keys):

        # look at each parameter in the penalty
        for (k, v) in pen.get_params(deep=False).items():

            # if this parameter is a TunerConfig pull out the base value
            base_value = get_base_config(v)

            # pull out flavor configs
            if isinstance(base_value, FlavorConfig):

                if pen_key == '':
                    # if the penalty is the root node then
                    # no need to prepend anything
                    flav_key = k
                else:
                    # prepend penalty
                    flav_key = pen_key + '__' + k
                flavor_keys.append(flav_key)

                if force_base:
                    flavors.append(base_value)
                else:
                    flavors.append(v)

    ###########################################
    # Possibly restrict the flavors we return #
    ###########################################
    if restrict is not None and len(flavors) > 0:
        assert restrict in ['adaptive', 'non_convex']

        # restrict flavors to only the requested flavor
        flavors, flavor_keys = zip(*[(f, k)
                                     for (f, k) in zip(flavors, flavor_keys)
                                     if get_base_config(f).name == restrict])

    return flavors, flavor_keys


def extract_flavors_and_pens(penalties, penalty_keys,
                             force_base=False, restrict=None):
    """
    Like extract_flavors(), but this also returns the parent penalties for each flavor.

    Output
    ------
    flavors, flavor_keys, penalties
    """
    penalty_map = {k: penalties[idx] for idx, k in enumerate(penalty_keys)}

    # extract all the non-convex penalties
    flavors, flavor_keys = extract_flavors(penalties=penalties,
                                           keys=penalty_keys,
                                           force_base=force_base,
                                           restrict=restrict)

    # get the non-convex function for each term
    parent_penalties = []
    for flav_key in flavor_keys:

        # pull out the parent penalty for this flavor
        parent_pen_key = get_parent_key(flav_key)
        parent_pen = penalty_map[parent_pen_key]

        parent_penalties.append(parent_pen)

    return flavors, flavor_keys, parent_penalties


def get_unflavored(penalty):
    """
    Returns an unflavored version of a penalty config.

    Parameters
    ----------
    penalty: PenaltyConfig, PenaltyTuner
        The penalty config; this will be modified in place.

    Output
    ------
    penalty: PenaltyConfig, PenaltyTuner
        The unflavored penalty config.
    """

    # pull out all flavors
    tree = build_penalty_tree(penalty)
    penalties, keys = extract_penalties(tree)
    flavors, keys = extract_flavors(penalties, keys)

    # set all flavor arguments to None
    penalty.set_params(**{key: None for key in keys})
    return penalty


def get_parent_key(key):
    """
    Gets the key of the parent object. Assumes keys are formatted like

    param1__param2__param__3__...

    Parameters
    ----------
    key: str
        The key whose parent we want.

    Output
    ------
    parent_key: str
        The parent key
    """
    return '__'.join(key.split('__')[:-1])


def get_enet_sum_name(flavor_key):
    """
    Gets the elastic net summand name from its flavor key.

    get_enet_sum_name('...__lasso_flavor') -> lasso

    Parameters
    ----------
    flavor_key: str
        The flavor key for this summand's flavor.

    Output
    ------
    name: str
        The summaned name.
    """

    # pull out the sum penalty for this flavor
    # TODO: is there an easier way to do this?
    flav_name = flavor_key.split('__')[-1]  # ...__lasso_flavor -> lasso_flavor
    return flav_name.split('_')[0]  # lasso_flavor -> lasso


def get_ancestor_keys(key, candidates):
    """
    Gets all ancestors for a key among a list of candidates.

    Parameters
    ----------
    key: str
        The keys who ancestors we want.

    candidates: list of str
        The candidate ancestors.

    Output
    ------
    ancestors: list of str
        The ancestors ordered so that the oldest ancestors come first.
    """
    # drop any duplicates
    candidates = list(dict.fromkeys(candidates))

    # pull out all ancestors
    ancestors = []
    for maybe_ancestor in candidates:
        # a key startes with all its ancestors
        if key.startswith(maybe_ancestor):
            ancestors.append(maybe_ancestor)

    # order ancestors so the oldest ancestors come first
    ancestors = sorted(ancestors, key=lambda s: len(s))

    return ancestors
