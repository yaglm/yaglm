from ya_glm.autoassign import autoassign
from inspect import Signature, signature, Parameter

from makefun import wraps, add_signature_parameters


def add_from_classes(*classes, add_first=True):
    """

    Parameters
    ----------
    classes:
        The classes whose init signatures we want to inherit.

    add_first: bool
        Add the parameters in first or last.

    Output
    -------
    init_wrapper: callable(init) --> callable
        A wrapper for __init__ that adds the keyword arguments from the super classes.
        Preference is given to init's params

    """

    def init_wrapper(init):

        # start with init's current parameters
        init_params = list(signature(init).parameters.values())
        init_params = init_params[1:]  # ignore self
        current_param_names = set(p.name for p in init_params)

        empty_init_params = set(['self', 'args', 'kwargs'])

        params = []
        if add_first:
            params.extend(init_params)

        for C in classes:
            # get params for this super class
            cls_params = signature(C.__init__).parameters

            # ignore classes with empty inits
            if set(cls_params) == empty_init_params:
                continue

            cls_params = list(cls_params.values())
            cls_params = cls_params[1:]  # ignore self
            # ignore parameter if it was already in init
            cls_params = [p for p in cls_params
                          if p.name not in current_param_names]
            current_param_names.update([p.name for p in cls_params])

            params.extend(cls_params)

        if not add_first:
            params.extend(init_params)

        # make sure self is first argument
        params.insert(0,
                      Parameter('self', kind=Parameter.POSITIONAL_OR_KEYWORD))

        @wraps(init, new_sig=Signature(params))
        @autoassign
        def __init__(self, **kws): pass

        return __init__

    return init_wrapper


def keep_agreeable(params, func):
    """
    Keeps the parameters that agree with the signature of func.

    Parameters
    ----------
    params: list of str
        The parameters we want to check.

    func: callable
        The function we want the parameters to work for.
    """
    sig = signature(func)
    return [p for p in params if p in sig.parameters]


def add_to_init(params, add_first=False):
    """
    Parameters
    ----------
    params: list of tuples
        Each tuple is (PARAM_NAME, PARAM_DEFAULT)

    add_first: bool
        Add the new parameters first.
    """

    def init_wrapper(init):
        new_sig = signature(init)

        # if add_first:
        #     params = params[::-1]

        for (name, default) in params:
            new_param = Parameter(name=name,
                                  kind=Parameter.POSITIONAL_OR_KEYWORD,
                                  default=default)
            if add_first:
                new_sig = add_signature_parameters(new_sig, first=new_param)
            else:
                new_sig = add_signature_parameters(new_sig, last=new_param)

        @wraps(init, new_sig=new_sig)
        @autoassign
        def __init__(self, **kws): pass

        return __init__

    return init_wrapper


def add_multi_resp_params(add=False):

    if add:
        return add_to_init(params=[('multi_task', False),
                                   ('nuc', False)])
    else:
        def same(init):
            return init
        return same
