from yaglm.opt.base import Zero, Sum
from yaglm.opt.BlockSeparable import BlockSeparable


# TODO: make this recursive for additive penalties
def split_smooth_and_non_smooth(func):
    """
    Splits a penalty function into smooth and non-smooth functions.

    Parameters
    ----------
    func: Func
        The input function to split

    Output
    ------
    smooth, non_smooth
    """

    if func is None:
        return None, None

    elif isinstance(func, Sum):

        # get the smooth and non-smooth components
        smooth_funcs, non_smooth_funcs = \
            zip(*(split_smooth_and_non_smooth(f) for f in func.funcs))

        # drop all nones
        smooth_funcs = [f for f in smooth_funcs if f is not None]
        non_smooth_funcs = [f for f in non_smooth_funcs if f is not None]

        # pull apart smooth and non-smooth functions
        # smooth_funcs = [f for f in func.funcs if f.is_smooth]
        # non_smooth_funcs = [f for f in func.funcs if not f.is_smooth]

        if len(smooth_funcs) == 1:
            smooth = smooth_funcs[0]
        elif len(smooth_funcs) > 1:
            smooth = Sum(smooth_funcs)
        else:
            # smooth = Zero()
            smooth = None

        if len(smooth_funcs) == 1:
            non_smooth = non_smooth_funcs[0]

        elif len(non_smooth_funcs) >= 1:
            non_smooth = Sum(non_smooth_funcs)
        else:
            non_smooth = None

        return smooth, non_smooth

    elif isinstance(func, BlockSeparable):

        # get the smooth/non-smooth components
        smooth_funcs, non_smooth_funcs = \
            zip(*(split_smooth_and_non_smooth(f) for f in func.funcs))

        n_smooth = sum(f is not None for f in smooth_funcs)
        n_non_smooth = sum(f is not None for f in non_smooth_funcs)

        if n_smooth >= 1:
            # replace the Nones with zeros
            # currently BlockSeparable() is not smart enough to
            # handle Nones
            smooth_funcs = [f if f is not None else Zero()
                            for f in smooth_funcs]

            smooth = BlockSeparable(funcs=smooth_funcs,
                                    groups=func.groups)
        else:
            smooth = None

        if n_non_smooth >= 1:
            # replace the Nones with zeros
            # currently BlockSeparable() is not smart enough to
            # handle Nones
            non_smooth_funcs = [f if f is not None else Zero()
                                for f in non_smooth_funcs]

            non_smooth = BlockSeparable(funcs=non_smooth_funcs,
                                        groups=func.groups)
        else:
            non_smooth = None

        return smooth, non_smooth

    else:
        smooth = None
        non_smooth = None

        if func.is_smooth:
            smooth = func
        else:
            non_smooth = func

        return smooth, non_smooth
