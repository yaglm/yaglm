from textwrap import dedent
from ya_glm.init_signature import _multi_resp_pen_params


def merge_param_docs(*descrs):
    return ''.join(descrs)


def has_and_not_none(x, attr):
    if hasattr(x, attr) and getattr(x, attr) is not None:
        return True
    else:
        return False


def make_est_docs(C, PEN, LOSS):

    #######################
    # documentation blurb #
    #######################
    if LOSS.is_multi_resp and has_and_not_none(PEN, '_pen_descr_mr'):
        pen_descr = PEN._pen_descr_mr
    else:
        pen_descr = PEN._pen_descr

    #########################
    # parameter description #
    #########################
    if has_and_not_none(LOSS, '_params_descr'):
        param_descr = merge_param_docs(PEN._params_descr, LOSS._params_descr)
    else:
        param_descr = PEN._params_descr

    if LOSS.is_multi_resp:
        param_descr = merge_param_docs(param_descr, _multi_resp_pen_params)

    ##############
    # attributes #
    ##############

    # TODO: get more attr descr from LOSS e.g. classes_
    if LOSS.is_multi_resp and has_and_not_none(PEN, '_attr_descr_mr'):
        attr_descr = PEN._attr_descr_mr
    else:
        attr_descr = PEN._attr_descr

    # TODO: add referecnes
    descr = merge_param_docs(LOSS._loss_descr, pen_descr)

    set_docs(C=C,
             descr=descr,
             param_descr=param_descr,
             attr_descr=attr_descr,
             refs=None)


def make_cv_docs(C, PEN_CV, PEN, LOSS):

    #######################
    # documentation blurb #
    #######################
    descr = PEN_CV._cv_descr

    descr += LOSS._loss_descr

    if LOSS.is_multi_resp and has_and_not_none(PEN, '_pen_descr_mr'):
        descr += PEN._pen_descr_mr
    else:
        descr += PEN._pen_descr

    set_docs(C=C, descr=descr,
             param_descr=PEN_CV._param_descr,
             attr_descr=PEN_CV._attr_descr,
             refs=None)


def set_docs(C, descr, param_descr, attr_descr,
             refs=None):
    """
    Sets the documentation for a penalized GLM class.

    Parameters
    ----------
    C:
        Sets C.__doc__

    descr: str
        Description of the object

    param_descr: str
        Description of the init parameters.

    attr_descr: str
        Description of the class attributes.
    """
    docs = dedent("""
    {}
    Parameters
    ----------
    {}
    Attributes
    ----------
    {}
    """.format(descr, param_descr, attr_descr))

    if refs is not None:
        docs += dedent("""
        References
        ----------
        {}
        """.format(refs))

    C.__doc__ = docs
