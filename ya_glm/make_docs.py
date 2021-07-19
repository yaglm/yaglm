from textwrap import dedent


def merge_param_docs(*descrs):
    return ''.join(descrs)


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
