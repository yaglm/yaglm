from ya_glm.config.base import safe_get_config


def is_flavored(penalty):
    """
    Checks if a penalty is flavored.

    Parameters
    ----------
    penalty: PenaltyConfig
        The penalty config object to check.

    Output
    ------
    is_flavored: bool
        Whether or not the penalty is flavored.
    """
    pen = safe_get_config(penalty)
    if hasattr(pen, 'flavor') and pen.flavor is not None:
        return True
    else:
        return False
