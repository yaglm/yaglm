from ya_glm.config.base import Config, safe_get_config
from ya_glm.autoassign import autoassign


class ConstraintConfig(Config):
    """
    Base constraint config object.
    """
    pass


class Positive(ConstraintConfig):
    """
    Constraints the coefficient entries to be non-negative.
    """
    pass


class Isotonic(ConstraintConfig):
    """
    Constraints the coefficient to be isotonic. For multiple response GLMs the constraint is applied to each response coefficeint vector.
    """
    pass


class Simplex(ConstraintConfig):
    """
    Constraints the coefficient to live in a simplex. For multiple response GLMs the constraint is applied to each response coefficeint vector.

    Parameters
    ----------
    radius: float
        The L1 norm radius of the simplex.
    """
    @autoassign
    def __init__(self, radius=1): pass


class LqBall(ConstraintConfig):
    """
    Constraints the coefficient to live in a Lq ball. For multiple response GLMs the constraint is applied to each response coefficeint vector.

    ||coef||_q  = radius

    Parameters
    ----------
    q: float
        Which norm.

    radius: float
        Raidus of the ball.
    """
    @ autoassign
    def __init__(self, q=2, radius=1): pass


class LinearEquality(ConstraintConfig):
    """
    Constraints the coefficient to safisfy a linear equality constraint

    mat @ coef = eq

    For multiple response GLMs the constraint is applied to each response coefficeint vector.

    Parameters
    ----------
    mat: array-like
        The matrix transform.

    eq: array-like
        The equality constraint.
    """
    @autoassign
    def __init__(self, mat, eq): pass


class Rank(ConstraintConfig):
    """
    Places a rank (upper bound) constraint on a matirx shaped coefficient.

    Parameters
    ----------
    rank: int
        The rank.
    """
    @autoassign
    def __init__(self, rank=1): pass


def get_constraint_config(config):
    """
    Gets the constraint config object. If a tuned constrainted is provided this will return the base constraint config.

    Parameters
    ----------
    config: str, ConstraintConfig, or TunerConfig
        The constraint.

    Output
    ------
    config: ConstraintConfig:
        The constraint config object.
    """
    if type(config) != str:
        return safe_get_config(config)
    else:
        return constraint_str2obj[config.lower()]


constraint_str2obj = {'pos': Positive(),
                      'isotonic': Isotonic(),
                      'simplex': Simplex(),
                      'lin_eq': LinearEquality(mat=None, eq=None),  # TODO: hack
                      'rank': Rank()
                      }

avail_constraints = list(constraint_str2obj.keys())
