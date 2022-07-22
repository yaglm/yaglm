from yaglm.config.base_params import ParamConfig
from yaglm.autoassign import autoassign


class ConstraintConfig(ParamConfig):
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


class DevecPSD(ConstraintConfig):
    """
    Assumes the coefficient is a vectorized (d x d) symmetric matrix. This constrains the devectorized version of the cofficient to be positive semi definite.

    Parameters
    ----------
    d: int
        The shape of the symmetric matrix.
    """
    @autoassign
    def __init__(self, d): pass


def get_constraint_config(config):
    """
    Gets the constraint config object.

    Parameters
    ----------
    config: str, ConstraintConfig, TunerConfig or None
        The constraint.

    Output
    ------
    config: ConstraintConfig, None
        The constraint config object.
    """
    if type(config) != str:
        return config
    else:
        return constraint_str2obj[config.lower()]


constraint_str2obj = {'pos': Positive(),
                      'isotonic': Isotonic(),
                      'simplex': Simplex(),

                      # TODO: hack since these are required!!
                      # how to handle this case??
                      'lin_eq': LinearEquality(mat=None, eq=None),

                      'devec_psd': DevecPSD(d=None),

                      'rank': Rank()
                      }

avail_constraints = list(constraint_str2obj.keys())
