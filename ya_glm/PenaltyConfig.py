import numpy as np
from scipy.linalg import svd
from inspect import signature
from copy import deepcopy

from ya_glm.opt.utils import euclid_norm
from ya_glm.opt.penalty.concave_penalty import get_penalty_func


class PenaltyConfig:
    """
    nuc
    group
    multi_task
    """
    def get_solve_kws(self):
        return deepcopy(self.__dict__)

    def get_penalty_kind(self):
        """
        Returns the penalty kind.

        Output
        ------
        pen_kind: str
            One of ['entrywise', 'group', 'multi_task', 'nuc']
        """

        if self.groups is not None:
            return 'group'

        elif self.multi_task:
            return 'multi_task'

        elif self.nuc:
            return 'nuc'

        else:
            return 'entrywise'

    def _get_coef_transform(self):
        pen_kind = self.get_penalty_kind()

        if pen_kind == 'entrywise':
            def transform(x):
                return abs(x)

        elif pen_kind == 'group':
            def transform(x):
                return np.array([euclid_norm(x[grp_idxs])
                                 for g, grp_idxs in enumerate(self.groups)])

        elif pen_kind == 'multi_task':
            def transform(x):
                return np.array([euclid_norm(x[r, :])
                                 for r in range(x.shape[0])])

        elif pen_kind == 'nuc':
            def transform(x):
                return svd(x)[1]

        return transform

    ################################
    # sub-classes should implement #
    ################################

    def validate(self):
        raise NotImplementedError

    def set_data(self, data):
        raise NotImplementedError


class ConvexPenalty(PenaltyConfig):

    def __init__(self, lasso_pen_val=None,
                 lasso_weights=None,
                 groups=None,
                 multi_task=False,
                 nuc=False,
                 ridge_pen_val=None,
                 ridge_weights=None,
                 tikhonov=None):

        if lasso_weights is not None and lasso_pen_val is None:
            lasso_pen_val = 1

        if lasso_pen_val is not None:
            self.lasso_pen_val = float(lasso_pen_val)
        else:
            self.lasso_pen_val = None

        if lasso_weights is not None:
            self.lasso_weights = np.array(lasso_weights)
        else:
            self.lasso_weights = None

        self.groups = groups
        self.multi_task = multi_task
        self.nuc = nuc

        self.tikhonov = tikhonov

        if (ridge_weights is not None or tikhonov is not None) \
                and ridge_pen_val is None:
            ridge_pen_val = 1

        if ridge_pen_val is not None:
            self.ridge_pen_val = float(ridge_pen_val)
        else:
            self.ridge_pen_val = None

        if ridge_weights is not None:
            self.ridge_weights = np.array(ridge_weights)
        else:
            self.ridge_weights = None

    def validate(self):
        if self.lasso_pen_val is not None and self.lasso_pen_val < 0:
            raise ValueError("lasso_pen_val should be non-negative")

        if self.lasso_weights is not None and self.lasso_weights.min() < 0:
            raise ValueError("lasso_weights should be non-negative")

        if self.ridge_pen_val is not None and self.ridge_pen_val < 0:
            raise ValueError("ridge_pen_val should be non-negative")

        if self.ridge_weights is not None and self.ridge_weights.min() < 0:
            raise ValueError("ridge_weights should be non-negative")

        if self.ridge_weights is not None and self.tikhonov is not None:
            raise ValueError("Both ridge_weigths and tikhonov"
                             "cannot both be provided")

        if sum([self.groups is not None,
                self.multi_task,
                self.nuc]) > 1:
            raise ValueError("At most one of groups, multi_task,"
                             " nuc can be provided")


class AdptPenalty(PenaltyConfig):

    def __init__(self, adpt_func='log',
                 adpt_func_kws={},
                 pertub_init='n_samples',
                 **cvx_pen_kws):

        self.adpt_func = adpt_func
        self.adpt_func_kws = adpt_func_kws
        self.pertub_init = pertub_init

        self.cvx_pen = ConvexPenalty(lasso_weights=np.empty(0), **cvx_pen_kws)

    @property
    def groups(self):
        return self.cvx_pen.groups

    @property
    def multi_task(self):
        return self.cvx_pen.multi_task

    @property
    def nuc(self):
        return self.cvx_pen.nuc

    def get_solve_kws(self):
        return self.cvx_pen.get_solve_kws()

    def validate(self):
        if len(self.cvx_pen.lasso_weights) == 0:
            raise RuntimeError("Adpative weights have not yet been set")

        self.cvx_pen.validate()

    # TODO: better name
    def set_data(self, data):
        if 'adpt_weights' in data:

            # if the adpative weights have been provided then use thes
            adpt_weights = data['adpt_weights']

        else:  # otherwise compute the adpative weights
            # pull out data we need to compute adpative weights
            coef_init = data['coef_init']
            if self.pertub_init == 'n_samples':
                n_samples = data['n_samples']
            else:
                n_samples = None

            adpt_weights = self.compute_adpative_weights(coef_init=coef_init,
                                                         n_samples=n_samples)

        self.set_adpt_weights(adpt_weights)

        return self

    def set_adpt_weights(self, adpt_weights):
        self.cvx_pen.lasso_weights = adpt_weights

    def compute_adpative_weights(self, coef_init, n_samples=None):
        """
        Computes the adpative weights from an initial coefficient.
        Note coef_init should be processed.
        """

        # possibly transform coef_init
        transform = self._get_coef_transform()
        coef_transf = transform(coef_init)

        # possibly perturb the transformed coefficient
        if type(self.pertub_init) == str and self.pertub_init == 'n_samples':
            coef_transf += 1 / n_samples
        elif self.pertub_init is not None:
            coef_transf += self.pertub_init

        # get adpative weights from the gradient of a concave function
        penalty_func = get_penalty_func(pen_func=self.adpt_func,
                                        pen_val=1,
                                        pen_func_kws=self.adpt_func_kws)
        adpt_weights = penalty_func.grad(coef_transf)
        return adpt_weights


class PenaltySequence:

    def __init__(self, penalty, lasso_pen_seq=None, ridge_pen_seq=None):
        self.penalty = penalty

        if lasso_pen_seq is not None:
            self.lasso_pen_seq = np.array(lasso_pen_seq)
        else:
            self.lasso_pen_seq = None

        if ridge_pen_seq is not None:
            self.ridge_pen_seq = np.array(ridge_pen_seq)
        else:
            self.ridge_pen_seq = None

    def validate(self):
        self.penalty.validate()

        # TODO: perhaps more
        if self.lasso_pen_seq is not None and self.ridge_pen_seq is not None:
            if len(self.lasso_pen_seq) != len(self.ridge_pen_seq):
                raise ValueError("lasso_pen_seq and ridge_pen_seq "
                                 "have different lengths")

    def get_solve_kws(self):
        kws = self.penalty.get_solve_kws()

        if self.lasso_pen_seq is not None:
            kws['lasso_pen_seq'] = self.lasso_pen_seq
            kws.pop('lasso_pen_val', None)

        if self.ridge_pen_seq is not None:
            kws['ridge_pen_seq'] = self.ridge_pen_seq
            kws.pop('ridge_pen_val', None)

        return kws


class ConcavePenalty(PenaltyConfig):
    def __init__(self, pen_val=0.0,
                 pen_func='scad',
                 pen_func_kws={},

                 groups=None,
                 multi_task=False,
                 nuc=False,

                 ridge_pen_val=None,
                 ridge_weights=None,
                 tikhonov=None):

        self.pen_val = pen_val
        self.pen_func = pen_func
        self.pen_func_kws = pen_func_kws

        self.groups = groups
        self.multi_task = multi_task
        self.nuc = nuc

        self.tikhonov = tikhonov

        if (ridge_weights is not None or tikhonov is not None) \
                and ridge_pen_val is None:
            ridge_pen_val = 1

        if ridge_pen_val is not None:
            self.ridge_pen_val = float(ridge_pen_val)
        else:
            self.ridge_pen_val = None

        if ridge_weights is not None:
            self.ridge_weights = np.array(ridge_weights)
        else:
            self.ridge_weights = None

    def validate(self):

        if self.pen_val is not None and self.pen_val < 0:
            raise ValueError("pen_val should be non-negative")

        if self.ridge_pen_val is not None and self.ridge_pen_val < 0:
            raise ValueError("ridge_pen_val should be non-negative")

        if self.ridge_weights is not None and self.ridge_weights.min() < 0:
            raise ValueError("ridge_weights should be non-negative")

        if self.ridge_weights is not None and self.tikhonov is not None:
            raise ValueError("Both ridge_weigths and tikhonov"
                             "cannot both be provided")

        if sum([self.groups is not None,
                self.multi_task,
                self.nuc]) > 1:
            raise ValueError("At most one of groups, multi_task,"
                             " nuc can be provided")

    def get_penalty_func(self):
        return get_penalty_func(pen_func=self.pen_func,
                                pen_val=self.pen_val,
                                pen_func_kws=self.pen_func_kws)

    def get_solve_kws(self):
        kws = deepcopy(self.__dict__)

        # rename several parameters
        # TODO: think carefully if this is the best way to hanlde this
        # e.g. ideally we don't rename anything. But the arguments to solve_glm
        # need something like nonconvex_func instead of pen_func
        kws['nonconvex_func'] = kws.pop('pen_func')
        kws['nonconvex_func_kws'] = kws.pop('pen_func_kws')
        kws['lasso_pen_val'] = kws.pop('pen_val')  # this one is especially egregious

        return kws


def get_convex_base_from_concave(penalty):

    permitted_params = [p for p in
                        signature(ConvexPenalty.__init__).parameters
                        if p != 'self']

    concave_params = penalty.__dict__
    params = {'lasso_pen_val': 1}
    for p in permitted_params:
        if p in concave_params.keys():
            params[p] = concave_params[p]

    return ConvexPenalty(**params)
