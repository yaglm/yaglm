import numpy as np

from yaglm.opt.constraint.convex import Constraint
from yaglm.opt.base import Func


class PSDCone(Constraint):
    """
    The positive semi-definite cone.

    Parameters
    ----------
    force_sym: bool
        Whether or not to force the input matrix to be symmetric. NOTE if the matrix is NOT symmetric, the current version is not exactly the prox.

        TODO: There is a way of getting the exact prox (@iain find reference) -- perhaps we should include this?

    """

    def __init__(self, force_sym=True):
        self.force_sym = force_sym

    def _prox(self, x, step=1):
        """
        Parameteres
        -----------
        x: array-like, shape (d x d)
            A symmetric matrix. NOTE this prox is not correct if x is not symmetric!
        """
        if self.force_sym:
            x = 0.5 * (x + x.T)
        return project_psd(x)

    @property
    def is_proximable(self):
        return True


class Devec2SymMat(Func):
    """
    Devectorizes a vector to a symetric matrix. Wraps a function defined on symmetric matrices so it takes a vector as input.

    Assumes the input is the entire matrix (i.e. contains some redundancy).

    Parameters
    ----------
    d: int
        Number of rows of the symmetrix matrix. Note x.shape should be (d^2, ).

    func:
        The function of the symmetric matrix.
    """
    def __init__(self, d, func):
        self.d = d
        self.func = func

    def _eval(self, x):
        x_mat = x.reshape(self.d, self.d)
        return self.func(x_mat)

    def _prox(self, x, step=1):
        x_mat = x.reshape(self.d, self.d)
        p_mat = self.func.prox(x_mat, step=step)
        return p_mat.ravel()

    def _grad(self, x, step=1):
        x_mat = x.reshape(self.d, self.d)
        g_mat = self.func.grad(x_mat)
        return g_mat.ravel()

    @property
    def is_proximable(self):
        return self.func.is_proximable

    @property
    def grad_lip(self):
        return self.func.grad_lip

    @property
    def is_smooth(self):
        return self.func.is_smooth


def project_psd(M):
    """
    Projects a symmetric matrix onto the positive semidefinite cone.

    Parameters
    ----------
    M : 2D-Array or Matrix
        A symmetric matrix.

    Returns
    -------
    Mplus : 2D-Array
        The projection of M onto the positive semidefinite cone.
    """
    # borrowed from https://github.com/jlsuarezdiaz/pyDML/blob/master/dml/dml_utils.pyx
    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = eigvals.astype(float)  # Remove residual imaginary part
    eigvecs = eigvecs.astype(float)
    eigvals[eigvals < 0.0] = 0.0  # MEJORAR ESTO
    diag_sdp = np.diag(eigvals)
    return eigvecs.dot(diag_sdp).dot(eigvecs.T)
