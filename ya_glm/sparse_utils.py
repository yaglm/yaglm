from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse import diags, issparse
from scipy.sparse.linalg import norm as norm_sparse
from numpy.linalg import norm

import numpy as np


def safe_norm(X, ord=None, axis=0):
    if is_sparse_or_lin_op(X):
        # TODO: check how this works for  linear operator
        return norm_sparse(X, ord=ord, axis=axis)

    else:
        return norm(X, ord=ord, axis=axis)


def is_sparse_or_lin_op(a):
    return issparse(a) or isinstance(a, LinearOperator)


def safe_hstack(tup):

    if any(is_sparse_or_lin_op(t) for t in tup):
        return HStacked(tup)
    else:
        return np.hstack(tup)


class HStacked(LinearOperator):
    """
    Represents np.hstack
    """
    def __init__(self, tup):

        n_rows = tup[0].shape[0]
        self.tup_n_cols = []
        self.tup = []
        for t in tup:
            assert t.shape[0] == n_rows
            if t.ndim == 0:
                self.tup.append(t.reshape(-1, 1))
            else:
                self.tup.append(t)

            self.tup_n_cols.append(t.shape[1])

        shape = (n_rows, sum(self.tup_n_cols))

        dtype = tup[0].dtype
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x):
        out = []
        left_idx = 0
        right_idx = 0
        for idx, n_cols in enumerate(self.tup_n_cols):
            right_idx += n_cols
            out.append(self.tup[idx]  @ x[left_idx:right_idx])
            left_idx += n_cols

        return sum(o for o in out)

    def _rmatvec(self, x):
        return np.concatenate([mat.T @ x for mat in self.tup])


class OnesOuterVec(LinearOperator):
    """
    Represents the outer product 1_n vec.T where 1_n is the vector of ones
    """
    def __init__(self, n_rows, vec):
        self.vec = np.asarray(vec).reshape(-1)
        shape = (n_rows, self.vec.shape[0])
        dtype = self.vec.dtype
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x):
        return np.repeat(self.vec.T.dot(x), self.shape[0])

    def _rmatvec(self, x):
        return self.vec * x.sum()


def centered_operator(X, center):
    return aslinearoperator(X) - OnesOuterVec(X.shape[0], center)


def center_scale_sparse(X, X_offset=None, X_scale=None):
    """
    Returns a linear operator representing a centered and scaled matrix

    X_cent_scale = (X - X_offset) @ diags(1 / X_scale)

    Output
    ------
    X_cent_scale: LinearOperator
    """
    if X_offset is None and X_scale is None:
        return X

    if X_offset is not None and X_scale is not None:
        X_offset_scale = X_offset / X_scale
        X_offset_scale = np.array(X_offset_scale).reshape(-1, 1)

    elif X_offset is not None:
        X_offset_scale = X_offset

    else:
        X_offset_scale = None

    if X_scale is not None:
        X_ = X @ diags(1 / X_scale)
    else:
        X_ = X

    if X_offset_scale is not None:
        return centered_operator(X=X_, center=X_offset_scale)
    else:
        return X_


def safe_row_scaled(mat, s):
    if is_sparse_or_lin_op(mat):
        return RowScaled(mat=mat, s=s)
    else:
        return diags(s) @ mat


def safe_col_scaled(mat, s):
    if is_sparse_or_lin_op(mat):
        return ColScaled(mat=mat, s=s)
    else:
        return mat @ diags(s)


class RowScaled(LinearOperator):
    def __init__(self, mat, s):
        self.s = np.array(s).reshape(-1).astype(mat.dtype)
        assert len(self.s) == mat.shape[0]
        self.s = diags(self.s)
        self.mat = mat
        super().__init__(dtype=mat.dtype, shape=mat.shape)

    def _matvec(self, x):
        return self.s @ (self.mat @ x)

    def _rmatvec(self, x):
        return self.mat.T @ (self.s @ x)


class ColScaled(LinearOperator):
    def __init__(self, mat, s):
        self.s = np.array(s).reshape(-1).astype(mat.dtype)
        assert len(self.s) == mat.shape[1]
        self.s = diags(self.s)
        self.mat = mat
        super().__init__(dtype=mat.dtype, shape=mat.shape)

    def _matvec(self, x):
        return self.mat @ (self.s @ x)

    def _rmatvec(self, x):
        return self.s @ (self.mat.T @ x)
