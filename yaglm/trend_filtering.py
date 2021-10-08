import numpy as np
from scipy.sparse import diags, lil_matrix


def get_tf_mat(d, k):
    """
    Gets the difference matrix for kth order trend fildering. See Section 2.1.2 of (Tibshirani and Taylor, 2011).

    Parameters
    ----------
    d: int
        Number of points.

    k: int
        Order of the difference.

    Output
    ------
    D: array-like, shape (d - k, d)
        The kth order difference matrix returned in a sparse matrix format.

    References
    ----------
    Tibshirani, R.J. and Taylor, J., 2011. The solution path of the generalized lasso. The annals of statistics, 39(3), pp.1335-1371.
    """
    D1d = get_tf1(d)
    return _get_diff_mat_recursive(d=d, k=int(k), D1d=D1d)


def get_tf1(d):
    """
    Returns the first order trend filtering matrix for d features.

    Output
    -------
    D: array-like, shape (d -1, d)
        The kth order difference matrix returned in a sparse matrix format.
    """
    D = diags(diagonals=[-np.ones(d), np.ones(d - 1)], offsets=[0, 1])
    D = D.tocsc()
    D = D[:-1, :]
    return D


def _get_diff_mat_recursive(d, k, D1d):
    """
    Recusrively computes the the kth order trend filtering matrix for d features using a cached first order trend filtering matrix.

    Parameters
    ----------
    d: int
        Number of features

    k: int
        Order

    D1d: array-like, shape (d - 1, d)
        The cahced first order trend filtering matrix.

    References
    ----------
    Tibshirani, R.J. and Taylor, J., 2011. The solution path of the generalized lasso. The annals of statistics, 39(3), pp.1335-1371.
    """
    if k == 1:
        return D1d
    else:
        # the first order trend filtering matrix on (d - k) nodes
        # just drop the last k-1 rows/columns of
        # the first order TF matrix on d nodes
        D1d_d_k = D1d[:-(k-1), :][:, :-(k-1)]

        # k-1 order diff mat
        Dkm1 = _get_diff_mat_recursive(d=d, k=k-1, D1d=D1d)
        return D1d_d_k @ Dkm1

# TODO: mabye delete this, but it could be useful for debugging
# def get_tf_mat_no_cache(d, k=1):
#     """
#     Gets the kth difference matrix. See Section 2.1.2 of (Tibshirani and Taylor, 2011).

#     Parameters
#     ----------
#     d: int
#         Number of points.

#     k: int
#         Order of the difference.

#     Output
#     ------
#     D: array-like, shape (d - k, d)
#         The kth order difference matrix returned in a sparse matrix format.

#     References
#     ----------
#     Tibshirani, R.J. and Taylor, J., 2011. The solution path of the generalized lasso. The annals of statistics, 39(3), pp.1335-1371.
#     """
#     if k == 1:
#         D = diags(diagonals=[-np.ones(d), np.ones(d - 1)], offsets=[0, 1])
#         D = D.tocsc()
#         D = D[:-1, :]
#         return D

#     else:
#         # first order diff for d - k + 1
#         D1d = get_tf_mat_no_cache(d=d-k+1, k=1)
#         Dkm1 = get_tf_mat_no_cache(d=d, k=k-1)  # k-1 order diff
#         return D1d @ Dkm1


# TODO-THINK-THROUGH: is this recursive computation the most efficient way to compute this
def get_graph_tf_mat(edgelist, n_nodes, k=1):
    """
    Creates the kth order graph trend filtering difference operator from an edgelist, see delta^(k) in Section 2.2 of (Wang et al, 2015).

    Parameters
    ----------
    array-like, shape (2, n_edges)
        The edgelist. Each row represents and contains the indices of the two nodes. The node in the first column gets a -1.

    n_nodes: int
        Total number of nodes in the graph.

    k: int
        The order of the differences.

    Output
    ------
    diff: scipy.sparse.csr_matrix, shape (n_edges, n_nodes)
        The kth order graph difference matrix matrix.

    References
    ----------
    Wang, Y.X., Sharpnack, J., Smola, A. and Tibshirani, R., 2015, February. Trend filtering on graphs. In Artificial Intelligence and Statistics (pp. 1042-1050). PMLR.
    """
    delta_1 = get_oriented_incidence_mat(edgelist, n_nodes)

    return _get_graph_tf_mat_recursive(edgelist, n_nodes, k=int(k),
                                       delta_1=delta_1)


def _get_graph_tf_mat_recursive(edgelist, n_nodes, k, delta_1):
    """
    Recursively computes the kth order graph trend filtering difference operator from an edgelist using a cached first order difference matrix.
    """

    if k == 1:
        return delta_1

    elif (k - 1) % 2 == 0:  # k-1 is even
        delta_k_1 = _get_graph_tf_mat_recursive(edgelist, n_nodes,
                                                k=k-1, delta_1=delta_1)
        return delta_1 @ delta_k_1

    else:  # k-1 is odd
        delta_k_1 = _get_graph_tf_mat_recursive(edgelist, n_nodes,
                                                k=k-1, delta_1=delta_1)

        return delta_1.T @ delta_k_1


def get_oriented_incidence_mat(edgelist, n_nodes):
    """
    Creates the oriented incidence matrix for a graph from its edgelist, see delta^(1) in Section 2.2 of (Wang et al, 2015).

    Parameters
    ----------
    array-like, shape (2, n_edges)
        The edgelist. Each row represents and contains the indices of the two nodes. The node in the first column gets a -1.

    n_nodes: int
        Total number of nodes in the graph.

    Output
    ------
    incidence_mat: scipy.sparse.csr_matrix, shape (n_edges, n_nodes)
        The oriented incidence matrix.

    References
    ----------
    Wang, Y.X., Sharpnack, J., Smola, A. and Tibshirani, R., 2015, February. Trend filtering on graphs. In Artificial Intelligence and Statistics (pp. 1042-1050). PMLR.
    """

    edgelist = np.array(edgelist)
    n_edges = edgelist.shape[0]

    incidence_mat = lil_matrix((n_edges, n_nodes))
    for idx, edge in enumerate(edgelist):
        incidence_mat[idx, edge[0]] = -1
        incidence_mat[idx, edge[1]] = 1

    return incidence_mat.tocsr()
