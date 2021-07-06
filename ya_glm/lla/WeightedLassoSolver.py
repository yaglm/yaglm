

class WeightedLassoSolver(object):
    """
    min_y loss(y) + ||y||_{w, 1}

    or

    min_{y, u} loss(y, u) + ||y||_{w, 1}
    """

    def solve(self, L1_weights, opt_init=None, opt_init_upv=None):
        """
        Parameters
        ----------
        L1_weights: array-like
            Weights for lasso penalty.

        opt_init: None, array-like
            Optional initialization for the penalized variable.

        opt_init_upv: None, array-like
            Optional initialization for the un-penalized variable.

        Output
        ------
        solution, upv_solution, other_data
        """
        raise NotImplementedError

    def loss(self, value, upv=None):
        """
        Returns the loss function

        loss(y) or loss(y, u)

        Parameters
        ----------
        value: array-like
            Value of the variable.

        upv: None, array-like
            Optional unpenalized variable.

        Output
        ------
        loss: float
        """
        raise NotImplementedError
