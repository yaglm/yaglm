from ya_glm.solver.FistaSolver import FistaSolver


def get_default_solver(loss, penalty=None):

    if loss.name == 'quantile':
        # avoid manditory installation of cvxpy
        from ya_glm.solver.QuantileLQProgSolver import QuantileLQProgSolver

        return QuantileLQProgSolver()

    else:
        # TODO: return anderson CD when applicable
        return FistaSolver()
