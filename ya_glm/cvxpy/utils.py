import cvxpy as cp


def solve_with_backups(problem, variable, verbosity=0, **kws):

    problem.solve(**kws)

    # try back up solvers if the solver did not work
    if variable.value is None:

        if verbosity >= 1:
            print("Solver {} failed".
                  format(problem.solver_stats.solver_name))

        # list available solvers we have not yet tried
        avail_solvers = cp.installed_solvers()
        avail_solvers.remove(problem.solver_stats.solver_name)

        success = False

        for solver in avail_solvers:
            kws['solver'] = solver

            # try each solver
            try:
                problem.solve(**kws)
                success = True

            except cp.SolverError as e:
                if verbosity >= 1:
                    print(e)

            if variable.value is not None:
                success = False

            # if we have succeded then we are done!
            if success:
                break

            else:
                if verbosity >= 1:
                    print("Solver {} failed".
                          format(problem.solver_stats.solver_name))
