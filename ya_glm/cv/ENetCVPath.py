import numpy as np
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier


from ya_glm.cv.CVPath import CVPathMixin, run_cv_path, \
    add_params_to_cv_results


class ENetCVPathMixin(CVPathMixin):

    def _get_solve_path_enet_base_kws(self):
        """
        Returns the key word argus for solve_path excluding the lasso_pen_seq and L2_pen_seq values.
        """
        raise NotImplementedError

    def _run_cv(self, estimator, X, y=None, cv=None, fit_params=None):
        # TODO: see if we can simplify this with self.get_tuning_sequence()

        # setup CV
        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        # setup path fitting function
        fit_and_score_path = self._fit_and_score_path_getter(estimator)

        kws = self._get_solve_path_enet_base_kws()

        if self._tune_l1_ratio() and self._tune_pen_val():
            # Tune over both l1_ratio and pen_val
            # for each l1_ratio, fit the pen_val path

            all_cv_results = []
            for l1_idx, l1_ratio in enumerate(self.l1_ratio_seq_):

                # set pen vals for this sequence
                pen_val_seq = self.pen_val_seq_[l1_idx, :]
                kws['lasso_pen_seq'] = pen_val_seq * l1_ratio
                kws['ridge_pen_seq'] = pen_val_seq * (1 - l1_ratio)

                # fit path
                cv_res, _ = \
                    run_cv_path(X=X, y=y,
                                fold_iter=cv.split(X, y),
                                fit_and_score_path=fit_and_score_path,
                                kws=kws,
                                fit_params=fit_params,
                                include_spilt_vals=False,  # maybe make this True?
                                add_params=False,
                                n_jobs=self.cv_n_jobs,
                                verbose=self.cv_verbose,
                                pre_dispatch=self.cv_pre_dispatch)

                all_cv_results.append(cv_res)

            # combine cv_results for each l1_ratio value
            cv_results = {}
            n_l1_ratios = len(self.l1_ratio_seq_)
            for name in all_cv_results[0].keys():
                cv_results[name] = \
                    np.concatenate([all_cv_results[i][name]
                                    for i in range(n_l1_ratios)])

        else:
            # only tune over one of l1_ratio or pen_val

            # setup L1/L2 penalty sequence
            if self._tune_pen_val():
                # tune over pen_val
                pen_val_seq = self.pen_val_seq_
                l1_ratio_seq = self.l1_ratio  # * np.ones_like(pen_val_seq)

            elif self._tune_l1_ratio():
                # tune over l1_ratio
                l1_ratio_seq = self.l1_ratio_seq_
                pen_val_seq = self.pen_val  # * np.ones_like(pen_val_seq)

            kws['lasso_pen_seq'] = pen_val_seq * l1_ratio_seq
            kws['ridge_pen_seq'] = pen_val_seq * (1 - l1_ratio_seq)

            # fit path
            cv_results, _ = \
                run_cv_path(X=X, y=y,
                            fold_iter=cv.split(X, y),
                            fit_and_score_path=fit_and_score_path,
                            kws=kws,
                            fit_params=fit_params,
                            include_spilt_vals=False,  # maybe make this True?
                            add_params=False,
                            n_jobs=self.cv_n_jobs,
                            verbose=self.cv_verbose,
                            pre_dispatch=self.cv_pre_dispatch)

        # add parameter sequence to CV results
        # this allows us to pass fit_path a one parameter sequence
        # while cv_results_ uses different names
        param_seq = self.get_tuning_sequence()
        cv_results = add_params_to_cv_results(param_seq=param_seq,
                                              cv_results=cv_results)

        return cv_results
