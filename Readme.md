# Yet another generalized linear model package

`ya_glm` aims to give you a fast, easy to use and flexible package for fitting a wide variety of penalized *generalized linear models* (GLM). Existing packages (e.g. [sklearn](https://scikit-learn.org/stable/), [lightning](https://github.com/scikit-learn-contrib/lightning), [statsmodels](https://www.statsmodels.org/), [glmnet](https://glmnet.stanford.edu/articles/glmnet.html), [pyglmnet](https://github.com/glm-tools/pyglmnet), [celer](https://github.com/mathurinm/celer), [andersoncd](https://github.com/mathurinm/andersoncd)) focus on the first two of these goals, but are not easy to customize and support a limited number GLM + penalty combinations.

We currently support the following loss functions

- Linear regression
- Logistic regression
- Multinomial regression
- Poisson regression
- Huber regression
- Quantile regression
- Multiple response versions of the linear, huber and poisson losses

the following basic penalties

- Lasso
- [Group Lasso](https://rss.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1467-9868.2005.00532.x?casa_token=wN_F5iYwNK4AAAAA:4PVnAz4icP5hR9FIRviV0zqnp_QAibv55uYkptKQKezvDoqtMzrSpFyHh15lL4IO1yFJ3Sfl4OwOuA) with user specified groups
- [Multi-task Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso) (i.e. L1 to L2 norm)
- Nuclear norm
- Ridge
- Weighed versions of the above
- [Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization#Tikhonov_regularization)

and the following more sophisticated penalties

- [Elastic net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) versions of the above
- [Adaptive Lasso](http://users.stat.umn.edu/~zouxx019/Papers/adalasso.pdf) versions of the above (including multi-task, group and nuclear norm)
- Folded concave penalties (FCP) such as [SCAD](https://fan.princeton.edu/papers/01/penlike.pdf) fit by applying the *local linear approximation* (LLA) algorithm to a "good enough" initializer such as the Lasso fit ([Zou and Li, 2008](http://www.personal.psu.edu/ril4/research/AOS0316.pdf); [Fan et al, 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4295817/)). We also provide concave versions of the group Lasso, multi-task Lasso and nuclear norm that are not discussed in the original paper.


The built in cross-validation functionality supports

- faster path algorithms for convex penalties and adaptive lasso (e.g. as in [sklearn.linear_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html))
- automatically generated tuning parameter path for any loss + penalty combination
- custom evaluation metrics
- custom selection rules such as the '1se' rule from the glmnet package

We provide a built in FISTA algorithm ([Beck and Teboulle, 2009](https://epubs.siam.org/doi/pdf/10.1137/080716542?casa_token=cjyK5OxcbSoAAAAA:lQOp0YAVKIOv2-vgGUd_YrnZC9VhbgWvZgj4UPbgfw8I7NV44K82vbIu0oz2-xAACBz9k0Lclw)) that covers most glm loss + non-smooth penalty combinations (`ya_glm.opt` is inspired by [pyunlocbox](https://github.com/epfl-lts2/pyunlocbox) and [lightning](https://github.com/scikit-learn-contrib/lightning)). **It is straightforward for you to plug in your favorite penalized GLM optimization algorithm.**

We aim to add additional loss functions (e.g. gamma, cox regression) and penalties (e.g. generalized Lasso, TV1)


 **Beware**: this is a preliminary release of the package; the documentation and testing may leave you wanting and the code may be subject to breaking changes in the near future.



# Installation
`ya_glm` can be installed via github
```
git clone https://github.com/idc9/ya_glm.git
python setup.py install
```

To use the backend from [andersoncd](https://github.com/mathurinm/andersoncd) you have to install their package manually -- see the github page.


# Example


```python
from ya_glm.estimator_getter import get_pen_glm
from ya_glm.toy_data import sample_sparse_multinomial, sample_sparse_lin_reg

# multinomial regression model with a row sparse coefficient matrix
X, y = sample_sparse_multinomial(n_samples=100, n_features=10, n_classes=3)[0:2]

# programatically generate any loss + penalty combination
Est, EstCV = get_pen_glm(loss_func='multinomial', # 'lin_reg', 'poisson', ...
                         penalty='lasso' # 'enet', 'adpt_lasso', 'adpt_enet', 'fcp_lla'
                        )


# fit using the sklearn API you know and love!
Est(multi_task=True).fit(X, y)
# Est().fit(X, y)  # entrywise Lasso
# Est(nuc=True).fit(X, y)  # nuclear norm

# tune the lasso penalty with cross-validation
# we automatically generate the tuning sequence
# for any loss + penalty combination (including concave ones!)
EstCV(cv_select_rule='1se', # here we select the penalty parameter with the 1se rule
      cv_n_jobs=-1 # parallelization over CV folds with joblib
     ).fit(X, y)


# Lets try a concave penalty such as the adaptive Lasso
# or a concave penalty fit with the LLA algorithm
Est_concave, EstCV_concave =  get_pen_glm(loss_func='multinomial', 
                                          penalty='adpt_lasso' # 'fcp_lla'
                                          )

# concave penalties require an initializer which is set via the 'init' argument
# by default we initialize with a LassoCV
est = Est_concave(init='default', multi_task=True).fit(X, y)

# but you can provide any init estimator you want
init = EstCV(estimator=Est(multi_task=True), cv=10)
est = Est_concave(init=init)
est_cv = EstCV_concave(estimator=est)


# Here we use an Elastic Net version of the Adaptive Group Lasso
# with user specified groups for a liner regression example
Est, EstCV = get_pen_glm(loss_func='lin_reg', penalty='adpt_enet')
X, y = sample_sparse_lin_reg(n_samples=100, n_features=10, n_nonzero=5)[0:2]

groups = [range(5), range(5, 10)]
est = Est(groups=groups)
EstCV(estimator=est).fit(X, y)


# Quantile regression with your favorite optimization algorithm
# you can easily provide your own optimization algorithm to be the backend solver
from ya_glm.backends.quantile_lp.glm_solver import solve_glm # Linear Program formulation

Est, EstCV = get_pen_glm(loss_func='quantile',
                         penalty='adpt_lasso',
                         backend = {'solve_glm': solve_glm}
                        )

Est(quantile=0.5).fit(X, y)
```


See the [docs/](docs/) folder for additional examples in jupyter notebooks (if they don't load on github try [nbviewer.jupyter.org/](https://nbviewer.jupyter.org/)).


# Help and Support

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
idc9@cornell.edu.



## Contributing

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.




# References



Zou, H., 2006. [The adaptive lasso and its oracle properties](http://users.stat.umn.edu/~zouxx019/Papers/adalasso.pdf). Journal of the American statistical association, 101(476), pp.1418-1429.

Zou, H. and Li, R., 2008. [One-step sparse estimates in nonconcave penalized likelihood models](http://www.personal.psu.edu/ril4/research/AOS0316.pdf). Annals of statistics, 36(4), p.1509.

Beck, A. and Teboulle, M., 2009. [A fast iterative shrinkage-thresholding algorithm for linear inverse problems](https://epubs.siam.org/doi/pdf/10.1137/080716542?casa_token=cjyK5OxcbSoAAAAA:lQOp0YAVKIOv2-vgGUd_YrnZC9VhbgWvZgj4UPbgfw8I7NV44K82vbIu0oz2-xAACBz9k0Lclw). SIAM journal on imaging sciences, 2(1), pp.183-202.

Fan, J., Xue, L. and Zou, H., 2014. [Strong oracle optimality of folded concave penalized estimation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4295817/). Annals of statistics, 42(3), p.819.