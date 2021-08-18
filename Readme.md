# Yet another generalized linear model package



`ya_glm` is an extensive, easy to use, flexible and fast package for fitting penalized *generalized linear models* (GLMs) and other supervised [M-estimators](https://en.wikipedia.org/wiki/M-estimator) in Python. It supports a wide variety of loss plus penalty combinations including standard penalties such as the Lasso, Ridge and ElasticNet as well as **structured sparsity** inducing penalties such as the group Lasso and nuclear norm. It also support potentially more accurate **adaptive **and **non-convex** (e.g. SCAD) versions of these penalties that come with strong statistical guarantees at limited additional computational expense. `ya_glm` was inspired by many existing GLM packages including [sklearn](https://scikit-learn.org/stable/), [lightning](https://github.com/scikit-learn-contrib/lightning), [statsmodels](https://www.statsmodels.org/), [pyglmnet](https://github.com/glm-tools/pyglmnet), [celer](https://github.com/mathurinm/celer), [andersoncd](https://github.com/mathurinm/andersoncd), [picasso](https://github.com/jasonge27/picasso), [tick](https://github.com/X-DataInitiative/tick), [regerg](https://github.com/regreg/regreg), [grpreg](https://github.com/pbreheny/grpreg), [ncreg](https://cran.r-project.org/web/packages/ncvreg/index.html), [glmnet](https://glmnet.stanford.edu/articles/glmnet.html).


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
- [Generalized Lasso](https://projecteuclid.org/journals/annals-of-statistics/volume-39/issue-3/The-solution-path-of-the-generalized-lasso/10.1214/11-AOS878.full) which includes other penalties such as the fused lasso
- Nuclear norm
- Ridge
- Weighed versions of the above
- [Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization#Tikhonov_regularization)
- constraints (e.g. enforce a positive coefficient)

and the following more sophisticated penalties

- [Elastic net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) versions of the above
- [Adaptive](http://users.stat.umn.edu/~zouxx019/Papers/adalasso.pdf) versions of the above (including elastic-net, multi-task, group and nuclear norm)
- Folded concave penalties (FCP) such as [SCAD](https://fan.princeton.edu/papers/01/penlike.pdf) fit by applying the *local linear approximation* (LLA) algorithm to a "good enough" initializer such as the Lasso fit ([Zou and Li, 2008](http://www.personal.psu.edu/ril4/research/AOS0316.pdf); [Fan et al, 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4295817/)). We also provide non-convex versions of the group Lasso, generalized Lasso and nuclear norm that are not discussed in the original paper.
- Non-convex penalties fit directly e.g. via proximal gradient descent

The adaptive Lasso and LLA algorithm come with strong statistical guarantees while their computational costs are not significantly worse than their convex cousins (see references below).

The built in cross-validation functionality supports

- automatically generated tuning parameter path for any loss + penalty combination
- faster path algorithms for convex penalties and adaptive lasso (e.g. as in [sklearn.linear_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html))
- custom evaluation metrics
- custom selection rules such as the '1se' rule from the glmnet package

We provide a built in FISTA algorithm ([Beck and Teboulle, 2009](https://epubs.siam.org/doi/pdf/10.1137/080716542?casa_token=cjyK5OxcbSoAAAAA:lQOp0YAVKIOv2-vgGUd_YrnZC9VhbgWvZgj4UPbgfw8I7NV44K82vbIu0oz2-xAACBz9k0Lclw)) that covers many loss + non-smooth penalty combinations (`ya_glm.opt` is inspired by [pyunlocbox](https://github.com/epfl-lts2/pyunlocbox) and [lightning](https://github.com/scikit-learn-contrib/lightning)). We also provide an augmented ADMM algorithm for generalized Lasso problems ([Zhu, 2017](https://www.tandfonline.com/doi/full/10.1080/10618600.2015.1114491)). **It is straightforward for you to plug in your favorite state of the art optimization algorithm using the `solver` argument (see below).**

We aim to add additional loss functions (e.g. hinge, gamma, cox regression).

 **Beware**: this is a preliminary release of the package; the code may be subject to breaking changes in the near future.


# Installation
`ya_glm` can be installed via github
```
git clone https://github.com/idc9/ya_glm.git
python setup.py install
```


# Example

Basic convex penalties

```python
# from ya_glm.models.Vanilla import Vanilla  # unpenalized GLMs
from ya_glm.models.Lasso import Lasso, LassoCV
# from ya_glm.models.Ridge import Ridge, RidgeCV
# from ya_glm.models.ENet import ENet, ENetCV

# sample multinomial regression model with a row sparse coefficient matrix
from ya_glm.toy_data import sample_sparse_multinomial
X, y = sample_sparse_multinomial(n_samples=100, n_features=10, n_classes=3)[0:2]


# fit using the sklearn API you know and love!
Lasso(loss='multinomial',  # specify loss function
      multi_task=True  # L1 to L2 norm i.e. group Lasso on the rows
      ).fit(X, y)
# Lasso().fit(X, y)  # entrywise Lasso
# Lasso(nuc=True).fit(X, y)  # nuclear norm


# tune the lasso penalty parameter with cross-validation
# we use path algorithms to quickly compute the tuning path for each CV fold
# we automatically generate the tuning sequence
# for any loss + penalty combination (including concave ones!)
LassoCV(cv_select_rule='1se',  # here we select the penalty parameter with the 1se rule
        cv_n_jobs=-1  # parallelization over CV folds with joblib
        ).fit(X, y)


# sample linear regression model with sparse coefficient
from ya_glm.toy_data import sample_sparse_lin_reg
X, y = sample_sparse_lin_reg(n_samples=100, n_features=10)[0:2]


# you can use group lasso with user specified groups
groups = [range(5), range(5, 10)]
Lasso(groups=groups).fit(X, y)  # group elastic net
```

Specifying the loss

```python
# specify the desired loss function
# 'lin_reg' is the default
Lasso(loss='lin_reg', # 'huber', 'quantile'
      ).fit(X, y)

# Some loss functions have additional parameters that can be specified
# with config objects
from ya_glm.loss.LossConfig import Quantile
Lasso(loss=Quantile(quantile=0.75)).fit(X, y)
```

Adaptive penalties

```python
from ya_glm.models.AdptLasso import AdptLasso, AdptLassoCV
# from ya_glm.models.AdptENet import AdptENet, AdptENetCV

# Adaptive penalties require an initial estimator
AdptLasso(init='default',  # default init = LassoCV
           ).fit(X, y)

# you can provide your favorite initializer object
AdptLasso(init=LassoCV()).fit(X, y)

# or specify the initialization yourself
import numpy as np
AdptLasso(init={'coef': np.zeros(X.shape[1])}).fit(X, y)

# The CV object knows to fit the initializer object
# before running cross-validation
AdptLassoCV().fit(X, y)
```

Non-convex penalties fit with the LLA algorithm

```python
from ya_glm.models.FcpLLA import FcpLLA, FcpLLACV

# Just like adaptive penalties, the LLA algorithm requires an initial estimator
FcpLLA(pen_func='scad', init='default').fit(X, y)

FcpLLACV().fit(X, y)
```

Non-convex penalties fit directly

```python
from ya_glm.models.NonConvex import NonConvex, NonConvexCV
from ya_glm.solver.FistaSolver import FistaSolver

NonConvex(init='zero', # initialize from 0 by default
          ).fit(X, y)

NonConvexCV().fit(X, y) # will use a path algorithm by default
```

Custom solvers

```python
# you can customize the solver using a solver config class
from ya_glm.solver.FistaSolver import FistaSolver
Lasso(solver=FistaSolver(rtol=1e-4))
# you can also provide your favorite solver
# by wrapping in a solver config class
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

Loh, P.L. and Wainwright, M.J., 2017. [Support recovery without incoherence: A case for nonconvex regularization](https://projecteuclid.org/journals/annals-of-statistics/volume-45/issue-6/Support-recovery-without-incoherence-A-case-for-nonconvex-regularization/10.1214/16-AOS1530.pdf). The Annals of Statistics, 45(6), pp.2455-2482.


Zhu, Y., 2017. [An augmented ADMM algorithm with application to the generalized lasso problem](https://www.tandfonline.com/doi/full/10.1080/10618600.2015.1114491). Journal of Computational and Graphical Statistics, 26(1), pp.195-204.