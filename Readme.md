# Yet another generalized linear model package



`ya_glm` aims to give you an extensive, easy to use, flexible and fast package for fitting penalized *generalized linear models* (GLMs) in Python. Existing packages (e.g. [sklearn](https://scikit-learn.org/stable/), [lightning](https://github.com/scikit-learn-contrib/lightning), [statsmodels](https://www.statsmodels.org/), [pyglmnet](https://github.com/glm-tools/pyglmnet), [celer](https://github.com/mathurinm/celer), [andersoncd](https://github.com/mathurinm/andersoncd), [picasso](https://github.com/jasonge27/picasso), [grpreg](https://github.com/pbreheny/grpreg), [ncreg](https://cran.r-project.org/web/packages/ncvreg/index.html), [glmnet](https://glmnet.stanford.edu/articles/glmnet.html)) focus on speed and ease of use, but support a limited number of loss + penalty combinations and are not easy to customize.


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

The adaptive Lasso and LLA algorithm come with strong statistical guarantees while their computational costs are not significantly worse than their convex cousins.

The built in cross-validation functionality supports

- faster path algorithms for convex penalties and adaptive lasso (e.g. as in [sklearn.linear_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html))
- automatically generated tuning parameter path for any loss + penalty combination
- custom evaluation metrics
- custom selection rules such as the '1se' rule from the glmnet package

We provide a built in FISTA algorithm ([Beck and Teboulle, 2009](https://epubs.siam.org/doi/pdf/10.1137/080716542?casa_token=cjyK5OxcbSoAAAAA:lQOp0YAVKIOv2-vgGUd_YrnZC9VhbgWvZgj4UPbgfw8I7NV44K82vbIu0oz2-xAACBz9k0Lclw)) that covers most glm loss + non-smooth penalty combinations (`ya_glm.opt` is inspired by [pyunlocbox](https://github.com/epfl-lts2/pyunlocbox) and [lightning](https://github.com/scikit-learn-contrib/lightning)). **It is straightforward for you to plug in your favorite state of the art penalized GLM optimization algorithm.**

We aim to add additional loss functions (e.g. gamma, cox regression) and penalties (e.g. generalized Lasso, TV1).


 **Beware**: this is a preliminary release of the package; the documentation and testing may leave you wanting and the code may be subject to breaking changes in the near future.



# Installation
`ya_glm` can be installed via github
```
git clone https://github.com/idc9/ya_glm.git
python setup.py install
```

To use the backend from [andersoncd](https://github.com/mathurinm/andersoncd) you have to install their package manually -- see the github page.


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

Specifying the GLM loss

```python
# specify the desired GLM loss function
# 'lin_reg' is the default
Lasso(loss='lin_reg', # 'huber', 'quantile'
      ).fit(X, y)


# Some loss functions have additional parameters that can be specified
# with config objects
from ya_glm.loss.LossConfig import Quantile
Lasso(loss=Quantile(quantile=0.75),
      ).fit(X, y)
```

Concave penalties

```python
# from ya_glm.models.AdptLasso import AdptLasso, AdptLassoCV
# from ya_glm.models.AdptENet import AdptENet, AdptENetCV
from ya_glm.models.FcpLLA import FcpLLA, FcpLLACV

# concave penalties require an initial estimator
FcpLLA(pen_func='scad',
       init='default',  # default init = LassoCV
       ).fit(X, y)

# you can provide your favorite initializer object
FcpLLA(init=LassoCV()).fit(X, y)

# or specify the initialization yourself
import numpy as np
FcpLLA(init={'coef': np.zeros(X.shape[1])}).fit(X, y)


# The CV object knows to fit the initializer object
# before running cross-validation
FcpLLACV().fit(X, y)
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