# Yet another generalized linear model package

`ya_glm` aims to give you a fast, easy to use and flexible package for fitting a wide variety of penalized *generalized linear models* (GLM). Existing packages (e.g. [sklearn](https://scikit-learn.org/stable/), [lightning](https://github.com/scikit-learn-contrib/lightning), [statsmodels](https://www.statsmodels.org/), [glmnet](https://glmnet.stanford.edu/articles/glmnet.html), [pyglmnet](https://github.com/glm-tools/pyglmnet), [celer](https://github.com/mathurinm/celer), [andersoncd](https://github.com/mathurinm/andersoncd)) accomplish the first two of these goals, but are not easy to customize and support a limited number GLM + penalty combinations.

 
 **Beware**: this is a preliminary release of the package; the documentation and testing may leave you wanting and the code may be subject to breaking changes in the near future.



# Installation
`ya_glm` can be installed via github
```
git clone https://github.com/idc9/ya_glm.git
python setup.py install
```

To use the backend from [andersoncd](https://github.com/mathurinm/andersoncd) you have to install their package manually -- see their github page.


# Example


```python
from ya_glm.backends.fista.LinearRegression import Lasso, LassoCV, RidgeCV, LassoENetCV, \
    GroupLassoENet, GroupLassoENetCV, \
    FcpLLA, FcpLLACV

from ya_glm.toy_data import sample_sparse_lin_reg

# sample some linear regression data
X, y = sample_sparse_lin_reg(n_samples=100, n_features=20)[0:2]

# fit the Lasso penalized linear regression model we all known and love
est = Lasso(pen_val=1).fit(X, y)

# tune the Lasso penalty using cross-validation
# just as in sklearn.linear_model.LassoCV we use a 
# path algorithm to make this fast and set the tuning
# parameter sequence with a sensible default
est_cv = LassoCV(cv_select_rule='1se').fit(X, y)

# or you could have picked a different penalty!
# est_cv = RidgeCV().fit(X, y)
# est_cv = LassoENetCV().fit(X, y)

# we support user specified groups!
groups = [range(10), range(10, 20)]
est = GroupLassoENet(groups=groups)
# and a cross-validation object that supports path solutions 
est_cv = GroupLassoENetCV(estimator=est).fit(X, y)


# folded concave penalty with SCAD penalty
# and initialized from the LassoCV solution
# see (Fan et al. 2014) for details
est = FcpLLA(init=LassoCV(), pen_func='scad')

# we can also tune this with cross-validation
est_cv = FcpLLACV(estimator=est).fit(X, y)
```

We support a variety of other loss functions such as **logistic regression**. We can generate these estimator objects programmatically to avoid writing a ton of code by hand.


```python
from ya_glm.estimator_getter import get_pen_glm, get_fcp_glm
from ya_glm.toy_data import sample_sparse_log_reg

# sample some logistic regression data
X, y = sample_sparse_log_reg(n_samples=100, n_features=20)[0:2]

# Get a penalized logistic regression estimator and corresponding cross-validation object
Est, EstCV = get_pen_glm(loss_func='log_reg', penalty='lasso')
# Est, EstCV = get_pen_glm(loss_func='log_reg', penalty='lasso_enet') # Elastic Net
# Est, EstCV = get_pen_glm(loss_func='log_reg', penalty='group_lasso')  # Group lasso

# Or a concave penalized logistic regression estimator
# Est, EstCV = get_fcp_glm(loss_func='log_reg', penalty='lasso')

est = Est().fit(X, y) # single fit
est_cv = EstCV().fit(X, y) # cross-validation
```


See the [docs/](docs/) folder for additional examples in jupyter notebooks (if they don't load on github try [nbviewer.jupyter.org/](https://nbviewer.jupyter.org/)).


# Currently supported features

We currently support the following loss functions

- Linear regression (single and multiple responses)
- Logistic regression
- Multinomial regression
- Huber regression (single and multiple responses)

and the following penalties

- Lasso
- Group Lasso with user specified groups
- Elastic net
- Ridge
- Tikhonov
- Nuclear norm
- Multi-task Lasso (i.e. L1 to L2 norm)
- Weighted versions of all of the above
- Folded concave penalties (FCP) such as SCAD

The FCP penalties are fit by applying the *local linear approximation* (LLA) algorithm to a "good enough" initializer such as the Lasso fit. See (Fan et al, 2014) for details. We provide concave versions of the group Lasso, multi-task Lasso and nuclear norm that are not discussed in the original paper.

We also supply built in cross-validation (CV) for each of these penalties. For the concave penalties (e.g. Lasso) our CV methods use faster path algorithms (as in sklearn.linear_model.LassoCV). Our CV function allow custom metrics and custom selection rules such as the '1se' rule from the glmnet package.

We aim to add additional loss functions including quantile regression, poisson, gamma, and cox regression, etc.


# What we provide on the backend


- Cross-validation support for

	- path algorithms with parallelization over folds
	- custom evaluation metrics
	- custom CV selection rule (e.g. the '1se' rule)
	- automatically generated tuning parameter sequence from the training data for both Lasso and concave penalties. This requires computing the largest reasonable penalty value for different combinations of loss + penalty.
	- see e.g. `ya_glm.GlmCV`, `ya_glm.fcp.GlmFcpCV`

- A  FISTA algorithm (Beck and Tebouule, 2009) for GLM loss + non-smooth penalty problems
	- see `ya_glm.opt`
	- this module  is inspired by [pyunlocbox](https://github.com/epfl-lts2/pyunlocbox) and [lightning](https://github.com/scikit-learn-contrib/lightning)


- Support for concave penalties such as SCAD
	- fit using the LLA algorithm (Zou and Li, 2008; Fan et al. 2014)
	- see `ya_glm.lla`
	- the LLA algorithm only needs you to provide a solver for GLM + weight Lasso problems

	
- Support for customization
	- straightforward to swap in your favorite GLM solver
	- see `ya_glm.backends.andersoncd` for an example
	- cross-validation tuning
	- monitor various metrics for cross-validation path


# Help and Support

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
idc9@cornell.edu.



## Contributing

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.




# References

Zou, H. and Li, R., 2008. [One-step sparse estimates in nonconcave penalized likelihood models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2759727/). Annals of statistics, 36(4), p.1509.

Beck, A. and Teboulle, M., 2009. [A fast iterative shrinkage-thresholding algorithm for linear inverse problems](https://epubs.siam.org/doi/pdf/10.1137/080716542?casa_token=cjyK5OxcbSoAAAAA:lQOp0YAVKIOv2-vgGUd_YrnZC9VhbgWvZgj4UPbgfw8I7NV44K82vbIu0oz2-xAACBz9k0Lclw). SIAM journal on imaging sciences, 2(1), pp.183-202.


Fan, J., Xue, L. and Zou, H., 2014. [Strong oracle optimality of folded concave penalized estimation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4295817/). Annals of statistics, 42(3), p.819.