# Yet another generalized linear model package


`ya_glm` is flexible, comprehensive and modern python package for fitting penalized generalized linear models and other supervised [M-estimators](https://en.wikipedia.org/wiki/M-estimator) in Python. It supports a wide variety of losses (linear, logistic, quantile, etc) combined with penalties and/or constraints. Beyond the basic lasso/ridge, `ya_glm` supports  **structured sparsity** penalties such as the nuclear norm and the [group](https://rss.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1467-9868.2005.00532.x?casa_token=wN_F5iYwNK4AAAAA:4PVnAz4icP5hR9FIRviV0zqnp_QAibv55uYkptKQKezvDoqtMzrSpFyHh15lL4IO1yFJ3Sfl4OwOuA), [exclusive](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-11/issue-2/Within-group-variable-selection-through-the-Exclusive-Lasso/10.1214/17-EJS1317.full), [graph fused](https://arxiv.org/pdf/1505.06475.pdf), and [generalized lasso](https://www.stat.cmu.edu/~ryantibs/papers/genlasso.pdf). It also support the more accurate **[adaptive](http://users.stat.umn.edu/~zouxx019/Papers/adalasso.pdf)** and **non-convex** (e.g. [SCAD](https://fan.princeton.edu/papers/01/penlike.pdf)) *flavors* of these penalties that typically come with strong statistical guarantees at limited additional computational expense. 

Parameter tuning methods including cross-validation, generalized cross-validation, and information criteria (e.g. AIC, BIC, [EBIC](https://www.jstor.org/stable/20441500)) come built-in. BIC-like information criteria are important for analysts interested in model selection [(Zhang et al, 2012)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.tm08013); these are augmented with built-in linear regression noise variance estimators ([Reid et al, 2016](https://www.jstor.org/stable/pdf/24721190.pdf?casa_token=wVML37DFzk4AAAAA:PCPZH8z98S_ZDNMyFxtec9-ZsIx73xoxDgWJUEObeJooVLwMWhOAn_Tnf2GQGL3H36XAROk5P08aNGcDnJUG95ahVwe1F57AsJg0_kxntX4UIoSoEAk); [Yu and Bien, 2019](https://academic.oup.com/biomet/article/106/3/533/5498375?casa_token=MSUn8MK2SgYAAAAA:r1tkX7-qUE7RIndcJk4_mfKUcuo3SuPImBy8pLX7H5rTA8cp_-7pUn-XzZzpAJuT_Blr8xmLFjvd); [Liu et al, 2020](https://academic.oup.com/biomet/article/107/2/481/5716270?casa_token=EYC-Z7uyoScAAAAA:6kQhSHg6NJEDWKAgJobCfV_HwNxa5uSWD38hzjW8zUj33n8EUJgzPWuT6yiVUVwmgVMook0oUajW)). Tuning parameter grids are automatically created from the data whenever possible.

`ya_glm` comes with a computational backend based on [FISTA](https://epubs.siam.org/doi/pdf/10.1137/080716542?casa_token=cjyK5OxcbSoAAAAA:lQOp0YAVKIOv2-vgGUd_YrnZC9VhbgWvZgj4UPbgfw8I7NV44K82vbIu0oz2-xAACBz9k0Lclw) with adaptive restarts, an [augmented ADMM](https://www.tandfonline.com/doi/full/10.1080/10618600.2015.1114491) algorithm, [cvxpy](https://www.cvxpy.org/index.html), and the [LLA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4295817/) algorithm for non-convex penalties. Path algorithms for fast tuning are supported. It is straightforward to supply your favorite, state of the art optimization algorithm to the package.


`ya_glm` follows a sklearn compatible API, is highly customizable and was inspired by many existing packages including [sklearn](https://scikit-learn.org/stable/), [lightning](https://github.com/scikit-learn-contrib/lightning), [statsmodels](https://www.statsmodels.org/), [pyglmnet](https://github.com/glm-tools/pyglmnet), [celer](https://github.com/mathurinm/celer), [andersoncd](https://github.com/mathurinm/andersoncd), [picasso](https://github.com/jasonge27/picasso), [tick](https://github.com/X-DataInitiative/tick), [PyUNLocBoX](https://github.com/epfl-lts2/pyunlocbox), [regerg](https://github.com/regreg/regreg), [grpreg](https://github.com/pbreheny/grpreg), [ncreg](https://cran.r-project.org/web/packages/ncvreg/index.html), and [glmnet](https://glmnet.stanford.edu/articles/glmnet.html).


 **Beware**: This is a preliminary release of Version 0.3.0. Not all features have been fully added and it has not yet been rigorously tested.


# Installation
`ya_glm` can be installed via github
```
git clone https://github.com/idc9/ya_glm.git
python setup.py install
```


# Examples

`ya_glm` should feel a lot like sklearn -- particularly [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html). The major difference is that we make extensive use of config objects to specify the loss, penalty, penalty flavor, constraint, and solver.


```python
from ya_glm.toy_data import sample_sparse_lin_reg

from ya_glm.GlmTuned import GlmCV, GlmTrainMetric

from ya_glm.config.loss import Huber
from ya_glm.config.penalty import Lasso, GroupLasso
from ya_glm.config.flavor import Adaptive, NonConvexLLA
from ya_glm.solver.FISTA import FISTA

from ya_glm.metrics.info_criteria import InfoCriteria
from ya_glm.infer.Inferencer import Inferencer
from ya_glm.infer.lin_reg_noise_var import ViaRidge

# sample sparse linear regression data
X, y, _ = sample_sparse_lin_reg(n_samples=100, n_features=10)

# fit a lasso penalty tuned via cross-validation with the 1se rule
GlmCV(loss='lin_reg',
      penalty=Lasso(),  # specify penalty with config object
      select_rule='1se'
      ).fit(X, y)

# fit an adaptive lasso tuned via cross-validation
# initialized with a lasso tuned with cross-validation
GlmCV(loss='lin_reg',
      penalty=Lasso(flavor=Adaptive()), 
      initializer='default'
      ).fit(X, y)

# fit an adaptive lasso and tuned via EBIC
# estimate the noise variance via a ridge-regression method
GlmTrainMetric(loss='lin_reg',
               penalty=Lasso(flavor=Adaptive()),
               
               inferencer=Inferencer(scale=ViaRidge()), # noise variance estimator
               scorer=InfoCriteria(crit='ebic') # Info criteria
               ).fit(X, y)

# fit a huber loss with a group SCAD penalty
# both the huber knot parameter and the SCAD penalty parameter are tuned with CV
# the LLA algorithm is initialized with a group Lasso penalty tuned via cross-validation
groups = [range(5), range(5, 10)]
GlmCV(loss=Huber().tune(knot=range(1, 5)),
      penalty=GroupLasso(groups=groups,
                         flavor=NonConvexLLA())
      ).fit(X, y)

# supply your favorite optimization algorithm!
solver = FISTA(max_iter=100)
GlmCV(loss='lin_reg', penalty='lasso', solver= solver)
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


Liu, X., Zheng, S. and Feng, X., 2020. [Estimation of error variance via ridge regression](https://academic.oup.com/biomet/article/107/2/481/5716270?casa_token=EYC-Z7uyoScAAAAA:6kQhSHg6NJEDWKAgJobCfV_HwNxa5uSWD38hzjW8zUj33n8EUJgzPWuT6yiVUVwmgVMook0oUajW). Biometrika, 107(2), pp.481-488.


Loh, P.L. and Wainwright, M.J., 2017. [Support recovery without incoherence: A case for nonconvex regularization](https://projecteuclid.org/journals/annals-of-statistics/volume-45/issue-6/Support-recovery-without-incoherence-A-case-for-nonconvex-regularization/10.1214/16-AOS1530.pdf). The Annals of Statistics, 45(6), pp.2455-2482.

Reid, S., Tibshirani, R. and Friedman, J., 2016. [A study of error variance estimation in lasso regression](https://www.jstor.org/stable/pdf/24721190.pdf?casa_token=wVML37DFzk4AAAAA:PCPZH8z98S_ZDNMyFxtec9-ZsIx73xoxDgWJUEObeJooVLwMWhOAn_Tnf2GQGL3H36XAROk5P08aNGcDnJUG95ahVwe1F57AsJg0_kxntX4UIoSoEAk). Statistica Sinica, pp.35-67.

Yu, G. and Bien, J., 2019. [Estimating the error variance in a high-dimensional linear model](https://academic.oup.com/biomet/article/106/3/533/5498375?casa_token=MSUn8MK2SgYAAAAA:r1tkX7-qUE7RIndcJk4_mfKUcuo3SuPImBy8pLX7H5rTA8cp_-7pUn-XzZzpAJuT_Blr8xmLFjvd). Biometrika, 106(3), pp.533-546.

Zhu, Y., 2017. [An augmented ADMM algorithm with application to the generalized lasso problem](https://www.tandfonline.com/doi/full/10.1080/10618600.2015.1114491). Journal of Computational and Graphical Statistics, 26(1), pp.195-204.

Zhang, Y., Li, R. and Tsai, C.L., 2010. [Regularization parameter selections via generalized information criterion](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.tm08013). Journal of the American Statistical Association, 105(489), pp.312-323.