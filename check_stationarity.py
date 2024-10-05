import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm.autonotebook import tqdm
import warnings
from scipy import optimize
import matplotlib.pyplot as plt
import joblib
from scipy.stats import boxcox, variation
import random
from scipy.special import inv_boxcox
from collections import namedtuple
from typing import Tuple
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

def _check_convert_y(y):
    assert not np.any(np.isnan(y)), "`y` should not have any nan values"
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.squeeze()
    assert y.ndim==1
    return y

def _check_stationary_adfuller(y, confidence, **kwargs):
    y = _check_convert_y(y)
    res = namedtuple("ADF_Test", ["stationary", "results"])
    result = adfuller(y, **kwargs)
    if result[1]>confidence:
        return res(False, result)
    else:
        return res(True, result)
    
def check_unit_root(y, confidence=0.05, adf_params={}):
    adf_params['regression'] = "c"
    return _check_stationary_adfuller(y, confidence, **adf_params)

def generate_autoregressive_series(length, phi):
    x_start = random.uniform(-1, 1)
    y = []
    for i in range(length):
        t = x_start*phi+random.uniform(-1, 1)
        y.append(t)
        x_start = t
    return np.array(y)

#https://towardsdatascience.com/heteroscedasticity-is-nothing-to-be-afraid-of-730dd3f7ca1f
def check_heteroscedastisticity(y, confidence=0.05):
    y = _check_convert_y(y)
    res = namedtuple("White_Test", ["heteroscedastic", "lm_statistic", "lm_p_value"])
    #Fitting a linear trend regression
    x = np.arange(len(y))
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    lm_stat, lm_p_value, f_stat, f_p_value = het_white(results.resid, x)
    if lm_p_value<confidence and f_p_value < confidence:
        hetero = True
    else:
        hetero = False
    return res(hetero, lm_stat, lm_p_value)

class LogTransformer:
    def __init__(self, add_one: bool = True) -> None:
        """The logarithmic transformer

        Args:
            add_one (bool, optional): Flag to add one to the series before applying log
                to avoid log 0. Defaults to True.
        """
        self.add_one = add_one

    def fit_transform(self, y: pd.Series):

        self.fit(y)
        return self.transform(y)

    def fit(self, y: pd.Series):

        return self

    def transform(self, y: pd.Series) -> pd.Series:

        return np.log1p(y) if self.add_one else np.log(y)

    def inverse_transform(self, y: pd.Series) -> pd.Series:

        return np.expm1(y) if self.add_one else np.exp(y)


class BoxCoxTransformer:
    def __init__(
        self,
        boxcox_lambda: float = None,
        seasonal_period: int = None,
        optimization="guerrero",
        bounds: Tuple[int, int] = (-1, 2),
        add_one: bool = True,
    ) -> None:

        assert optimization in [
            "guerrero",
            "loglikelihood",
        ], "`optimization should be one of ['guerrero', 'loglikelihood']"
        self.boxcox_lambda = boxcox_lambda
        self.seasonal_period = seasonal_period
        self.optimization = optimization
        self.add_one = add_one
        # input checks on bounds
        if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[1] < bounds[0]:
            raise ValueError(
                f"`bounds` must be a tuple of length 2, and upper should be greater than lower, but found: {bounds}"
            )
        self.bounds = bounds
        if boxcox_lambda is None:
            self._do_optimize = True
            if optimization == "guerrero" and seasonal_period is None:
                raise ValueError(
                    "For Guerrero method of finding optimal lambda for box-cox transform, seasonal_period is needed."
                )
        else:
            self._do_optimize = False
        self._is_fited = False

    def fit_transform(self, y: pd.Series):
  
        self.fit(y)
        return self.transform(y)

    def _add_one(self, y):
        if self.add_one:
            return y + 1
        else:
            return y

    def _subtract_one(self, y):
        if self.add_one:
            return y - 1
        else:
            return y

    def _optimize_lambda(self, y):
        if self.optimization == "loglikelihood":
            _, lmbda = boxcox(y)
        elif self.optimization == "guerrero":
            lmbda = self._guerrero(y, self.seasonal_period, self.bounds)
        return lmbda

    # Adapted from https://github.com/alan-turing-institute/sktime/blob/db0242f6e0230ee3a96d0a62973535d9c328c2ea/sktime/transformations/series/boxcox.py#L305
    @staticmethod
    def _guerrero(x, sp, bounds=None):
        r"""
        Returns lambda estimated by the Guerrero method [Guerrero].
        Parameters
        ----------
        x : ndarray
            Input array. Must be 1-dimensional.
        sp : integer
            Seasonal periodicity value. Must be an integer >= 2
        bounds : {None, (float, float)}, optional
            Bounds on lambda to be used in minimization.
        Returns
        -------
        lambda : float
            Lambda value that minimizes the coefficient of variation of
            variances of the time series in different periods after
            Box-Cox transformation [Guerrero].
        References
        ----------
        [Guerrero] V.M. Guerrero, "Time-series analysis supported by Power
        Transformations ", Journal of Forecasting, Vol. 12, 37-48 (1993)
        https://doi.org/10.1002/for.3980120104
        """

        if sp is None or not isinstance(sp, int) or sp < 2:
            raise ValueError(
                "Guerrero method requires an integer seasonal periodicity (sp) value >= 2."
            )

        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Data must be 1-dimensional.")

        num_obs = len(x)
        len_prefix = num_obs % sp

        x_trimmed = x[len_prefix:]
        x_mat = x_trimmed.reshape((-1, sp))
        x_mean = np.mean(x_mat, axis=1)

        # [Guerrero, Eq.(5)] uses an unbiased estimation for
        # the standard deviation
        x_std = np.std(x_mat, axis=1, ddof=1)

        def _eval_guerrero(lmb, x_std, x_mean):
            x_ratio = x_std / x_mean ** (1 - lmb)
            x_ratio_cv = variation(x_ratio)
            return x_ratio_cv

        return optimize.fminbound(
            _eval_guerrero, bounds[0], bounds[1], args=(x_std, x_mean)
        )

    def fit(self, y: pd.Series):
  
     
        y = self._add_one(y)
        if self._do_optimize:
            self.boxcox_lambda = self._optimize_lambda(y)
        self._is_fitted = True
        return self

    def transform(self, y: pd.Series) -> pd.Series:


        y = self._add_one(y)
        return pd.Series(boxcox(y.values, lmbda=self.boxcox_lambda), index=y.index)

    def inverse_transform(self, y: pd.Series) -> pd.Series:
    

        return pd.Series(
            self._subtract_one(inv_boxcox(y.values, self.boxcox_lambda)), index=y.index
        )



np.random.seed(42)
tqdm.pandas()

preprocessed = Path("data/london_smart_meters/preprocessed")

# # Unit Roots

# ## Plotting Autoregressive series with different $\phi$


length = 100
index = pd.date_range(start="2021-11-03", periods=length)
#unit root
_y_random = pd.Series(np.random.randn(length), index=index)
y_unit_root = _y_random.cumsum()
_y_random = pd.Series(np.random.randn(length), index=index)
t = np.arange(len(_y_random))
y_hetero = (_y_random*t)


res = check_unit_root(y_unit_root, confidence=0.05)

print(f"Stationary: {res.stationary} | p-value: {res.results[1]}")


#shifting the series into positive domain
_y_hetero = y_hetero-y_hetero.min()
log_transformer = LogTransformer(add_one=True)
y_log = log_transformer.fit_transform(_y_hetero)
fig, axs = plt.subplots(2)
_y_hetero.plot(title="Heteroscedastic",ax=axs[0])
y_log.plot(title="Log Transform",ax=axs[1])
plt.tight_layout()
plt.show()
hetero_res = check_heteroscedastisticity(y_log, confidence=0.05)
# mann_kendall_res = check_trend(y_diff, confidence=0.05, mann_kendall=True)
print(f"White Test for Heteroscedasticity: {hetero_res.heteroscedastic} with a p-value of {hetero_res.lm_p_value}")

# ## Box-Cox Transforms

#shifting the series into positive domain
_y_hetero = y_hetero-y_hetero.min()
#Arbritarily divided the data into sub-series of length 25
boxcox_transformer = BoxCoxTransformer(seasonal_period=25, add_one=True, optimization="guerrero")
y_boxcox = boxcox_transformer.fit_transform(_y_hetero)
print(f"Optimal Lambda: {boxcox_transformer.boxcox_lambda}")
fig, axs = plt.subplots(2)
_y_hetero.plot(title="Heteroscedastic",ax=axs[0])
y_boxcox.plot(title="Log Transform",ax=axs[1])
plt.tight_layout()
plt.show()
hetero_res = check_heteroscedastisticity(y_boxcox, confidence=0.05)
# mann_kendall_res = check_trend(y_diff, confidence=0.05, mann_kendall=True)
print(f"White Test for Heteroscedasticity: {hetero_res.heteroscedastic} with a p-value of {hetero_res.lm_p_value}")

# ### Optimizing for $\lambda$ with Loglikelihood

#shifting the series into positive domain
_y_hetero = y_hetero-y_hetero.min()
boxcox_transformer = BoxCoxTransformer(add_one=True, optimization="loglikelihood")
y_boxcox = boxcox_transformer.fit_transform(_y_hetero)
print(f"Optimal Lambda: {boxcox_transformer.boxcox_lambda}")
fig, axs = plt.subplots(2)
_y_hetero.plot(title="Heteroscedastic",ax=axs[0])
y_boxcox.plot(title="Log Transform",ax=axs[1])
plt.tight_layout()
plt.show()
hetero_res = check_heteroscedastisticity(y_boxcox, confidence=0.05)
# mann_kendall_res = check_trend(y_diff, confidence=0.05, mann_kendall=True)
print(f"White Test for Heteroscedasticity: {hetero_res.heteroscedastic} with a p-value of {hetero_res.lm_p_value}")