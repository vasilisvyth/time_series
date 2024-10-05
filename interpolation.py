import warnings
from sklearn.utils import check_array
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import warnings
import numpy as np
import logging
from sklearn.utils import check_array

def generate_random_column_samples(column):
    col_mask = np.isnan(column)
    n_missing = np.sum(col_mask)
    if n_missing == len(column):
        logging.warn("No observed values in column")
        return np.zeros_like(column)

    mean = np.nanmean(column)
    std = np.nanstd(column)

    if np.isclose(std, 0):
        return np.array([mean] * n_missing)
    else:
        return np.random.randn(n_missing) * std + mean


class Solver(object):
    def __init__(
            self,
            fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None):
        self.fill_method = fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer

    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if v is None or isinstance(v, (float, int)):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(field_list))

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            warnings.simplefilter("always")
            warnings.warn("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def fill(
            self,
            X,
            missing_mask,
            fill_method=None,
            inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries
        missing_mask : np.array
            Boolean array indicating where NaN entries are
        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column
        inplace : bool
            Modify matrix or fill a copy
        """
        X = check_array(X, force_all_finite=False)

        if not inplace:
            X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        X = check_array(X, force_all_finite=False)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def project_result(self, X):
        """
        First undo normalization and then clip to the user-specified min/max
        range.
        """
        X = np.asarray(X)
        if self.normalizer is not None:
            X = self.normalizer.inverse_transform(X)
        return self.clip(X)

    def solve(self, X, missing_mask):
        """
        Given an initialized matrix X and a mask of where its missing values
        had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = X_original.copy()
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)
        X_filled = self.fill(X, missing_mask, inplace=True)
        if not isinstance(X_filled, np.ndarray):
            raise TypeError(
                "Expected %s.fill() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_filled)))

        X_result = self.solve(X_filled, missing_mask)
        if not isinstance(X_result, np.ndarray):
            raise TypeError(
                "Expected %s.solve() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_result)))

        X_result = self.project_result(X=X_result)
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def fit(self, X, y=None):
        """
        Fit the imputer on input `X`.
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.fit not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only fit_transform is "
            "supported at this time." % (
                self.__class__.__name__,))

    def transform(self, X, y=None):
        """
        Transform input `X`.
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.transform not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only %s.fit_transform is "
            "supported at this time." % (
                self.__class__.__name__, self.__class__.__name__))
    

class SeasonalInterpolation(Solver):
    def __init__(
        self,
        seasonal_period: int,
        decomposition_strategy: str = "additive",
        decomposition_args: dict = {},
        interpolation_strategy: str = "linear",
        interpolation_args: dict = {},
        fill_border_values: int = 0,
        min_value: int = None,
        max_value: int = None,
        verbose: bool = True,
    ):

        Solver.__init__(
            self, fill_method="zero", min_value=min_value, max_value=max_value
        )
        # Check if dec_model has a valid value:
        if decomposition_strategy not in ["multiplicative", "additive"]:
            raise ValueError(
                decomposition_strategy + " is not a supported decomposition strategy."
            )
        if (
            interpolation_strategy in ["spline", "polynomial"]
            and "order" not in interpolation_args.keys()
        ):  # ['linear', 'nearest', "zero", "quadratic", "spline", "polynomial"]
            raise ValueError(
                interpolation_strategy
                + " interpolation strategy needs an order to be sopecified in the interpolation_args."
            )
        self.interpolation_strategy = interpolation_strategy
        self.decomposition_strategy = decomposition_strategy
        extrapolate = decomposition_args["extrapolate_trend"] if "extrapolate_trend" in decomposition_args.keys() else "freq"
        decomposition_args.update(
            {"model": decomposition_strategy, "period": seasonal_period, "extrapolate_trend": extrapolate}
        )
        interpolation_args.update(
            {
                "method": interpolation_strategy,
            }
        )
        self.interpolation_args = interpolation_args
        self.decomposition_args = decomposition_args
        self.fill_border_values = fill_border_values
        self.verbose = verbose

    def fit_transform(self, X, y=None):

        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = check_array(X, force_all_finite=False)
        if missing_mask.sum() == 0:
            warnings.warn(
                "[Seasonal Interpolation] Warning: provided matrix doesn't contain any missing values."
            )
            warnings.warn(
                "[Seasonal Interpolation] The algorithm will run, but will return an unchanged matrix."
            )
        X_filled = (
            pd.DataFrame(X)
            .interpolate(axis=0, **self.interpolation_args)
            .fillna(self.fill_border_values)
            .values
        )
        trends = []
        resids = []
        seasonality = []
        for col in range(X_original.shape[1]):
            decomposition = seasonal_decompose(
                X_filled[:, col], **self.decomposition_args
            )
            trends.append(decomposition.trend)
            resids.append(decomposition.resid)
            seasonality.append(decomposition.seasonal)
        trends = np.vstack(trends).T
        resids = np.vstack(resids).T
        seasonality = np.vstack(seasonality).T
        if self.decomposition_strategy == "additive":
            deseasonalized = trends + resids
        elif self.decomposition_strategy == "multiplicative":
            deseasonalized = trends * resids
        deseasonalized[missing_mask] = np.nan
        deseasonalized = (
            pd.DataFrame(deseasonalized)
            .interpolate(axis=0, **self.interpolation_args)
            .fillna(self.fill_border_values)
            .values
        )
        if self.decomposition_strategy == "additive":
            X_result = deseasonalized + seasonality
        elif self.decomposition_strategy == "multiplicative":
            X_result = deseasonalized * seasonality
        X_result = self.clip(X_result)
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def fit(self, X, y=None):
 
        raise ValueError(
            "%s.fit not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only fit_transform is "
            "supported at this time." % (self.__class__.__name__,)
        )

    def transform(self, X, y=None):

        raise ValueError(
            "%s.transform not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only %s.fit_transform is "
            "supported at this time."
            % (self.__class__.__name__, self.__class__.__name__)
        )
