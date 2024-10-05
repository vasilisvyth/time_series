import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

import warnings
from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_list_like
from window_ops.rolling import (
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_std,
)

from src.utils.data_utils import _get_32_bit_dtype

ALLOWED_AGG_FUNCS = ["mean", "max", "min", "std"]
SEASONAL_ROLLING_MAP = {
    "mean": seasonal_rolling_mean,
    "min": seasonal_rolling_min,
    "max": seasonal_rolling_max,
    "std": seasonal_rolling_std,
}

# adapted from gluonts
def time_features_from_frequency_str(freq_str: str) -> List[str]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """

    features_by_offsets = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.MonthBegin: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.MonthEnd: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.Week: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
        ],
        offsets.Day: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.BusinessDay: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.Hour: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
        ],
        offsets.Minute: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
            "Minute",
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return feature

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y, YS   - yearly
            alias: A
        M, MS   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)


# adapted from fastai
def add_temporal_features(
    df: pd.DataFrame,
    field_name: str,
    frequency: str,
    add_elapsed: bool = True,
    prefix: str = None,
    drop: bool = True,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds columns relevant to a date in the column `field_name` of `df`.

    Args:
        df (pd.DataFrame): Dataframe to which the features need to be added
        field_name (str): The date column which should be encoded using temporal features
        frequency (str): The frequency of the date column so that only relevant features are added.
            If frequency is "Weekly", then temporal features like hour, minutes, etc. doesn't make sense.
        add_elapsed (bool, optional): Add time elapsed as a monotonically increasing function. Defaults to True.
        prefix (str, optional): Prefix to the newly created columns. If left None, will use the field name. Defaults to None.
        drop (bool, optional): Flag to drop the data column after feature creation. Defaults to True.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
    field = df[field_name]
    prefix = (re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix) + "_"
    attr = time_features_from_frequency_str(frequency)
    _32_bit_dtype = "int32"
    added_features = []
    for n in attr:
        if n == "Week":
            continue
        df[prefix + n] = (
            getattr(field.dt, n.lower()).astype(_32_bit_dtype)
            if use_32_bit
            else getattr(field.dt, n.lower())
        )
        added_features.append(prefix + n)
    # Pandas removed `dt.week` in v1.1.10
    if "Week" in attr:
        week = (
            field.dt.isocalendar().week
            if hasattr(field.dt, "isocalendar")
            else field.dt.week
        )
        df.insert(
            3, prefix + "Week", week.astype(_32_bit_dtype) if use_32_bit else week
        )
        added_features.append(prefix + "Week")
    if add_elapsed:
        mask = ~field.isna()
        df[prefix + "Elapsed"] = np.where(
            mask, field.values.astype(np.int64) // 10**9, None
        )
        if use_32_bit:
            if df[prefix + "Elapsed"].isnull().sum() == 0:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("int32")
            else:
                df[prefix + "Elapsed"] = df[prefix + "Elapsed"].astype("float32")
        added_features.append(prefix + "Elapsed")
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df, added_features


def add_lags(
    df: pd.DataFrame,
    lags: List[int],
    column: str,
    ts_id: str = None,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Create Lags for the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        lags (List[int]): List of lags to be created
        column (str): Name of the column to be lagged
        ts_id (str, optional): Column name of Unique ID of a time series to be grouped by before applying the lags.
            If None assumes dataframe only has a single timeseries. Defaults to None.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """

    _32_bit_dtype = _get_32_bit_dtype(df[column])

    assert (
        ts_id in df.columns
    ), "`ts_id` should be a valid column in the provided dataframe"
    if use_32_bit and _32_bit_dtype is not None:
        col_dict = {
            f"{column}_lag_{l}": df.groupby([ts_id])[column]
            .shift(l)
            .astype(_32_bit_dtype)
            for l in lags
        }
    else:
        col_dict = {
            f"{column}_lag_{l}": df.groupby([ts_id])[column].shift(l) for l in lags
        }
    df = df.assign(**col_dict)
    added_features = list(col_dict.keys())
    return df, added_features


def add_rolling_features(
    df: pd.DataFrame,
    rolls: List[int],
    column: str,
    agg_funcs: List[str] = ["mean", "std"],
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Add rolling statistics from the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        rolls (List[int]): Different windows over which the rolling aggregations to be done
        column (str): The column used for feature engineering
        agg_funcs (List[str], optional): The different aggregations to be done on the rolling window. Defaults to ["mean", "std"].
        ts_id (str, optional): Unique id for a time series. Defaults to None.
        n_shift (int, optional): Number of time steps to shift before computing rolling statistics.
            Typically used to avoid data leakage. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """

    _32_bit_dtype = _get_32_bit_dtype(df[column])
    
    assert (
        ts_id in df.columns
    ), "`ts_id` should be a valid column in the provided dataframe"
    rolling_df = pd.concat(
        [
            df.groupby(ts_id)[column]
            .shift(n_shift)
            .rolling(l)
            .agg({f"{column}_rolling_{l}_{agg}": agg for agg in agg_funcs})
            for l in rolls
        ],
        axis=1,
    )

    df = df.assign(**rolling_df.to_dict("list"))
    added_features = rolling_df.columns.tolist()
    if use_32_bit and _32_bit_dtype is not None:
        df[added_features] = df[added_features].astype(_32_bit_dtype)
    return df, added_features


def add_seasonal_rolling_features(
    df: pd.DataFrame,
    seasonal_periods: List[int],
    rolls: List[int],
    column: str,
    agg_funcs: List[str] = ["mean", "std"],
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Add seasonal rolling statistics from the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        seasonal_periods (List[int]): List of seasonal periods over which the seasonal rolling operations should be done
        rolls (List[int]): List of seasonal rolling window over which the aggregation functions will be applied
        column (str): [description]
        agg_funcs (List[str], optional): The different aggregations to be done on the rolling window. Defaults to ["mean", "std"].. Defaults to ["mean", "std"].
        ts_id (str, optional): Unique id for a time series. Defaults to None.
        n_shift (int, optional): The number of seasonal shifts to be applied before the seasonal rolling operation.
            Typically used to avoid data leakage. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List]: Returns a tuple of the new dataframe and a list of features which were added
    """
   
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    agg_funcs = {agg: SEASONAL_ROLLING_MAP[agg] for agg in agg_funcs}
    added_features = []
    for sp in seasonal_periods:
        
        assert (
            ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_{sp}_seasonal_rolling_{l}_{name}": df.groupby(ts_id)[
                    column
                ]
                .transform(
                    lambda x: agg(
                        x.shift(n_shift * sp).values,
                        season_length=sp,
                        window_size=l,
                    )
                )
                .astype(_32_bit_dtype)
                for (name, agg) in agg_funcs.items()
                for l in rolls
            }
        else:
            col_dict = {
                f"{column}_{sp}_seasonal_rolling_{l}_{name}": df.groupby(ts_id)[
                    column
                ].transform(
                    lambda x: agg(
                        x.shift(n_shift * sp).values,
                        season_length=sp,
                        window_size=l,
                    )
                )
                for (name, agg) in agg_funcs.items()
                for l in rolls
            }
        df = df.assign(**col_dict)
        added_features += list(col_dict.keys())
    return df, added_features


def add_ewma(
    df: pd.DataFrame,
    column: str,
    alphas: List[float] = [0.5],
    spans: List[float] = None,
    ts_id: str = None,
    n_shift: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Create Exponentially Weighted Average for the column provided and adds them as other columns in the provided dataframe

    Args:
        df (pd.DataFrame): The dataframe in which features needed to be created
        column (str): Name of the column to be lagged
        alphas (List[float]): List of alphas (smoothing parameters) using which ewmas are be created
        spans (List[float]): List of spans using which ewmas are be created. When we refer to a 60 period EWMA, span is 60.
            alpha = 2/(1+span). If span is given, we ignore alpha.
        ts_id (str, optional): Unique ID of a time series to be grouped by before applying the lags.
            If None assumes dataframe only has a single timeseries. Defaults to None.
        n_shift (int, optional): Number of time steps to shift before computing ewma.
            Typically used to avoid data leakage. Defaults to 1.
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.

    Returns:
        Tuple(pd.DataFrame, List): Returns a tuple of the new dataframe and a list of features which were added
    """
    if spans is not None:
        assert isinstance(
            spans, list
        ), "`spans` should be a list of all required period spans"
        use_spans = True
    if alphas is not None:
        assert isinstance(
            alphas, list
        ), "`alphas` should be a list of all required smoothing parameters"
    if spans is None and alphas is None:
        raise ValueError(
            "Either `alpha` or `spans` should be provided for the function to"
        )
    assert (
        column in df.columns
    ), "`column` should be a valid column in the provided dataframe"
    _32_bit_dtype = _get_32_bit_dtype(df[column])
    if ts_id is None:
        warnings.warn(
            "Assuming just one unique time series in dataset. If there are multiple, provide `ts_id` argument"
        )
        # Assuming just one unique time series in dataset
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                .astype(_32_bit_dtype)
                for param in (spans if use_spans else alphas)
            }
        else:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                for param in (spans if use_spans else alphas)
            }
    else:
        assert (
            ts_id in df.columns
        ), "`ts_id` should be a valid column in the provided dataframe"
        if use_32_bit and _32_bit_dtype is not None:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df.groupby(
                    [ts_id]
                )[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                .astype(_32_bit_dtype)
                for param in (spans if use_spans else alphas)
            }
        else:
            col_dict = {
                f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": df.groupby(
                    [ts_id]
                )[column]
                .shift(n_shift)
                .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                )
                .mean()
                for param in (spans if use_spans else alphas)
            }
    df = df.assign(**col_dict)
    return df, list(col_dict.keys())
