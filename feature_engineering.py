import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.autonotebook import tqdm
from IPython.display import display, HTML

# from src.feature_engineering.temporal_features import add_temporal_features
from features_utils import add_temporal_features, add_lags, add_rolling_features, add_seasonal_rolling_features, add_ewma


np.random.seed(42)
tqdm.pandas()

preprocessed = Path("data/london_smart_meters/preprocessed")

#Readin the missing value imputed and train test split data
try:
    train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed.parquet")
    val_df = pd.read_parquet(preprocessed/"selected_blocks_val_missing_imputed.parquet")
    test_df = pd.read_parquet(preprocessed/"selected_blocks_test_missing_imputed.parquet")
except FileNotFoundError:
    print(""" File not found""")

train_df["type"] = "train"
val_df["type"] = "val"
test_df["type"] = "test"
full_df = pd.concat([train_df, val_df, test_df]).sort_values(["LCLid", "timestamp"])
del train_df, test_df, val_df

    
lags = (
    (np.arange(5) + 1).tolist()
    + (np.arange(5) + 46).tolist()
    + (np.arange(5) + (48 * 7) - 2).tolist()
)


full_df, added_features = add_lags(
    full_df, lags=lags, column="energy_consumption", ts_id="LCLid", use_32_bit=True
)
print(f"Features Created: {','.join(added_features)}")

# ## Rolling


full_df, added_features = add_rolling_features(
    full_df,
    rolls=[3, 6, 12, 48],
    column="energy_consumption",
    agg_funcs=["mean", "std"],
    ts_id="LCLid",
    use_32_bit=True,
)
print(f"Features Created: {','.join(added_features)}")




full_df, added_features = add_seasonal_rolling_features(
    full_df,
    rolls=[3],
    seasonal_periods=[48, 48 * 7],
    column="energy_consumption",
    agg_funcs=["mean", "std"],
    ts_id="LCLid",
    use_32_bit=True,
)
print(f"Features Created: {','.join(added_features)}")

# EWMA

#
t = np.arange(25).tolist()
plot_df = pd.DataFrame({"Timesteps behind t": t})
for alpha in [0.3, 0.5, 0.8]:
    weights = [alpha * math.pow((1 - alpha), i) for i in t]
    span = (2 - alpha) / alpha
    halflife = math.log(1 - alpha) / math.log(0.5)
    plot_df[f"Alpha={alpha} | Span={span:.2f}"] = weights



full_df, added_features = add_ewma(
    full_df,
    spans=[48 * 60, 48 * 7, 48],
    column="energy_consumption",
    ts_id="LCLid",
    use_32_bit=True,
)
print(f"Features Created: {','.join(added_features)}")

# ## Temporal Features

full_df, added_features = add_temporal_features(
    full_df,
    field_name="timestamp",
    frequency="30min",
    add_elapsed=True,
    drop=False,
    use_32_bit=True,
)
print(f"Features Created: {','.join(added_features)}")


full_df[full_df["type"] == "train"].drop(columns="type").to_parquet(
    preprocessed / "selected_blocks_train_missing_imputed_feature_engg.parquet"
)
full_df[full_df["type"] == "val"].drop(columns="type").to_parquet(
    preprocessed / "selected_blocks_val_missing_imputed_feature_engg.parquet"
)
full_df[full_df["type"] == "test"].drop(columns="type").to_parquet(
    preprocessed / "selected_blocks_test_missing_imputed_feature_engg.parquet"
)


