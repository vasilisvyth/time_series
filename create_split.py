from utils import reduce_memory_footprint
from utils import compact_to_expanded

from interpolation import SeasonalInterpolation

import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
import statsmodels.api as sm
import warnings
import random
from IPython.display import display, HTML
np.random.seed(42)
random.seed(42)
tqdm.pandas()

try:
    lclid_acorn_map = pd.read_pickle("data/london_smart_meters/preprocessed/london_smart_meters_lclid_acorn_map.pkl")
except FileNotFoundError:
    print('no file')

affluent_households = lclid_acorn_map.loc[lclid_acorn_map.Acorn_grouped=="Affluent", ["LCLid",'file']]
adversity_households = lclid_acorn_map.loc[lclid_acorn_map.Acorn_grouped=="Adversity", ["LCLid",'file']]
comfortable_households = lclid_acorn_map.loc[lclid_acorn_map.Acorn_grouped=="Comfortable", ["LCLid",'file']]

selected_households = pd.concat(
    [
        affluent_households.sample(20, random_state=76),
        comfortable_households.sample(20, random_state=76),
        adversity_households.sample(20, random_state=76),
    ]
)
selected_households['block']=selected_households.file.str.split("_", expand=True).iloc[:,1].astype(int)

# extracting the paths to the different blocks and extracting the starting and ending blocks
path_blocks = [
    (p, *list(map(int, p.name.split("_")[5].split(".")[0].split("-"))))
    for p in Path("data/london_smart_meters/preprocessed").glob(
        "london_smart_meters_merged_block*"
    )
]

household_df_l = []
for path, start_b, end_b in tqdm(path_blocks):
    block_df = pd.read_parquet(path)
    selected_households['block'].between
    mask = selected_households['block'].between(start_b, end_b)
    lclids = selected_households.loc[mask, "LCLid"]
    household_df_l.append(block_df.loc[block_df.LCLid.isin(lclids)])

block_df = pd.concat(household_df_l)
del household_df_l


#Converting to expanded form
exp_block_df = compact_to_expanded(block_df, timeseries_col = 'energy_consumption',
static_cols = ["frequency", "series_length", "stdorToU", "Acorn", "Acorn_grouped", "file"],
time_varying_cols = ['holidays', 'visibility', 'windBearing', 'temperature', 'dewPoint',
       'pressure', 'apparentTemperature', 'windSpeed', 'precipType', 'icon',
       'humidity', 'summary'],
ts_identifier = "LCLid")


# ## Reduce Memory Footprint

exp_block_df.info(memory_usage="deep", verbose=False)

exp_block_df = reduce_memory_footprint(exp_block_df)

exp_block_df.info(memory_usage="deep", verbose=False)

# # Train Test Valildation Split

# We are going to keep 2014 data as the validation and test period. We have 2 months(Jan and Feb) of data in 2014. Jan is Validation and Feb is Test

test_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==2)
val_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==1)

train = exp_block_df[~(val_mask|test_mask)]
val = exp_block_df[val_mask]
test = exp_block_df[test_mask]
print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
print(f"Max Date in Train: {train.timestamp.max()} | Min Date in Validation: {val.timestamp.min()} | Min Date in Test: {test.timestamp.min()}")

train.to_parquet("data/london_smart_meters/preprocessed/selected_blocks_train.parquet")
val.to_parquet("data/london_smart_meters/preprocessed/selected_blocks_val.parquet")
test.to_parquet("data/london_smart_meters/preprocessed/selected_blocks_test.parquet")

# ## Train Test Split after filling in missing values

block_df.energy_consumption = block_df.energy_consumption.progress_apply(lambda x: SeasonalInterpolation(seasonal_period=48*7).fit_transform(x.reshape(-1,1)).squeeze())

#Converting to expanded form
exp_block_df = compact_to_expanded(block_df, timeseries_col = 'energy_consumption',
static_cols = ["frequency", "series_length", "stdorToU", "Acorn", "Acorn_grouped", "file"],
time_varying_cols = ['holidays', 'visibility', 'windBearing', 'temperature', 'dewPoint',
       'pressure', 'apparentTemperature', 'windSpeed', 'precipType', 'icon',
       'humidity', 'summary'],
ts_identifier = "LCLid")


exp_block_df.info(memory_usage="deep", verbose=False)

exp_block_df = reduce_memory_footprint(exp_block_df)

exp_block_df.info(memory_usage="deep", verbose=False)

test_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==2)
val_mask = (exp_block_df.timestamp.dt.year==2014) & (exp_block_df.timestamp.dt.month==1)

train = exp_block_df[~(val_mask|test_mask)]
val = exp_block_df[val_mask]
test = exp_block_df[test_mask]
print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
print(f"Max Date in Train: {train.timestamp.max()} | Min Date in Validation: {val.timestamp.min()} | Min Date in Test: {test.timestamp.min()}")

train.to_parquet("data/london_smart_meters/preprocessed/selected_blocks_train_missing_imputed.parquet")
val.to_parquet("data/london_smart_meters/preprocessed/selected_blocks_val_missing_imputed.parquet")
test.to_parquet("data/london_smart_meters/preprocessed/selected_blocks_test_missing_imputed.parquet")


