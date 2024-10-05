import numpy as np
import pandas as pd


import warnings
from pathlib import Path



from itertools import cycle


from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor



from tqdm.autonotebook import tqdm

def mae(actuals, predictions):
    return np.nanmean(np.abs(actuals-predictions))

def mse(actuals, predictions):
    return np.nanmean(np.power(actuals-predictions, 2))

def forecast_bias_aggregate(actuals, predictions):
    return 100*(np.nansum(predictions)-np.nansum(actuals))/np.nansum(actuals)


np.random.seed(42)
tqdm.pandas()

preprocessed = Path("data/london_smart_meters/preprocessed")
output = Path("data/london_smart_meters/output")

try:
    train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed_feature_engg.parquet")
    # Read in the Validation dataset as test_df so that we predict on it
    test_df = pd.read_parquet(preprocessed/"selected_blocks_val_missing_imputed_feature_engg.parquet")

except FileNotFoundError:
    print("""File not found.""")


from ml_forecasting import (
    FeatureConfig,
    MissingValueConfig,
    MLForecast,
    ModelConfig,
    calculate_metrics,
)

feat_config = FeatureConfig(
    date="timestamp",
    target="energy_consumption",
    continuous_features=[
        "visibility",
        "windBearing",
        "temperature",
        "dewPoint",
        "pressure",
        "apparentTemperature",
        "windSpeed",
        "humidity",
        "energy_consumption_lag_1",
        "energy_consumption_lag_2",
        "energy_consumption_lag_3",
        "energy_consumption_lag_4",
        "energy_consumption_lag_5",
        "energy_consumption_lag_46",
        "energy_consumption_lag_47",
        "energy_consumption_lag_48",
        "energy_consumption_lag_49",
        "energy_consumption_lag_50",
        "energy_consumption_lag_334",
        "energy_consumption_lag_335",
        "energy_consumption_lag_336",
        "energy_consumption_lag_337",
        "energy_consumption_lag_338",
        "energy_consumption_rolling_3_mean",
        "energy_consumption_rolling_3_std",
        "energy_consumption_rolling_6_mean",
        "energy_consumption_rolling_6_std",
        "energy_consumption_rolling_12_mean",
        "energy_consumption_rolling_12_std",
        "energy_consumption_rolling_48_mean",
        "energy_consumption_rolling_48_std",
        "energy_consumption_48_seasonal_rolling_3_mean",
        "energy_consumption_48_seasonal_rolling_3_std",
        "energy_consumption_336_seasonal_rolling_3_mean",
        "energy_consumption_336_seasonal_rolling_3_std",
        "energy_consumption_ewma_span_2880",
        "energy_consumption_ewma_span_336",
        "energy_consumption_ewma_span_48",
        "timestamp_Elapsed",
        "timestamp_Month_sin_1",
        "timestamp_Month_sin_2",
        "timestamp_Month_sin_3",
        "timestamp_Month_sin_4",
        "timestamp_Month_sin_5",
        "timestamp_Month_cos_1",
        "timestamp_Month_cos_2",
        "timestamp_Month_cos_3",
        "timestamp_Month_cos_4",
        "timestamp_Month_cos_5",
        "timestamp_Hour_sin_1",
        "timestamp_Hour_sin_2",
        "timestamp_Hour_sin_3",
        "timestamp_Hour_sin_4",
        "timestamp_Hour_sin_5",
        "timestamp_Hour_cos_1",
        "timestamp_Hour_cos_2",
        "timestamp_Hour_cos_3",
        "timestamp_Hour_cos_4",
        "timestamp_Hour_cos_5",
        "timestamp_Minute_sin_1",
        "timestamp_Minute_sin_2",
        "timestamp_Minute_sin_3",
        "timestamp_Minute_sin_4",
        "timestamp_Minute_sin_5",
        "timestamp_Minute_cos_1",
        "timestamp_Minute_cos_2",
        "timestamp_Minute_cos_3",
        "timestamp_Minute_cos_4",
        "timestamp_Minute_cos_5",
    ],
    categorical_features=[
        "holidays",
        "precipType",
        "icon",
        "summary",
        "timestamp_Month",
        "timestamp_Quarter",
        "timestamp_WeekDay",
        "timestamp_Dayofweek",
        "timestamp_Dayofyear",
        "timestamp_Hour",
        "timestamp_Minute",
    ],
    boolean_features=[
        "timestamp_Is_quarter_end",
        "timestamp_Is_quarter_start",
        "timestamp_Is_year_end",
        "timestamp_Is_year_start",
        "timestamp_Is_month_start",
    ],
    index_cols=["timestamp"],
    exogenous_features=[
        "holidays",
        "precipType",
        "icon",
        "summary",
        "visibility",
        "windBearing",
        "temperature",
        "dewPoint",
        "pressure",
        "apparentTemperature",
        "windSpeed",
        "humidity",
    ],
)


missing_value_config = MissingValueConfig(
    bfill_columns=[
        "energy_consumption_lag_1",
        "energy_consumption_lag_2",
        "energy_consumption_lag_3",
        "energy_consumption_lag_4",
        "energy_consumption_lag_5",
        "energy_consumption_lag_46",
        "energy_consumption_lag_47",
        "energy_consumption_lag_48",
        "energy_consumption_lag_49",
        "energy_consumption_lag_50",
        "energy_consumption_lag_334",
        "energy_consumption_lag_335",
        "energy_consumption_lag_336",
        "energy_consumption_lag_337",
        "energy_consumption_lag_338",
        "energy_consumption_rolling_3_mean",
        "energy_consumption_rolling_3_std",
        "energy_consumption_rolling_6_mean",
        "energy_consumption_rolling_6_std",
        "energy_consumption_rolling_12_mean",
        "energy_consumption_rolling_12_std",
        "energy_consumption_rolling_48_mean",
        "energy_consumption_rolling_48_std",
        "energy_consumption_48_seasonal_rolling_3_mean",
        "energy_consumption_48_seasonal_rolling_3_std",
        "energy_consumption_336_seasonal_rolling_3_mean",
        "energy_consumption_336_seasonal_rolling_3_std",
        "energy_consumption_ewma__span_2880",
        "energy_consumption_ewma__span_336",
        "energy_consumption_ewma__span_48",
    ],
    ffill_columns=[],
    zero_fill_columns=[],
)

def evaluate_model(
    model_config,
    feature_config,
    missing_config,
    train_features,
    train_target,
    test_features,
    test_target,
):
    ml_model = MLForecast(
        model_config=model_config,
        feature_config=feat_config,
        missing_config=missing_value_config,
    )
    ml_model.fit(train_features, train_target)
    y_pred = ml_model.predict(test_features)
    feat_df = ml_model.feature_importance()
    metrics = calculate_metrics(test_target, y_pred, model_config.name, train_target)
    return y_pred, metrics, feat_df



# Running XGB Random Forest, and LightGBM

lcl_ids = sorted(train_df.LCLid.unique())
models_to_run = [
    ModelConfig(
        model=XGBRFRegressor(random_state=42, max_depth=4),
        name="XGB Random Forest",
        normalize=False,
        fill_missing=False,
    ),
    ModelConfig(
        model=LGBMRegressor(random_state=42),
        name="LightGBM",
        normalize=False,
        fill_missing=False,
    ),
]

all_preds = []
all_metrics = []


for lcl_id in tqdm(lcl_ids):
    for model_config in models_to_run:
        model_config = model_config.clone()
        X_train, y_train, _ = feat_config.get_X_y(
            train_df.loc[train_df.LCLid == lcl_id, :],
            categorical=False,
            exogenous=False,
        )
        X_test, y_test, _ = feat_config.get_X_y(
            test_df.loc[test_df.LCLid == lcl_id, :], categorical=False, exogenous=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred, metrics, feat_df = evaluate_model(
                model_config,
                feat_config,
                missing_value_config,
                X_train,
                y_train,
                X_test,
                y_test,
            )
        y_pred.name = "predictions"
        y_pred = y_pred.to_frame()
        y_pred["LCLid"] = lcl_id
        y_pred["Algorithm"] = model_config.name
        metrics["LCLid"] = lcl_id
        metrics["Algorithm"] = model_config.name
        y_pred["energy_consumption"] = y_test.values
        all_preds.append(y_pred)
        all_metrics.append(metrics)

pred_df = pd.concat(all_preds)



for model_config in models_to_run:
    pred_mask = pred_df.Algorithm==model_config.name

    print(f"Algorithm: {model_config.name}")
    print(f"MAE: {mae(pred_df.loc[pred_mask, 'energy_consumption'], pred_df.loc[pred_mask, 'predictions'])}")
    print(f"MSE: {mse(pred_df.loc[pred_mask, 'energy_consumption'], pred_df.loc[pred_mask, 'predictions'])}")
    print(f"Forecast Bias: {forecast_bias_aggregate(pred_df.loc[pred_mask, 'energy_consumption'], pred_df.loc[pred_mask, 'predictions'])}")

