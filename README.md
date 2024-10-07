Data:

Create a folder named data and put [this folder](https://drive.google.com/drive/folders/1osVwoZmfAC1sZcIwz9jJVUb4UkIxiIXy?usp=sharing) there

Scripts
1. check_stationarity.py

This script checks the stationarity of time series data and performs the following transformations and tests:

    Log Transformation: Applies a logarithmic transformation to stabilize variance and make the data more normally distributed.
    Box-Cox Transformation: Implements the Box-Cox method to transform non-normal dependent variables into a normal shape.
    Heteroscedasticity Test (P-Value): White’s Lagrange Multiplier Test for Heteroscedasticity.
    Unit Root Test: Performs a unit root test (ADF test) to determine if the time series is stationary or contains a unit root.

2. feature_engineering.py

This script creates new features to improve the predictive power of machine learning models. It includes:

    Seasonal Rolling Statistics: Generates rolling mean and standard deviation features that capture seasonal trends in the data.
    Temporal Features: Creates time-based features, such as day of the week, month, or year, to capture patterns over time.

3. run_ml_algorithms.py

This script runs machine learning algorithms for time series regression. It currently supports:

    Random Forest Regressor: A powerful ensemble learning method based on decision trees.
    XGBRFRegressor: XGBoost's Random Forest variant, which combines the benefits of gradient boosting and random forest methods for robust predictions.

