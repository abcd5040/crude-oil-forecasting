import numpy as np
import pandas as pd
import io
import sklearn
import datetime as dt
from arch import arch_model
from datetime import datetime
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import Holt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def get_WTI_data():
    df = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\DCOILWTICO.csv")
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.set_index('observation_date') # indexing is required for time-based interpolation
    return df 

def get_Brent_data():
    df = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\DCOILBRENTEU.csv")
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.set_index('observation_date') # indexing is required for time-based interpolation
    return df 

def get_refinery_US_data():
    df = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\U.S._Gross_Inputs_to_Refineries.csv")
    df['Month'] = pd.to_datetime(df['Month'], format = '%b %Y')
    df = df.set_index('Month') # indexing is required for time-based interpolation and reversing
    df = df.reindex(index=df.index[::-1]) 
    return df 

def get_Cushing_inv_data():
    df = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\Weekly_Cushing_OK_Ending_Stocks_excluding_SPR_of_Crude_Oil.csv")
    df['Week of'] = pd.to_datetime(df['Week of'])
    df = df.set_index('Week of') # indexing is required for time-based interpolation and reversing
    df = df.reindex(index=df.index[::-1])
    return df 

def get_US_inv_no_SPR_data():
    df = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\Weekly_U.S._Ending_Stocks_excluding_SPR_of_Crude_Oil_and_Petroleum_Products.csv")
    df['Week of'] = pd.to_datetime(df['Week of'])
    df = df.set_index('Week of') # indexing is required for time-based interpolation and reversing
    df = df.reindex(index=df.index[::-1]) 
    return df 

def get_US_inv_data():
    df = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\Weekly_U.S._Ending_Stocks_of_Crude_Oil_and_Petroleum_Products.csv")
    df['Week of'] = pd.to_datetime(df['Week of'])
    df = df.set_index('Week of') # indexing is required for time-based interpolation and reversing
    df = df.reindex(index=df.index[::-1])
    return df 

def plot_df(df, x, y, title="", xlabel='timestamp', ylabel='sell', dpi=100):
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='tab:red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation
    plt.margins(x=0)

def get_seasonal_strength(series, period):
    stl = STL(series, period=period, robust=True).fit()
    seasonal = stl.seasonal
    resid = stl.resid

    var_resid = np.var(resid)
    var_seas_resid = np.var(seasonal + resid)

    fh_strength = max(0, 1 - (var_resid / var_seas_resid))
    return fh_strength

def EDA(df, y, cycle):
    is_na = df[y].isna() # cumsum method for missing data analysis
    gap_groups = (~is_na).cumsum()
    gap_sizes = is_na[is_na].groupby(gap_groups).size() 
    gap_frequencies = gap_sizes.value_counts().sort_index() 
    print(gap_frequencies)

    missing_percentage = df[y].isna().mean() * 100
    print(f"Missing data: {missing_percentage:.2f}%")

    df[y] = df[y].interpolate(method='time') # linear interpolation

    # seasonal decomposition
    result = seasonal_decompose(df[y], model='additive', period=cycle) # ~261 business days/year
    result.plot()

    fh_strength_annual = get_seasonal_strength(df[y], period=cycle)
    print(f"Annual Seasonal Strength: {fh_strength_annual:.4f}")

    plt.show()

def check_feature_redundancy(X_df):    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i) 
        for i in range(X_df.shape[1])
    ]
    
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    
    for index, row in vif_data.iterrows():
        status = "safe"
        if row["VIF"] >= 10:
            status = "critical redundancy - drop or combine"
        elif row["VIF"] >= 5:
            status = "high multicollinearity"
            
        print(f"{row['Feature']:<20} | VIF: {row['VIF']:>6.2f} | {status}")
        
    return vif_data

def test_stationarity(series):
    result = adfuller(series.dropna()) # dropna is mandatory for differenced data
    
    print(f"ADF statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4f}")
    print(f"critical value (5%): {result[4]['5%']:.4f}")
    
    print("\n")

# ----- main -----

b_days = pd.date_range(start='2011-07-20', end='2026-05-31', freq='B') # truncated data to May as refinery data is only available up to May 2026

df = get_WTI_data()
df['DCOILWTICO'] = df['DCOILWTICO'].interpolate(method='time') # linear interpolation
#EDA(df, 'DCOILWTICO', cycle=261)

df_e1 = get_Brent_data()
df_e1['DCOILBRENTEU'] = df_e1['DCOILBRENTEU'].interpolate(method='time')
#EDA(df_e1, 'DCOILBRENTEU', cycle=261)

df_e2 = get_US_inv_no_SPR_data()
#EDA(df_e2, 'Weekly U.S. Ending Stocks excluding SPR of Crude Oil and Petroleum Products (Thousand Barrels)', cycle=52)

df_e3 = get_refinery_US_data()
#EDA(df_e3, 'U.S. Gross Inputs to Refineries (Thousand Barrels per Day)', cycle=12)

df_e4 = get_Cushing_inv_data()
#EDA(df_e4, 'Weekly Cushing OK Ending Stocks excluding SPR of Crude Oil (Thousand Barrels)', cycle=52)

# aligning data:
# capture target boundary thresholds
start_bound, end_bound = b_days.min(), b_days.max()

# reindex target data and check for stationarity 
df = df.reindex(b_days)
df = df['DCOILWTICO']

# test_stationarity(df)

# apply first-order differencing
# df_diff = df.diff()
# test_stationarity(df_diff)

# consolidate datasets into a dictionary
raw_exog_dict = {
    'Brent': df_e1,
    'US_inv_no_SPR': df_e2,
#    'refinery_US': df_e3,
    'Cushing_inv': df_e4
}

aligned_features = {}

for name, dataframe in raw_exog_dict.items():
    # force alignment to the time index
    aligned = dataframe.reindex(df.index)
    filled = aligned.ffill()

    # lag to eliminate lookahead bias and compensate for info processing delays, etc
    shifted = filled.shift(3)

    for col in shifted.columns:
        aligned_features[col] = shifted[col]

# merging into final array
X_matrix = pd.DataFrame(aligned_features)
final_dataset = pd.concat([df, X_matrix], axis=1).dropna()

# drop rows with NaN values in either df or X_matrix
combined_mask = df.notna() & X_matrix.notna().all(axis=1) 
df = df.loc[combined_mask]
X_matrix = X_matrix.loc[combined_mask]

scaler = StandardScaler() # normalization step
X_scaled_values = scaler.fit_transform(X_matrix)
X_matrix = pd.DataFrame(X_scaled_values, index=X_matrix.index, columns=X_matrix.columns)

# check_feature_redundancy(X_matrix)

# X_matrix represents all collected data, separate into training and testing sets
X_train = X_matrix.loc[:'2025-12-31']
X_test = X_matrix.loc['2026-01-01':]
df_train = df.loc[:'2025-12-31']
df_test = df.loc['2026-01-01':]

# determine the appropriate ARIMA orders (p, d, q) using ACF and PACF plots
# apply a first-difference for stationarity
# stationary_target = df_train.diff().dropna()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# plot_acf(stationary_target, lags=40, ax=ax1, alpha=0.05)
# ax1.set_title("Autocorrelation Function (ACF) - Identifies 'q'")
# ax1.set_xlabel("Lags")
# ax1.set_ylabel("Correlation Coefficient")
# ax1.grid(True, linestyle="--", alpha=0.5)

# plot_pacf(stationary_target, lags=40, ax=ax2, alpha=0.05, method="ywm")
# ax2.set_title("Partial Autocorrelation Function (PACF) - Identifies 'p'")
# ax2.set_xlabel("Lags")
# ax2.set_ylabel("Partial Correlation Coefficient")
# ax2.grid(True, linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.show()

# fitting to ARIMAX model
arimax_model = ARIMA(df_train, exog=X_train, order=(0, 1, 1))
arimax_results = arimax_model.fit(method="innovations_mle")
arimax_residuals = arimax_results.resid

# print(arimax_results.summary())

# skewed Student's t-distribution (for heavy kurtosis and -6.13 skew)
garch_skewt_model = arch_model(arimax_residuals, p=1, q=1, vol='GARCH', dist='skewt')
garch_skewt_result = garch_skewt_model.fit(disp='off')

print(garch_skewt_result.summary())

# conditional volatility (standard deviation scale)
# conditional_vol = garch_skewt_result.conditional_volatility

# plt.figure(figsize=(12, 5))
# plt.plot(conditional_vol, color='darkred', label='GARCH Conditional Volatility')
# plt.title('Time-Varying Volatility Clustering (ARIMAX-GARCH)')
# plt.xlabel('Observations')
# plt.ylabel('Volatility')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

test_window = 100
train_end_idx = len(df_train)

model_predictions = []
naive_predictions = []
actual_values = []


print("rolling backtest:")

for i in range(test_window):
    # split data dynamically (expanding window)
    current_train_y = df.iloc[train_end_idx + i:]
    current_train_exog = X_matrix.iloc[train_end_idx + i:]
    
    actual_val = df_test.iloc[i]
    actual_values.append(actual_val)
    
    # naive: next value is the last value
    naive_val = current_train_y.iloc[-1]
    naive_predictions.append(naive_val)
    
    future_exog = X_test.iloc[i]
    
    try:
        # fit ARIMAX(0,1,1) on the current training window
        model = ARIMA(endog=current_train_y, exog=current_train_exog, order=(0, 1, 1))
        model_fit = model.fit()
        
        # forecast 1 step ahead
        pred_val = model_fit.forecast(steps=1, exog=future_exog).values[0]
        model_predictions.append(pred_val)
    except Exception as e:
        # fallback if optimization fails to converge
        model_predictions.append(naive_val)

actuals = np.array(actual_values)
preds_model = np.array(model_predictions)
preds_naive = np.array(naive_predictions)

# error metrics
mae_model = mean_absolute_error(actuals, preds_model)
mae_naive = mean_absolute_error(actuals, preds_naive)

rmse_model = np.sqrt(root_mean_squared_error(actuals, preds_model))
rmse_naive = np.sqrt(root_mean_squared_error(actuals, preds_naive))

# Theil's U statistic (U < 1 beats the naive baseline)
theils_u = np.sqrt(np.sum((preds_model - actuals)**2) / np.sum((preds_naive - actuals)**2))

results_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE'],
    'ARIMAX Model': [mae_model, rmse_model],
    'Naive Baseline': [mae_naive, rmse_naive]
}).set_index('Metric')

print(results_df.round(4))
print(f"Theil's U statistic: {theils_u:.4f}")
