import numpy as np
import pandas as pd
import io
import sklearn
import datetime as dt
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
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import Holt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def get_WTI_data():
    df = pd.read_csv("DCOILWTICO.csv")
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.set_index('observation_date') # indexing is required for time-based interpolation
    return df 

def get_Brent_data():
    df = pd.read_csv("DCOILBRENTEU.csv")
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.set_index('observation_date') # indexing is required for time-based interpolation
    return df 

def get_refinery_US_data():
    df = pd.read_csv("U.S._Gross_Inputs_to_Refineries.csv")
    df['Month'] = pd.to_datetime(df['Month'], format = '%b %Y')
    df = df.set_index('Month') # indexing is required for time-based interpolation and reversing
    df = df.reindex(index=df.index[::-1]) 
    return df 

def get_Cushing_inv_data():
    df = pd.read_csv("Weekly_Cushing_OK_Ending_Stocks_excluding_SPR_of_Crude_Oil.csv")
    df['Week of'] = pd.to_datetime(df['Week of'])
    df = df.set_index('Week of') # indexing is required for time-based interpolation and reversing
    df = df.reindex(index=df.index[::-1])
    return df 

def get_US_inv_no_SPR_data():
    df = pd.read_csv("Weekly_U.S._Ending_Stocks_excluding_SPR_of_Crude_Oil_and_Petroleum_Products.csv")
    df['Week of'] = pd.to_datetime(df['Week of'])
    df = df.set_index('Week of') # indexing is required for time-based interpolation and reversing
    df = df.reindex(index=df.index[::-1]) 
    return df 

def get_US_inv_data():
    df = pd.read_csv("Weekly_U.S._Ending_Stocks_of_Crude_Oil_and_Petroleum_Products.csv")
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

# ----- main -----

b_days = pd.date_range(start='2011-07-20', end='2025-05-31', freq='B') # truncated data to May as refinery data is only available until May 2026

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

#        aligning data:
# capture target boundary thresholds
start_bound, end_bound = b_days.min(), b_days.max()

# reindex target data
df = df.reindex(b_days)
df = df['DCOILWTICO']

# consolidate raw datasets into a dictionary
raw_exog_dict = {
    'Brent': df_e1,
    'US_inv_no_SPR': df_e2,
#    'refinery_US': df_e3,
    'Cushing_inv': df_e4
}

aligned_features = {}

for name, dataframe in raw_exog_dict.items():
    # transform the raw data into percentage change to capture relative changes
    # dataframe = dataframe.pct_change().dropna()

    # force alignment to the time index
    aligned = dataframe.reindex(df.index)
    filled = aligned.ffill()

    # lag to compensate for information processing delays, etc. 
    shifted = filled.shift(3)
    
    # verify the feature isn't entirely filled with NaNs
    if shifted.dropna().empty:
        print(f"exogenous variable '{name}' has no data within your time frame - automatically skipping this feature")
        continue

    for col in shifted.columns:
        aligned_features[col] = shifted[col]

# merging into final array
X_matrix = pd.DataFrame(aligned_features)
final_dataset = pd.concat([df, X_matrix], axis=1).dropna()

# # create a rolling business-day counter (instead of slow s=261 seasonal order)
# years_series = pd.Series(X_matrix.index.year, index=X_matrix.index)
# business_day_of_year = years_series.groupby(years_series).cumcount()

# # Generate 3 orders of Fourier terms to capture complex seasonal waves
# for k in range(1, 4):
#     X_matrix[f'sin_seasonal_k{k}'] = np.sin(2 * np.pi * k * business_day_of_year / 261)
#     X_matrix[f'cos_seasonal_k{k}'] = np.cos(2 * np.pi * k * business_day_of_year / 261)

# # flag the last 2 business days and first 2 business days of every month
# is_month_end = (X_matrix.index + pd.offsets.BMonthEnd(0)) == X_matrix.index
# is_month_start = (X_matrix.index + pd.offsets.BMonthBegin(0)) == X_matrix.index

# X_matrix['month_turnover_signal'] = (is_month_end | is_month_start).astype(int)

# drop rows with NaN values in either df or X_matrix
combined_mask = df.notna() & X_matrix.notna().all(axis=1) 
df = df.loc[combined_mask]
X_matrix = X_matrix.loc[combined_mask]

scaler = StandardScaler() # normalization step
X_scaled_values = scaler.fit_transform(X_matrix)
X_matrix = pd.DataFrame(X_scaled_values, index=X_matrix.index, columns=X_matrix.columns)

#check_feature_redundancy(X_matrix)
#raise SystemExit

# X_matrix represents all collected data, separate into training and testing sets
X_train = X_matrix.loc[:'2024-12-31']
X_test = X_matrix.loc['2025-01-01':]
df_train = df.loc[:'2024-12-31']
df_test = df.loc['2025-01-01':]

# fitting to ARIMAX model
arimax_model = ARIMA(df_train, exog=X_train, order=(0, 1, 1))
arimax_results = arimax_model.fit(method="innovations_mle")

print(arimax_results.summary())

# compare ARIMA(1,0,0) and ARIMA(1,1,0) models on df to baseline and ARIMAX results
{
# history_y_train = df_train.copy()
# df_test_series = df_test.copy()

# preds_ar_100 = []
# preds_ar_110 = []

# for i in range(len(df_test_series)):
#     current_true_y = df_test_series.iloc[i]
#     current_date = df_test_series.index[i]
    
#     model_100 = ARIMA(history_y_train, order=(1, 0, 0))
#     fit_100 = model_100.fit()
#     pred_100 = fit_100.forecast(steps=1)
#     preds_ar_100.append(pred_100.iloc[0])
    
#     model_110 = ARIMA(history_y_train, order=(1, 1, 0))
#     fit_110 = model_110.fit()
#     pred_110 = fit_110.forecast(steps=1)
#     preds_ar_110.append(pred_110.iloc[0])
    
#     new_row = pd.Series([current_true_y], index=[current_date])
#     history_y_train = pd.concat([history_y_train, new_row])
#     history_y_train.index.freq = 'B'

# # calculate evaluation metrics
# mae_100 = mean_absolute_error(df_test_series, preds_ar_100)
# mae_110 = mean_absolute_error(df_test_series, preds_ar_110)

# print(f"1. Your Previous ARX(1) MAE:      1.0369")
# print(f"2. Simple Naive Model MAE:        1.0113")
# print(f"3. Pure AR(1, 0, 0) Level MAE:   {mae_100:.4f}")
# print(f"4. Pure AR(1, 1, 0) Diff MAE:    {mae_110:.4f}")

# raise SystemExit
}

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
# raise SystemExit

# backtesting for 2025
history_y = df_train.copy()
history_X = X_train.copy()
predictions = []

for i in range(len(df_test)):
    # get the current step's target and exogenous variables
    current_y = df_test.iloc[i]
    current_X = X_test.iloc[[i]]
    
    # refit the model with updated history
    model = ARIMA(history_y, exog=history_X, order=(0, 1, 1))
    model_fit = model.fit(method="innovations_mle")
    
    # predict 1 step ahead 
    pred = model_fit.forecast(steps=1, exog=current_X)
    predictions.append(pred.iloc[0])
    
    # append observed values to history
    history_y = pd.concat([history_y, pd.Series([current_y], index=[df_test.index[i]])])
    history_X = pd.concat([history_X, current_X])

# convert predictions to series
y_pred = pd.Series(predictions, index=df_test.index)

# evaluate metrics
mae = mean_absolute_error(df_test, y_pred)
rmse = root_mean_squared_error(df_test, y_pred)

# naive baseline: predict today's value as tomorrow's value
naive_preds = df_test.shift(1).bfill()
naive_mae = mean_absolute_error(df_test, naive_preds)

print(f"naive model MAE: {naive_mae:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
