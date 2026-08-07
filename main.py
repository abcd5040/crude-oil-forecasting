import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io 
import datetime as dt # date and time manipulation
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization
import matplotlib.dates as mdates  # date formatting
import seaborn as sns             # statistical data visualization
import statsmodels.api as sm     # statistical modeling
from statsmodels.tsa.seasonal import seasonal_decompose # time series decomposition
from statsmodels.tsa.seasonal import STL

def get_WTI_data():
    df_o = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\DCOILWTICO.csv")
    df_o['observation_date'] = pd.to_datetime(df_o['observation_date'])
    df_o = df_o.set_index('observation_date') # indexing is required for time-based interpolation
    return df_o 

def get_Brent_data():
    df_o = pd.read_csv("C:\\Users\\Joshua Liang\\Desktop\\crude oil forecasting\\DCOILBRENTEU.csv")
    df_o['observation_date'] = pd.to_datetime(df_o['observation_date'])
    df_o = df_o.set_index('observation_date') # indexing is required for time-based interpolation
    return df_o 

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


def EDA(df, y):
    is_na = df[y].isna() # cumsum method for missing data analysis
    gap_groups = (~is_na).cumsum()
    gap_sizes = is_na[is_na].groupby(gap_groups).size() 
    gap_frequencies = gap_sizes.value_counts().sort_index() 
    print(gap_frequencies)

    missing_percentage = df[y].isna().mean() * 100
    print(f"Missing data: {missing_percentage:.2f}%")

    df[y] = df[y].interpolate(method='time') # linear interpolation


    # seasonal decomposition
    result = seasonal_decompose(df[y], model='additive', period=261) # ~261 business days/year
    result.plot()

    fh_strength_annual = get_seasonal_strength(df[y], period=261)
    print(f"Annual Seasonal Strength: {fh_strength_annual:.4f}")

    plt.show()

# ----- main -----

df_o = get_Brent_data()
# print(df_o.head()) # test for data retrieval

EDA(df_o, 'DCOILBRENTEU')