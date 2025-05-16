from dataclasses import dataclass
import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import (
    check_seasonality, plot_acf, plot_pacf, remove_seasonality,remove_trend,
    stationarity_test_adf
)
from darts.utils.model_selection import train_test_split

import data_utils as du
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import plotters
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



@dataclass
class ARIMAWrapper:
    """
    Wrapper class for ARIMA

    Parameters
    ----------
    ts : pd.DataFrame | TimeSeries
        Input time series
    split_point : datetime | float
        Split point for train and test
    start_date_slice : datetime | None, optional
        Start date to slice the time series, by default None
    end_date_slice : datetime | None, optional
        End date to slice the time series, by default None
    """

    # public
    ts: pd.DataFrame | TimeSeries
    split_point: datetime | float
    start_date_slice: datetime | None = None
    end_date_slice: datetime | None = None

    # private
    ts_train: pd.DataFrame | TimeSeries | None = None
    ts_test: pd.DataFrame | TimeSeries | None = None
    arima_model: ARIMA | None = None
    arima_model_fit = None
    forecasted_series: pd.Series | None = None
    order: tuple[int, int, int] | None = None

    def __post_init__(self):
        if self.start_date_slice is not None:
            self.ts = self.ts.loc[self.ts.index >= self.start_date_slice]
        if self.end_date_slice is not None:
            self.ts = self.ts.loc[self.ts.index <= self.end_date_slice]
        self.ts = self.ts.sort_index()

        print("Stationarity test for whole series with 0 lag:")
        self.check_stationarity()

        print("Train-Test split at", self.split_point)
        self.train_test_split()

        print("Stationarity test for train series with 0 lag:")
        self.check_stationarity(self.ts_train)

    def check_stationarity(self, df: pd.DataFrame | None = None) -> None:
        if df is None:
            df = self.ts
        adf_result = adfuller(df)
        if adf_result[1] <= 0.05:
            print("Reject the null hypothesis. The series is likely stationary.\n")
        else:
            print("Fail to reject the null hypothesis. The series is likely non-stationary.\n")

    def plot_acf_pacf(self, df: pd.DataFrame | None = None) -> None:
        if df is None:
            df = self.ts
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(df, lags=40, ax=plt.gca())
        plt.subplot(122)
        plot_pacf(df, lags=40, ax=plt.gca())
        plt.show()
    
    def train_test_split(self) -> None:
        if isinstance(self.split_point, float):
            print("Using darts train_test_split function")
            self.ts_train, self.ts_test = train_test_split(self.ts, train_size=self.split_point)
        self.ts_train, self.ts_test = du.ts_train_test_split(self.ts, self.split_point)

    def get_best_ARIMA_params(
                                self,
                                df: pd.DataFrame | None = None
                            ) -> tuple[int, int, int, ARIMA]:
        if df is None:
            df = self.ts
        best_aic = float('inf')
        best_order = None
        best_model = None

        # Loop over a range of p, d, q values
        for p in range(0, 6):           # Testing p values from 0 to 5
            for d in range(0, 3):       # Testing d values from 0 to 2
                for q in range(0, 6):   # Testing q values from 0 to 5
                    try:
                        model = ARIMA(self.ts_train, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except:   # this needs to continue if mnodel cannot converge
                        continue

        print("--------------------------------")
        print(f'Best ARIMA order: {best_order} selected')
        self.arima_model = best_model
        self.order = best_order
        return self.order
    
    def run_ARIMA(self, p, d, q) -> ARIMA:
        if self.ts_train is None:
            raise ValueError("ts_train is empty. Run train_test_split first.")
        
        print("--------------------------------")
        print(f'Running ARIMA({p}, {d}, {q}) on train set')
        self.arima_model = ARIMA(self.ts_train, order=(p, d, q))
        self.arima_model_fit = self.arima_model.fit()
        self.order = (p, d, q)
        print(self.arima_model_fit.summary())

    def get_forecast(self, n_pred: int | None = None) -> pd.Series:
        if self.ts_test is None:
            raise ValueError("ts_test is empty. Run train_test_split first.")
        
        if self.arima_model is None:
            print("self.arima_model model not found. Auto run best ARIMA.")
            p, d, q, self.arima_model = self.get_best_ARIMA_params()
            print(f"Selected ARIMA({p}, {d}, {q})")
        self.arima_forecast_fit = self.arima_model.fit()
        
        if n_pred is None:
            n_pred = len(self.ts_test)
        
        print("--------------------------------")
        print(f'Forecasting {n_pred} steps ahead')
        self.forecasted_series = self.arima_model_fit.forecast(steps=n_pred)
        return self.forecasted_series

    def plot_forecast(self):
        arima_pred_all = self.ts[['price']]
        arima_forecast = self.forecasted_series
        arima_forecast.name = 'price_prediction'
        arima_forecast.to_frame()
        arima_pred_all = arima_pred_all.join(arima_forecast)
        plotters.plotly_actual_predict(
                arima_pred_all,
                'price',
                'price_prediction',
                f'Day-ahead Price [EUR/MWh] Predicted with ARIMA{self.order}',
                )
        
    def get_mae_rmse(self):
        self.mae = mean_absolute_error(self.ts_test, self.forecasted_series)
        self.mse = mean_squared_error(self.ts_test, self.forecasted_series)
        self.rmse = np.sqrt(self.mse)
        print(f"RSME for ARIMA{self.order}: {self.rmse}")
        print(f"MAE for ARIMA{self.order}: {self.mae}")
