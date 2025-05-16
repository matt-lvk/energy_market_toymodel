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
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from abc import ABC, abstractmethod
from prophet import Prophet


@dataclass
class ModelWrapper(ABC):
    """
    Parent class for model wrapper
    """
    # public
    ts: pd.DataFrame | TimeSeries
    split_point: datetime | float
    start_date_slice: datetime | None = None
    end_date_slice: datetime | None = None

    def df_slicer(self):
        if self.start_date_slice is not None:
            self.ts = self.ts.loc[self.ts.index >= self.start_date_slice]
        if self.end_date_slice is not None:
            self.ts = self.ts.loc[self.ts.index <= self.end_date_slice]
        self.ts = self.ts.sort_index()

    def train_test_split(self) -> tuple[pd.DataFrame | TimeSeries, pd.DataFrame | TimeSeries]:
        if isinstance(self.split_point, float):
            print("Using darts train_test_split function")
            return train_test_split(self.ts, train_size=self.split_point)
        return du.ts_train_test_split(self.ts, self.split_point)

    @abstractmethod
    def get_forecast(self):
        pass

    @abstractmethod
    def get_rmse_mae(self):
        pass

@dataclass
class ARIMAWrapper(ModelWrapper):
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
    # private
    ts_train: pd.DataFrame | TimeSeries | None = None
    ts_test: pd.DataFrame | TimeSeries | None = None
    arima_model: ARIMA | None = None
    arima_model_fit = None
    forecasted_series: pd.Series | None = None
    order: tuple[int, int, int] | None = None

    def __post_init__(self):
        self.df_slicer()

        print("Stationarity test for whole series with 0 lag:")
        self.check_stationarity()

        print("Train-Test split at", self.split_point)
        self.ts_train, self.ts_test = self.train_test_split()

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
    
    def run_ARIMA(self, p, d, q, method: str | None = None) -> ARIMA:
        if self.ts_train is None:
            raise ValueError("ts_train is empty. Run train_test_split first.")
        
        print("--------------------------------")
        print(f'Running ARIMA({p}, {d}, {q}) on train set')
        self.arima_model = ARIMA(self.ts_train, order=(p, d, q))
        self.arima_model_fit = self.arima_model.fit(method=method)
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
        arima_pred_all.sort_index(inplace=True)

        plotters.plotly_actual_predict(
                arima_pred_all,
                'price',
                'price_prediction',
                f'Day-ahead Price [EUR/MWh] Predicted with ARIMA{self.order}',
                )
        
    def get_rmse_mae(self) -> tuple[float, float]:
        rmse = np.sqrt(mean_squared_error(self.ts_test, self.forecasted_series))
        mae = mean_absolute_error(self.ts_test, self.forecasted_series)

        rmse = round(rmse, 3)
        mae = round(mae, 3)

        print(f"RSME for ARIMA{self.order}: {round(rmse, 4)}")
        print(f"MAE for ARIMA{self.order}: {round(mae, 4)}")
        return rmse, mae


@dataclass
class XGBWrapper(ModelWrapper):
    # private
    xbg_train: pd.DataFrame | None = None
    xbg_test: pd.DataFrame | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.DataFrame | None = None
    y_test: pd.DataFrame | None = None
    model: xgb.XGBRegressor | None = None
    forecasted_array: np.ndarray | None = None
    
    def __post_init__(self):
        self.df_slicer()

        print("Train-Test split at", self.split_point)
        self.xgb_train, self.xgb_test = self.train_test_split()

        self.X_train, self.y_train = self.xgb_train.drop(['price', 'datetime'], axis=1), self.xgb_train['price']
        self.X_test, self.y_test = self.xgb_test.drop(['price', 'datetime'], axis=1), self.xgb_test['price']

    def add_lagged_MA_price(self, n: int):
        if n > 1:
            print(f"Adding 1-lagged hour of price to X_train and X_test")
            self.X_train['mean_previous_1_hrs'] = self.y_train.shift(1)
            self.X_test['mean_previous_1_hrs'] = self.y_test.shift(1)

        for i in range(2, n + 1):
            print(f"Adding mean of previous {i} hours of price to X_train and X_test")
            self.X_train[f'mean_previous_{i}_hrs'] = self.y_train.shift(4)\
                                    .rolling(window=f'{i}H', min_periods=1).mean()
            self.X_test[f'mean_previous_{i}_hrs'] = self.y_test.shift(4)\
                                    .rolling(window=f'{i}H', min_periods=1).mean()
            
    def run_xgb(self):
        self.model = xgb.XGBRegressor()
        self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                verbose=False
                )
        print("XGBoost model fitted.")

    @property
    def plot_importance(self):
        
        if self.model is None:
            raise ValueError("self.model is empty. Run run_xgb first.")
        plot_importance(self.model)
        plt.show()

    def get_forecast(self) -> np.ndarray:
        self.forecasted_array = self.model.predict(self.X_test)
        return self.forecasted_array
    
    def plot_forecast(self):
        if self.forecasted_array is None:
            raise ValueError("self.forecasted_array is empty. Run get_forecast first.")
        
        test = self.xgb_test.copy()
        test['price_prediction'] = self.forecasted_array
        predicted_XGBoost = pd.concat([test, self.xgb_train], sort=False)
        predicted_XGBoost.sort_index(inplace=True)

        plotters.plotly_actual_predict(
                predicted_XGBoost,
                'price',
                'price_prediction',
                'Day-ahead Price [EUR/MWh] Predicted with XGBoost',
                )
    
    def get_rmse_mae(self) -> tuple[float, float]:
        test = self.xgb_test.copy()
        test['price_prediction'] = self.forecasted_array

        rmse = np.sqrt(mean_squared_error(y_true=self.xgb_test['price'],
                    y_pred=test['price_prediction']))
        mae = mean_absolute_error(y_true=self.xgb_test['price'],
                    y_pred=test['price_prediction'])
        print(f"RSME for XGBoost: {round(rmse, 3)}")
        print(f"MAE for XGBoost: {round(mae, 3)}")
        return rmse, mae

    @property
    def get_forecast_df(self) -> pd.Series:
        forecast = self.xgb_test.copy()
        forecast['price_prediction'] = self.forecasted_array
        forecast['datetime'] = self.xgb_test.index
        forecast.sort_index(inplace=True)
        return forecast


@dataclass
class ProphetWrapper(ModelWrapper):
    # private
    prophet_train: pd.DataFrame | None = None
    prophet_test: pd.DataFrame | None = None
    prophet_model: xgb.XGBRegressor | None = None
    forecasted_df: pd.DataFrame | None = None

    def __post_init__(self):
        self.df_slicer()

        print("Train-Test split at", self.split_point)
        train, test = self.train_test_split()

        # Adjust index for prophet"
        self.ts = self.index_to_column(self.ts, 'price')
        self.prophet_train = self.index_to_column(train, 'price')
        self.prophet_test = self.index_to_column(test, 'price')

    def index_to_column(
                    self,
                    df: pd.DataFrame,
                    target_variable: str
                    ) -> pd.DataFrame:
        data = df.copy()
        data['Datetime'] = pd.to_datetime(data.index)
        data.reset_index(drop=True, inplace=True)
        data = data.sort_values('Datetime')
        
        data = data.rename(columns={'Datetime': 'ds', target_variable: 'y'})
        return data
    
    def run_prophet(self) -> None:
        self.prophet_model = Prophet(interval_width=0.95)

        self.prophet_model.fit(self.prophet_train)
        print("Prophet model fitted. Use get_forecast() to get forecast.")
    
    def get_forecast(self) -> pd.DataFrame:
        self.forecasted_df = self.prophet_model.predict(
                        self.prophet_test[['ds']]
                        )  # Keep the dataset format
        return self.forecasted_df
    
    def plot_forecast(self) -> None:
        df = self.ts[['ds', 'y']]
        df = df.merge(self.forecasted_df, on='ds', how='left')
        df.set_index('ds', inplace=True)

        plotters.plotly_actual_predict(
                df,
                'y',
                'yhat',
                'Day-ahead Price [EUR/MWh] Predicted with Prophet',
                )
        
    def get_rmse_mae(self) -> tuple[float, float]:
        prophet_mae = round(mean_absolute_error(self.prophet_test['y'],
                                            self.forecasted_df['yhat']), 3)
        
        prophet_rsme = round(np.sqrt(mean_squared_error(self.prophet_test['y'],
                                                    self.forecasted_df['yhat'])), 3)
        
        print(f"RSME for Prophet: {round(prophet_rsme, 3)}")
        print(f"MAE for Prophet: {round(prophet_mae, 3)}")
        return prophet_rsme, prophet_mae