# Inbuilt
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import plot_importance  # ,plot_tree

# TS Lib
from darts import TimeSeries
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from scipy.linalg import LinAlgError

# ML
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Custom Utilities
import plotters
import data_utils as du


seed_value = 11


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
            # return train_test_split(self.ts, train_size=self.split_point)
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
    A wrapper for the ARIMA model, providing functionality to preprocess time series data,
    determine stationarity, plot ACF and PACF graphs, determine the best ARIMA parameters,
    run the ARIMA model, forecast future values, and evaluate model performance.

    Attributes
    ----------
    ts : pd.DataFrame | TimeSeries
        The time series data as a dataframe or TimeSeries object.
    split_point : datetime | float
        The point at which the data is split into training and testing sets.
        If it's a float, it represents the proportion of the dataset to include in the train split.
    start_date_slice : datetime, optional
        Optional starting date for slicing the time series data.
    end_date_slice : datetime, optional
        Optional ending date for slicing the time series data.    

    Methods
    -------
    plot_acf_pacf(df=None)
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the time series.
    get_best_ARIMA_params(df=None)
        Determines the best ARIMA parameters based on the lowest Akaike Information Criterion (AIC).
    fit_model(p, d, q, method=None)
        Runs the ARIMA model on the training set with provided order and fitting method.
    get_forecast(n_pred=None)
        Forecasts future values of the time series based on the fitted ARIMA model.
    plot_forecast()
        Plots the forecasted values against the actual values in the time series.
    get_rmse_mae()
        Calculates the Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) of the forecasted values.

    Notes
    -----
    The class assumes that the input time series includes a 'price' column that is used as the target variable.
    The `ts` attribute from the parent class is used as the input time series data.
    """
    # private
    ts_train: pd.DataFrame | TimeSeries | None = None
    ts_test: pd.DataFrame | TimeSeries | None = None
    model: ARIMA | None = None
    model_fit: ARIMAResults | None = None
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
        """
        Find the best ARIMA parameters for the given time series. This method is 
        done by looping (p,d,q) at max 6, 3, 6 to find the lowest AIC (which
        presents information loss)
        
        Parameters
        ----------
        df : pd.DataFrame | None, optional
            Time series data, by default None

        Returns
        -------
        tuple[int, int, int, ARIMA]
            The parameters (p, d, q) and the best model
        """
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
                    except LinAlgError:   # this needs to continue if mnodel cannot converge
                        continue

        print("--------------------------------")
        print(f'Best ARIMA order: {best_order} selected')
        self.arima_model = best_model
        self.order = best_order
        return self.order
    
    def fit_model(self, p, d, q, method: str | None = None) -> None:
        if self.ts_train is None:
            raise ValueError("ts_train is empty. Run train_test_split first.")
        
        print("--------------------------------")
        print(f'Running ARIMA({p}, {d}, {q}) on train set')
        self.model = ARIMA(self.ts_train, order=(p, d, q))
        self.model_fit = self.model.fit(method=method)
        self.order = (p, d, q)
        print(self.model_fit.summary())

    def get_forecast(self, n_pred: int | None = None) -> pd.Series:
        if self.ts_test is None:
            raise ValueError("ts_test is empty. Run train_test_split first.")
        
        if self.model is None:
            print("self.arima_model model not found. Auto run best ARIMA.")
            p, d, q, self.model = self.get_best_ARIMA_params()
            print(f"Selected ARIMA({p}, {d}, {q})")
        self.model_fit = self.model.fit()
        
        if n_pred is None:
            n_pred = len(self.ts_test)
        
        print("--------------------------------")
        print(f'Forecasting {n_pred} steps ahead')
        self.forecasted_series = self.model_fit.forecast(steps=n_pred)
        return self.forecasted_series

    def plot_forecast(
            self,
            title: str | None = None
            ) -> None:
        if not title:
            title = f'Day-ahead Price [EUR/MWh] Predicted with ARIMA{self.order}'
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
                title,
                self.split_point,
                self.start_date_slice,
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
    """
    A wrapper for the XGBoost regression model specifically tailored for time series forecasting.

    Attributes
    ----------
    ts : pd.DataFrame | TimeSeries
        The time series data as a dataframe or TimeSeries object.
    split_point : datetime | float
        The point at which the data is split into training and testing sets.
        If it's a float, it represents the proportion of the dataset to include in the train split.
    start_date_slice : datetime, optional
        Optional starting date for slicing the time series data.
    end_date_slice : datetime, optional
        Optional ending date for slicing the time series data.    

    Methods
    -------
    add_lagged_MA_price(hours=None, days=None)
        Adds lagged moving average price as a new feature to the training and testing sets.
    run_xgb()
        Initializes and fits the XGBoost model on the training data.
    plot_importance
        A property that plots the feature importance of the fitted XGBoost model.
    get_forecast() -> np.ndarray
        Predicts the target values using the fitted XGBoost model and returns the forecasted numpy array.
    plot_forecast()
        Plots the actual versus predicted prices using the forecasted data.
    get_rmse_mae() -> tuple[float, float]
        Calculates and returns the Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) of the predictions.
    get_forecast_df() -> pd.Series
        A property that returns the forecasted values as a pandas Series with the datetime index.
    """
    # private
    xbg_train: pd.DataFrame | None = None
    xbg_test: pd.DataFrame | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.DataFrame | None = None
    y_test: pd.DataFrame | None = None
    model: xgb.XGBRegressor | None = None
    forecasted_array: np.ndarray | None = None
    predicted_XGBoost: pd.DataFrame | None = None
    n_estimators: int = None

    def __post_init__(self):
        self.df_slicer()

        print("Train-Test split at", self.split_point)
        self.xgb_train, self.xgb_test = self.train_test_split()

        self.X_train, self.y_train = self.xgb_train.drop(['price', 'datetime'], axis=1), self.xgb_train['price']
        self.X_test, self.y_test = self.xgb_test.drop(['price', 'datetime'], axis=1), self.xgb_test['price']

    def add_lagged_MA_price(self,
                            hours: int | list,
                            days: int | None = None,
                            rolling_mean: bool = False
                            ) -> None:
        
        """
        Add n-lagged moving average price to X_train and X_test.

        Parameters
        ----------
        hours : int or list
            Number of hours to lag. If not specified, days is used.
        days : int, optional
            Number of days to lag. If not specified, hours is used.

        Notes
        -----
        Only one of hours or days can be specified. If both are specified,
        an Exception is raised.

        For example, if hours=3, the mean of the previous 3 hours of price will
        be added to X_train and X_test. If days=1, the mean of the previous 24
        hours of price will be added to X_train and X_test.
        """
        periods = []

        if isinstance(hours, list):
            periods.extend(hours)
        elif isinstance(hours, int):
            periods.append(hours)

        if days is not None:
            periods.extend([i * 24 for i in range(1, days + 1)])

        print("Adding n-lagged hour of price to X_train and X_test")
        # self.X_train['mean_previous_1_hrs'] = self.y_train.shift(1)
        # self.X_test['mean_previous_1_hrs'] = self.y_test.shift(1)

        for i in periods:
            if rolling_mean:
                print(f"Adding mean of previous {i} hours of price to X_train and X_test")
                self.X_train[f'mean_previous_{i}_hrs'] = self.y_train.shift(i)\
                                        .rolling(window=f'{i}H', min_periods=1).mean()
                self.X_test[f'mean_previous_{i}_hrs'] = self.y_test.shift(i)\
                                        .rolling(window=f'{i}H', min_periods=1).mean()
            print(f"Adding mean of previous {i} hours of price to X_train and X_test [Rolling Mean]")
            self.X_train[f'mean_previous_{i}_hrs'] = self.y_train.shift(i)\
                        .rolling(window=f'{i}H', min_periods=1).mean()
            self.X_test[f'mean_previous_{i}_hrs'] = self.y_test.shift(i)\
                                        .rolling(window=f'{i}H', min_periods=1).mean()
        
    def run_xgb(self):
        self.model = xgb.XGBRegressor(
                                    seed=seed_value,
                                    n_estimators=self.n_estimators
                                    )
        print(f"XGBoost model initialized with seed {seed_value} and {self.n_estimators} trees.")

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
        self.predicted_XGBoost = pd.concat([test, self.xgb_train], sort=False)
        self.predicted_XGBoost.sort_index(inplace=True)

        plotters.plotly_actual_predict(
                self.predicted_XGBoost,
                'price',
                'price_prediction',
                'Day-ahead Price [EUR/MWh] Predicted with XGBoost',
                self.split_point,
                self.start_date_slice,
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
    """
    A wrapper class for the Prophet forecasting model, facilitating the preprocessing
    of time series data, training of the Prophet model, and generation of forecasts.

    Attributes
    ----------
    ts : pd.DataFrame | TimeSeries
        The time series data as a dataframe or TimeSeries object.
    split_point : datetime | float
        The point at which the data is split into training and testing sets.
        If it's a float, it represents the proportion of the dataset to include in the train split.
    start_date_slice : datetime, optional
        Optional starting date for slicing the time series data.
    end_date_slice : datetime, optional
        Optional ending date for slicing the time series data.

    Methods
    -------
    index_to_column(df, target_variable)
        Converts a dataframe index to a column and renames it for Prophet compatibility.
    run_prophet()
        Trains the Prophet model using the training dataset.
    get_forecast()
        Generates a forecast using the trained Prophet model and the test dataset.
    plot_forecast()
        Plots the actual versus predicted values using the forecast data.
    get_rmse_mae() -> tuple[float, float]
        Calculates and returns the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
        of the forecast.
    """
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
        """
        Adjust the dataframe to be used in Prophet. Namely, it renames datatime 
        index into 'ds' and target variable into 'y'

        Parameters
        ----------
        df : pd.DataFrame
            Time series data
        target_variable : str
            column name of the target variable

        Returns
        -------
        pd.DataFrame
            Dataframe with adjusted index for Prophet
        """
        data = df.copy()
        data['Datetime'] = pd.to_datetime(data.index)
        data.reset_index(drop=True, inplace=True)
        data = data.sort_values('Datetime')
        
        data = data.rename(columns={'Datetime': 'ds', target_variable: 'y'})
        return data
    
    def run_prophet(self) -> None:
        # interval_width=0.95 produce a prediction interval that is
        # designed to contain the true future value 95% of the time
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
                self.split_point,
                self.start_date_slice,
                )
                
    def get_rmse_mae(self) -> tuple[float, float]:
        prophet_mae = round(mean_absolute_error(self.prophet_test['y'],
                                            self.forecasted_df['yhat']), 3)
        
        prophet_rsme = round(np.sqrt(mean_squared_error(self.prophet_test['y'],
                                                    self.forecasted_df['yhat'])), 3)
        
        print(f"RSME for Prophet: {round(prophet_rsme, 3)}")
        print(f"MAE for Prophet: {round(prophet_mae, 3)}")
        return prophet_rsme, prophet_mae


@dataclass
class SARIMAXWrapper(ARIMAWrapper):
    """
    A wrapper for the SARIMAX model (inherenting from ARIMAWrapper),
    providing functionality to preprocess time series data,
    determine stationarity, plot ACF and PACF graphs, determine the best SARIMAX
    parameters, run the ARIMA model, forecast future values, and
    evaluate model performance.

    Attributes
    ----------
    ts : pd.DataFrame | TimeSeries
        The time series data as a dataframe or TimeSeries object.
    split_point : datetime | float
        The point at which the data is split into training and testing sets.
        If it's a float, it represents the proportion of the dataset to include in the train split.
    start_date_slice : datetime, optional
        Optional starting date for slicing the time series data.
    end_date_slice : datetime, optional
        Optional ending date for slicing the time series data.

    Methods
    -------
    plot_acf_pacf(df=None)
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the time series.
    get_best_ARIMA_params(df=None)
        Determines the best ARIMA parameters based on the lowest Akaike Information Criterion (AIC).
    fit_model(order:tuple, seasonal_order:tuple)
        Runs the ARIMA model on the training set with provided order and fitting method.
    get_forecast(n_pred=None)
        Forecasts future values of the time series based on the fitted ARIMA model.
    plot_forecast()
        Plots the forecasted values against the actual values in the time series.
    get_rmse_mae()
        Calculates the Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) of the forecasted values.

    Notes
    -----
    The class assumes that the input time series includes a 'price' column that is used as the target variable.
    The `ts` attribute from the parent class is used as the input time series data.
    """
    # Attribute
    seasonal_order: tuple[int, int, int, int] | None = None
    exog: pd.DataFrame | None = None

    # Friend
    model: SARIMAX | None = None
    model_fit: SARIMAXResults | None = None
    
    def fit_model(
                self,
                order: tuple[int, int, int] | None = None,
                ) -> None:
        if order:
            self.order = order
        if self.ts_train is None:
            raise ValueError("ts_train is empty. Run train_test_split first.")
        
        print("--------------------------------")
        print(f'Running SARIMAX({self.order}) on train set')
        self.model = SARIMAX(
                                endog=self.ts_train,
                                enxog=self.exog,
                                order=self.order,
                                seasonal_order=self.seasonal_order,
                                )
        self.model_fit = self.model.fit()
        print(self.model_fit.summary())

    def get_forecast(self, n_pred: int | None = None) -> pd.Series:
        if self.model is None:
            print("self.arima_model model not found. Auto run best SARIMAX(p,d,q).")
            self.fit_model()
            print(f"Selected ARIMA({self.order})")
        self.model_fit = self.model.fit()
        
        if n_pred is None:
            n_pred = len(self.ts_test)
        
        print("--------------------------------")
        print(f'Forecasting {n_pred} steps ahead')
        self.forecasted_series = self.model_fit.forecast(steps=n_pred)
        return self.forecasted_series
    
    def get_rmse_mae(self) -> tuple[float, float]:
        rmse = np.sqrt(mean_squared_error(self.ts_test, self.forecasted_series))
        mae = mean_absolute_error(self.ts_test, self.forecasted_series)

        rmse = round(rmse, 3)
        mae = round(mae, 3)

        print(f"RSME for SARIMAX{self.order},{self.seasonal_order}: {round(rmse, 4)}")
        print(f"MAE for SARIMAX{self.order},{self.seasonal_order}: {round(mae, 4)}")
        return rmse, mae