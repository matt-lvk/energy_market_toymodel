from datetime import datetime, timedelta

# Data manipulation
import numpy as np
import pandas as pd
from forecast_utils import ARIMAWrapper, XGBWrapper, ProphetWrapper
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BacktestParent(ABC):
    """parent class for backtesting"""
    cleaned_merged_data: pd.DataFrame
    short_split_date: datetime
    n_backward: int = 60
    forecast_window: int = 24
    forecast_metric: dict[str, list] | None = None

    @abstractmethod
    def get_windowed_rsme_mae():
        pass


@dataclass
class BacktestXGB(BacktestParent):
    """
    A class that extends BacktestParent to perform backtesting using the XGBoost model.

    This class uses the XGBWrapper to create, train, and evaluate an XGBoost model
    within a specified backtesting time window.

    Attributes
    ----------
    cleaned_merged_data : pd.DataFrame
        The preprocessed data set to be used for backtesting.
    short_split_date : datetime
        The date at which to split the data into training and testing sets.
    n_backward : int, default 60
        The number of days to look back from the `short_split_date` to create the training set.
    forecast_window : int, default 24
        The size of the window, in hours, for which to forecast ahead from the `short_split_date`.
    forecast_metric : dict[str, list] | None, optional
        A dictionary to store forecast metrics, initialized to None by default.
    
    Methods
    -------
    __post_init__()
        Initializes the backtesting process by setting up the XGBWrapper with the
        appropriate data slices and training the model.
    get_windowed_rsme_mae() -> tuple[float, float]
        Retrieves the RSME and MAE for the forecasted window from the XGBWrapper.
    get_windowed_forecast() -> pd.DataFrame
        Retrieves the forecasted DataFrame with actual and predicted values for the window.
    """
    short_xgb_obj: XGBWrapper | None = None

    def __post_init__(self):
        self.forecast_metric = {}

        start_slice = self.short_split_date - timedelta(days=self.n_backward)
        end_slice = self.short_split_date + timedelta(hours=self.forecast_window)
        print("Running XGBoost: starting at", start_slice)
        print(f"ending at {end_slice}")
        
        self.short_xgb_obj = XGBWrapper(
                        self.cleaned_merged_data.copy(),
                        self.short_split_date,
                        start_date_slice=start_slice,
                        end_date_slice=end_slice
                        )
        
        self.short_xgb_obj.add_lagged_MA_price(hours=4, days=3)
        self.short_xgb_obj.run_xgb()
        self.short_xgb_obj.get_forecast()

    def get_windowed_rsme_mae(self) -> tuple[float, float]:
        rmse, mae = self.short_xgb_obj.get_rmse_mae()
        return rmse, mae
    
    def get_windowed_forecast(self) -> pd.DataFrame:
        return self.short_xgb_obj.get_forecast_df
    
    def plot_forecast(self):
        self.short_xgb_obj.plot_forecast()
    

@dataclass
class BacktestProphet(BacktestParent):
    """
    A class that extends BacktestParent to perform backtesting using the Prophet model.

    This class uses the ProphetWrapper to create, train, and evaluate an Prophet model
    within a specified backtesting time window.

    Attributes
    ----------
    cleaned_merged_data : pd.DataFrame
        The preprocessed data set to be used for backtesting.
    short_split_date : datetime
        The date at which to split the data into training and testing sets.
    n_backward : int, default 60
        The number of days to look back from the `short_split_date` to create the training set.
    forecast_window : int, default 24
        The size of the window, in hours, for which to forecast ahead from the `short_split_date`.
    forecast_metric : dict[str, list] | None, optional
        A dictionary to store forecast metrics, initialized to None by default.
    
    Methods
    -------
    __post_init__()
        Initializes the backtesting process by setting up the XGBWrapper with the
        appropriate data slices and training the model.
    get_windowed_rsme_mae() -> tuple[float, float]
        Retrieves the RSME and MAE for the forecasted window from the XGBWrapper.
    get_windowed_forecast() -> pd.DataFrame
        Retrieves the forecasted DataFrame with actual and predicted values for the window.
    """
    prophet_model_obj: ProphetWrapper | None = None

    def __post_init__(self):
        start_slice = self.short_split_date - timedelta(days=self.n_backward)
        end_slice = self.short_split_date + timedelta(hours=self.forecast_window)
        print("Running XGBoost: starting at", start_slice)
        print(f"ending at {end_slice}")

        self.prophet_model_obj = ProphetWrapper(
                self.cleaned_merged_data.copy(),
                self.short_split_date,
                start_date_slice=start_slice,
                end_date_slice=end_slice
                )
        self.prophet_model_obj.run_prophet()
        self.prophet_model_obj.get_forecast()
    
    def get_windowed_rsme_mae(self) -> tuple[float, float]:
        rmse, mae = self.prophet_model_obj.get_rmse_mae()
        return rmse, mae


@dataclass
class BacktestARIMA(BacktestParent):
    """
    A class that extends BacktestParent to perform backtesting using the ARIMA model.

    This class uses the ARIMAWrapper to create, train, and evaluate an ARIMA model
    within a specified backtesting time window.

    Attributes
    ----------
    cleaned_merged_data : pd.DataFrame
        The preprocessed data set to be used for backtesting.
    short_split_date : datetime
        The date at which to split the data into training and testing sets.
    n_backward : int, default 60
        The number of days to look back from the `short_split_date` to create the training set.
    forecast_window : int, default 24
        The size of the window, in hours, for which to forecast ahead from the `short_split_date`.
    forecast_metric : dict[str, list] | None, optional
        A dictionary to store forecast metrics, initialized to None by default.
    
    Methods
    -------
    __post_init__()
        Initializes the backtesting process by setting up the XGBWrapper with the
        appropriate data slices and training the model.
    get_windowed_rsme_mae() -> tuple[float, float]
        Retrieves the RSME and MAE for the forecasted window from the XGBWrapper.
    get_windowed_forecast() -> pd.DataFrame
        Retrieves the forecasted DataFrame with actual and predicted values for the window.
    """
    arima_model_obj: ARIMAWrapper | None = None

    def __post_init__(self):
        start_slice = self.short_split_date - timedelta(days=self.n_backward)
        end_slice = self.short_split_date + timedelta(hours=self.forecast_window)
        print("Running XGBoost: starting at", start_slice)
        print(f"ending at {end_slice}")

        self.arima_model_obj = ARIMAWrapper(
                self.cleaned_merged_data.copy(),
                self.short_split_date,
                start_date_slice=start_slice,
                end_date_slice=end_slice
                )
        self.arima_model_obj.run_ARIMA(3, 1, 4)
        self.arima_model_obj.get_forecast()
    
    def get_windowed_rsme_mae(self) -> tuple[float, float]:
        rmse, mae = self.arima_model_obj.get_rmse_mae()
        return rmse, mae
    