from datetime import datetime, timedelta

# Data manipulation
import numpy as np
import pandas as pd
from forecast_utils import ARIMAWrapper, XGBWrapper, ProphetWrapper
from dataclasses import dataclass
import data_utils as du
from abc import ABC, abstractmethod
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


@dataclass
class BacktestParent(ABC):
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
    

@dataclass
class BacktestProphet(BacktestParent):
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
    