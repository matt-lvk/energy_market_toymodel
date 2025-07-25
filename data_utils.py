import numpy as np
import pandas as pd
from datetime import datetime

import holidays
from meteostat import Point, Hourly
from darts import TimeSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the date features from the DataFrame such as hour, dayofweek, quarter, 
    year, dayofyear, dayofmonth, month, weekofyear

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional features:
        - is_weekend: boolean, True if the day is a weekend
        - datetime: datetime column
        - hour: int, hour of the day
        - dayofweek: int, day of the week (0-6)
        - is_weekend: boolean, True if the day is a weekend
        - quarter: int, quarter of the year
        - year: int, year
        - dayofyear: int, day of the year
        - dayofmonth: int, day of the month
        - month: int, month
        - weekofyear: int, week of the year
        - is_holiday: boolean, True if the day is a holiday in Germany or Luxembourg
    """
    df = df.copy()
    df['is_weekend'] = df.index.dayofweek.isin([5, 6])
    df['datetime'] = df.index
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df.index.dayofweek.isin([5, 6])
    df['quarter'] = df['datetime'].dt.quarter
    df['year'] = df['datetime'].dt.year
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['dayofmonth'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['weekofyear'] = df['datetime'].dt.isocalendar().week
    df['is_holiday'] = df['datetime'].dt.date.isin(
                                            holidays.country_holidays('DE') +
                                            holidays.country_holidays('LU')
                                            )

    return df


def add_locations_weather(
                    df: pd.DataFrame,
                    locations: dict[str, tuple[float, float]]
                    ) -> pd.DataFrame:
    """
    Use meteostat to fetch weather data to the given DataFrame from the given 
    locations.
    
    If more than one location is given, the function will fetch weather data 
    for each location and return its mean value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    locations : dict[str, tuple[float, float]]
        Mapping of country names to tuples of latitude and longitude.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for weather data from each of the given locations.
    """
    start_datetime = df.index.min()
    end_datetime = df.index.max()

    weather_data = {}

    for country, (lat, lon) in locations.items():
        pin = Point(lat, lon)
        weather = Hourly(pin, start_datetime, end_datetime)
        weather_df = weather.fetch()
        weather_data[country] = weather_df

    return weather_df


def show_nrows_around_target(
                        df: pd.DataFrame,
                        column: str,
                        print_rows: bool = False,
                        n=10,
                        target=np.nan,
                        ) -> pd.DataFrame:
    
    """
    Print the given number of rows before and after the target value in the 
    given column. Also returns the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Name of the column to look for the target value
    n : int, optional
        Number of rows to print before and after the target value, by default 10
    target : any, optional
        Value to look for in the column, by default np.nan

    Returns
    -------
    pd.DataFrame
        The last printed DataFrame

    Notes
    -----
    If the target value is np.nan, it will look for NaN values in the column.
    """
    if target is not np.nan:
        nan_indices = df[df[column] == target].index
    else:
        nan_indices = df[df[column].isna()].index

    for idx in nan_indices:
        start = max(0, idx - n)  # Ensure we don't go below 0
        end = min(len(df), idx + n + 1)  # Ensure we don't go beyond the DataFrame length
        
        if print_rows:
            print(f"\nNaN at index {idx}:")
            print(df.iloc[start:end])
            print("-" * 40)
    return df.iloc[start:end]


def ts_train_test_split(df: pd.DataFrame | TimeSeries,
                        split_date: datetime
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into training and testing sets based on a given date.
    Training data will include the split_date.

    Parameters
    ----------
    df : pd.DataFrame | TimeSeries
        Time series data
    split_date : datetime
        Date to split the data

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of training and testing data
    """
    if not (isinstance(df.index, pd.DatetimeIndex)
            or isinstance(df.index, datetime)):
        raise Exception("DF index must be a DatetimeIndex or datetime object.")
    
    train = df.loc[df.index <= split_date].copy()
    test = df.loc[df.index > split_date].copy()
    return train, test


def get_rmse_mae(y_true, y_pred) -> tuple[float, float]:
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)

    return rmse, mae


def print_rmse_mae_table(
                        forecast_metric: dict[str, list[float, float]],
                        title: str | None = None
                        ) -> None:
    if title:
        print(title)
    table = [[key] + (value if isinstance(value, list) else list(value.values())) 
            for key, value in forecast_metric.items()]
    print(tabulate(table, headers=["Model", "RMSE", "MAE"], tablefmt="grid"))
    print("-" * 60)