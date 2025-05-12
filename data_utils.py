import pandas as pd
import holidays
from meteostat import Point, Hourly
from datetime import datetime
from darts import TimeSeries
import numpy as np


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
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
                        n=10,
                        target=np.nan,
                        ) -> pd.DataFrame:
    
    if target is not np.nan:
        nan_indices = df[df[column] == target].index
    else:
        nan_indices = df[df[column].isna()].index

    for idx in nan_indices:
        start = max(0, idx - n)  # Ensure we don't go below 0
        end = min(len(df), idx + n + 1)  # Ensure we don't go beyond the DataFrame length
        
        print(f"\nNaN at index {idx}:")
        print(df.iloc[start:end])
        print("-" * 40)
    return df.iloc[start:end]


def ts_train_test_split(df: pd.DataFrame | TimeSeries,
                        split_date: datetime
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[df.index <= split_date].copy()
    test = df.loc[df.index > split_date].copy()
    return train, test
