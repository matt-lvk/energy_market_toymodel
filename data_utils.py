import pandas as pd
import holidays
from meteostat import Point, Hourly

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date-related features to a DataFrame with a datetime index.

    This function creates new columns in the DataFrame based on the datetime index,
    including features like hour, day of week, is weekend, quarter, year, day of year,
    day of month, month, and week of year.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime index.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with additional date-related columns:
        - is_weekend: boolean, True if the day is Saturday or Sunday
        - datetime: datetime object, copied from the index
        - hour: int, hour of the day (0-23)
        - dayofweek: int, day of the week (0-6, where 0 is Monday)
        - quarter: int, quarter of the year (1-4)
        - year: int, year
        - dayofyear: int, day of the year (1-366)
        - dayofmonth: int, day of the month (1-31)
        - month: int, month (1-12)
        - weekofyear: int, ISO week number of the year (1-53)

    Notes
    -----
    This function creates a copy of the input DataFrame and does not modify the original.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2021-01-01', periods=5, freq='D')
    >>> df = pd.DataFrame({'value': range(5)}, index=dates)
    >>> df_with_features = add_date_features(df)
    >>> print(df_with_features.columns)
    Index(['value', 'is_weekend', 'datetime', 'hour', 'dayofweek', 'quarter',
            'year', 'dayofyear', 'dayofmonth', 'month', 'weekofyear'],
        dtype='object')
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


def show_nrows_around_nan(df: pd.DataFrame, column: str, n=10) -> pd.DataFrame:
    # Find indices of NaN values
    nan_indices = df[df[column].isna()].index

    for idx in nan_indices:
        start = max(0, idx - n)  # Ensure we don't go below 0
        end = min(len(df), idx + n + 1)  # Ensure we don't go beyond the DataFrame length
        
        print(f"\nNaN at index {idx}:")
        print(df.iloc[start:end])
        print("-" * 40)
    return df.iloc[start:end]