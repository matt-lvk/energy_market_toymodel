import pandas as pd


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

    return df


def show_price_around_nan(df: pd.DataFrame, column: str, n=10) -> pd.DataFrame:
    # Find indices of NaN values
    nan_indices = df[df[column].isna()].index

    for idx in nan_indices:
        start = max(0, idx - n)  # Ensure we don't go below 0
        end = min(len(df), idx + n + 1)  # Ensure we don't go beyond the DataFrame length
        
        print(f"\nNaN at index {idx}:")
        print(df.iloc[start:end])
        print("-" * 40)
    return df.iloc[start:end]