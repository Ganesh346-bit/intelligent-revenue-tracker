import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load revenue data from CSV, parse dates, and return a DataFrame.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    return df
