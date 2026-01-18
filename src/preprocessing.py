import pandas as pd
from src.config import DROP_COLUMNS

def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    return df