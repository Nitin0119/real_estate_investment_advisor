import pandas as pd
from datetime import date

current_year = date.today().year

transport_map = {
    "Low":0,
    "Medium":1,
    "High":2,
}

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_of_property"] = current_year - df['year_built']
    df['price_per_sqft'] = (df['price_in_lakhs']*100000)/ df['size_in_sqft']
    df['transport_score'] = df['public_transport_accessibility'].map(transport_map)
    df['amenity_count'] = df['amenities'].apply(lambda x: len(str(x).split(',')))
    return df
