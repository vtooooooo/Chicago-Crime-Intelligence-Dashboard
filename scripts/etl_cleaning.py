import pandas as pd
import numpy as np
import os

def load_data(path):
    print("Loading raw dataset")
    df = pd.read_csv(path, low_memory=False)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def clean_dates(df):
    print("Cleaning date columns")
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df['Updated_On'] = pd.to_datetime(df['Updated_On'], format='mixed', errors='coerce')
    
    # Extract useful time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    return df

def clean_booleans(df):
    print("🔄 Converting boolean columns...")
    df['Arrest'] = df['Arrest'].astype('bool')
    df['Domestic'] = df['Domestic'].astype('bool')
    return df

def handle_missing(df):
    print("🧼 Handling missing values...")

    df['Location_Description'] = df['Location_Description'].fillna("Unknown")

    # fill numeric NAs
    num_cols = ['Community_Area', 'Ward', 'District', 'Beat']
    for col in num_cols:
        df[col] = df[col].fillna(-1).astype(int)

    # Replace missing coordinates with None
    df['Latitude'] = df['Latitude'].fillna(np.nan)
    df['Longitude'] = df['Longitude'].fillna(np.nan)

    return df

def drop_duplicates(df):
    print("🗑 Removing duplicates...")
    before = df.shape[0]
    df = df.drop_duplicates(subset=['ID'])
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows")
    return df

def save_cleaned(df, output_dir="data_clean/"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(output_dir, "crimes_cleaned.csv")
    parquet_path = os.path.join(output_dir, "crimes_cleaned.parquet")
    
    print("💾 Saving cleaned dataset...")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print("✔ Cleaned data saved!")
    print(f"CSV: {csv_path}")
    print(f"PARQUET: {parquet_path}")

def run_etl():
    raw_path = "data_raw/chicago_crimes.csv"

    df = load_data(raw_path)
    df = drop_duplicates(df)
    df = clean_dates(df)
    df = clean_booleans(df)
    df = handle_missing(df)

    save_cleaned(df)
    
if __name__ == "__main__":
    run_etl()