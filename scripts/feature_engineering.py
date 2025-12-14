import pandas as pd
import numpy as np
import os

def load_cleaned_data(path="data_clean/crimes_cleaned.csv"):
    print("Loading cleaned dataset")
    df = pd.read_csv(path, low_memory=False, parse_dates=["Date", "Updated_On"])
    print(f"Loaded: {df.shape}")
    return df

# ----------------------------
# TIME-BASED FEATURE ENGINEERING
# ----------------------------

def add_time_features(df):
    print("⏱️ Adding time features...")

    df["Month_Name"] = df["Date"].dt.month_name()
    df["Quarter"] = df["Date"].dt.quarter

    # Weekend flag
    df["Is_Weekend"] = df["DayOfWeek"].isin(["Saturday", "Sunday"])

    # Time of day categories
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    df["Time_Of_Day"] = df["Hour"].apply(get_time_of_day)

    return df

# ----------------------------
# LOCATION-BASED FEATURES
# ----------------------------

def add_location_features(df):
    print("📍 Adding location features...")

    df["Has_Coordinates"] = df["Latitude"].notnull() & df["Longitude"].notnull()
    df["Is_Domestic"] = df["Domestic"].astype(int)

    return df

# ----------------------------
# CRIME CATEGORY FEATURES
# ----------------------------

def add_crime_features(df):
    print("🚨 Adding crime category features...")

    # Simplify crime type into broad groups
    violent_types = ["HOMICIDE", "ASSAULT", "BATTERY", "ROBBERY", "CRIM SEXUAL ASSAULT"]
    property_types = ["BURGLARY", "THEFT", "MOTOR VEHICLE THEFT", "ARSON"]
    quality_of_life = ["NARCOTICS", "LIQUOR LAW VIOLATION", "PUBLIC PEACE VIOLATION", "PROSTITUTION"]

    def classify(primary_type):
        if primary_type in violent_types:
            return "Violent Crime"
        elif primary_type in property_types:
            return "Property Crime"
        elif primary_type in quality_of_life:
            return "Quality of Life Crime"
        else:
            return "Other Crime"

    df["Crime_Category"] = df["Primary_Type"].apply(classify)

    # Arrest flag as int
    df["Arrest_Flag"] = df["Arrest"].astype(int)

    return df

# ----------------------------
# SAVE FEATURE ENGINEERED DATA
# ----------------------------

def save_feature_engineered(df, output_path="data_clean/crimes_featured.parquet"):
    print("💾 Saving feature engineered dataset...")
    df.to_parquet(output_path, index=False)
    print(f"✔ Saved at {output_path}")

# ----------------------------
# MAIN RUNNER
# ----------------------------

def run_feature_engineering():
    df = load_cleaned_data()
    df = add_time_features(df)
    df = add_location_features(df)
    df = add_crime_features(df)

    save_feature_engineered(df)


if __name__ == "__main__":
    run_feature_engineering()