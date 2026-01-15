import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Loading the trained model
MODEL_PATH = "../notebooks/crime_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Error: Model not found. Please run the training step first.")
    exit()

# Mappings (Must match training data exactly)
TYPE_MAP = {0: 'Drug/Vice', 1: 'Property', 2: 'Violent', 3: 'Other'}
DAY_MAP = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}

def get_all_crime_probabilities(date_str, hour, lat, lng):
    """
    Returns the probabilities of ALL crime categories for a given time and location.
    """
    # --- A. Process Date Features ---
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return "Error: Invalid Date Format (Use YYYY-MM-DD)"

    year = dt.year
    month = dt.month
    day_str = dt.strftime('%a')
    day_numeric = DAY_MAP.get(day_str, 0)

    # --- B. Create Input DataFrame ---
    X_input = pd.DataFrame(
        [[lat, lng, year, month, hour, day_numeric]], 
        columns=['latitude', 'longitude', 'year', 'month', 'hour', 'dayofweek']
    )

    # --- C. Get Probabilities ---
    probs = model.predict_proba(X_input)[0]
    
    # --- D. Format Output ---
    results = {}
    print(f"\n--- Probabilities for {date_str} {hour}:00 at ({lat}, {lng}) ---")
    
    # Loop through all classes in order 0-3
    for i in range(len(TYPE_MAP)):
        crime_type = TYPE_MAP[i]
        probability = probs[i] * 100
        results[crime_type] = probability
        print(f"{crime_type:<12}: {probability:.2f}%")
        
    return results

if __name__ == "__main__":
    # Test Case
    probs = get_all_crime_probabilities(
        date_str="2025-12-31", 
        hour=23, 
        lat=41.8781, 
        lng=-87.6298
    )