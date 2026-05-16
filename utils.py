import pandas as pd
import numpy as np

def prepare_features(df):
    """
    Prepares raw dataframe for modeling by encoding categorical variables 
    and selecting features.
    """
    df_processed = df.copy()
    
    # Extract time features
    if 'timestamp' in df_processed.columns:
        df_processed['hour'] = pd.to_datetime(df_processed['timestamp']).dt.hour
        df_processed['day_of_week'] = pd.to_datetime(df_processed['timestamp']).dt.dayofweek
    
    # One-hot encode weather condition
    if 'weather_condition' in df_processed.columns:
        weather_dummies = pd.get_dummies(df_processed['weather_condition'], prefix='weather')
        # Ensure all expected columns exist to prevent errors during inference
        expected_weathers = ['weather_Clear', 'weather_Fog', 'weather_Rain', 'weather_Snow']
        for col in expected_weathers:
            if col not in weather_dummies.columns:
                weather_dummies[col] = 0
        df_processed = pd.concat([df_processed, weather_dummies], axis=1)
    
    features = [
        'traffic_flow_rate', 'lane_occupancy_ratio', 'vehicle_speed_avg', 
        'sudden_braking_events', 'weather_Clear', 'weather_Fog', 
        'weather_Rain', 'weather_Snow'
    ]
    
    # Add time features if they exist
    if 'hour' in df_processed.columns:
        features.extend(['hour', 'day_of_week'])
        
    return df_processed[features]

def style_risk_level(val):
    """Pandas style function for risk categories."""
    color = 'green' if val == 'Low' else 'orange' if val == 'Medium' else 'red'
    return f'color: {color}; font-weight: bold;'
