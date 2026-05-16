import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TrafficDataGenerator:
    """
    Generates synthetic realistic traffic data for the Cyber-Physical System Digital Twin.
    Simulates physical layer parameters (speed, flow, occupancy, weather, events).
    """
    def __init__(self, num_records=1000, random_seed=42):
        self.num_records = num_records
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def generate_data(self):
        """Generates the main synthetic dataset."""
        # Timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.num_records / 60) # Simulate 1 record per minute
        timestamps = pd.date_range(start=start_time, end=end_time, periods=self.num_records)

        # Base physical parameters
        time_of_day = timestamps.hour + timestamps.minute / 60.0
        
        # Rush hour simulation (peaks around 8 AM and 5 PM)
        rush_hour_factor = np.exp(-0.5 * ((time_of_day - 8) / 1.5)**2) + np.exp(-0.5 * ((time_of_day - 17) / 2.0)**2)
        
        # Base distributions
        weather_conditions = np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], size=self.num_records, p=[0.6, 0.25, 0.1, 0.05])
        weather_impact = {'Clear': 1.0, 'Rain': 0.8, 'Fog': 0.6, 'Snow': 0.4}
        weather_multiplier = np.array([weather_impact[w] for w in weather_conditions])

        traffic_flow_rate = np.clip(np.random.normal(1500, 300, self.num_records) * (1 + rush_hour_factor * 1.5), 200, 5000)
        lane_occupancy_ratio = np.clip(traffic_flow_rate / 6000.0 + np.random.normal(0, 0.05, self.num_records), 0.05, 0.98)
        
        # Speed logic: higher occupancy -> lower speed. Worse weather -> lower speed.
        free_flow_speed = 100 # km/h
        vehicle_speed_avg = free_flow_speed * (1 - lane_occupancy_ratio**2) * weather_multiplier
        vehicle_speed_avg = np.clip(vehicle_speed_avg + np.random.normal(0, 5, self.num_records), 5, 120)

        # Sudden braking events
        sudden_braking_events = np.random.poisson((lane_occupancy_ratio * 3) + (1 - weather_multiplier) * 5)

        # Targets/Labels for supervised learning
        # Collision Probability Logic: High occupancy + High sudden braking + Bad weather = High Probability
        collision_probability = (lane_occupancy_ratio * 0.4) + (sudden_braking_events / 10.0 * 0.4) + ((1 - weather_multiplier) * 0.2)
        collision_probability = np.clip(collision_probability + np.random.normal(0, 0.05, self.num_records), 0, 1)

        congestion_level = np.clip(lane_occupancy_ratio * 0.8 + (150 - vehicle_speed_avg)/150 * 0.2 + np.random.normal(0, 0.05, self.num_records), 0, 1)

        # Risk Categories based on collision probability and congestion
        risk_score = (collision_probability * 0.7) + (congestion_level * 0.3)
        risk_category = []
        for score in risk_score:
            if score < 0.3:
                risk_category.append('Low')
            elif score < 0.65:
                risk_category.append('Medium')
            else:
                risk_category.append('High')

        df = pd.DataFrame({
            'timestamp': timestamps,
            'traffic_flow_rate': traffic_flow_rate,
            'lane_occupancy_ratio': lane_occupancy_ratio,
            'vehicle_speed_avg': vehicle_speed_avg,
            'weather_condition': weather_conditions,
            'sudden_braking_events': sudden_braking_events,
            'collision_probability': collision_probability,
            'congestion_level': congestion_level,
            'risk_category': risk_category
        })
        
        return df

if __name__ == "__main__":
    generator = TrafficDataGenerator(num_records=5000)
    df = generator.generate_data()
    df.to_csv("synthetic_traffic_data.csv", index=False)
    print("Synthetic data generated and saved to synthetic_traffic_data.csv")
