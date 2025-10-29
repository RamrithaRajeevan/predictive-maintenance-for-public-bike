import pandas as pd
import numpy as np

# Number of samples
num_samples = 500

np.random.seed(42)

# Generate realistic features
data = {
    "distance_travelled_km": np.random.randint(0, 5000, num_samples),
    "road_roughness_index": np.random.randint(1, 11, num_samples),
    "component_age_mo": np.random.randint(1, 60, num_samples),
    "rider_weight_kg": np.random.randint(40, 150, num_samples),
}

df = pd.DataFrame(data)

# Failure probability formulas
brake_failure_prob = (df["distance_travelled_km"] / 2500) + (df["rider_weight_kg"] / 500) + (df["road_roughness_index"] / 40)
chain_failure_prob = (df["distance_travelled_km"] / 1800) + (df["component_age_mo"] / 100) + (df["road_roughness_index"] / 30)
tire_failure_prob = (df["distance_travelled_km"] / 2000) + (df["road_roughness_index"] / 25)
gear_failure_prob = (df["distance_travelled_km"] / 2200) + (df["component_age_mo"] / 120) + (df["road_roughness_index"] / 35)

# Add noise for realism
noise = np.random.normal(0, 0.1, num_samples)
failure_prob = (brake_failure_prob + chain_failure_prob + tire_failure_prob + gear_failure_prob) / 4 + noise

# Target variable
df["will_fail_soon"] = (failure_prob > 0.8).astype(int)

# Save dataset
df.to_csv("bike_component_data.csv", index=False)
print("✅ Data generated and saved as 'bike_component_data.csv'")
