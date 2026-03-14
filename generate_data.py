import pandas as pd
import numpy as np

# Generate time arrays
t_normal1 = np.linspace(0, 50, 5000)
t_damage = np.linspace(50, 70, 2000)
t_traffic = np.linspace(70, 90, 2000)
t_normal2 = np.linspace(90, 110, 2000)

print("Generating complex telemetry data...")

# 1. Normal State (Healthy bridge, light load, 5Hz natural mode)
normal_reading1 = np.sin(2 * np.pi * 5 * t_normal1) + np.random.normal(0, 0.4, 5000)
normal_status1 = np.zeros(5000)

# 2. Damaged State (Stiffness drops -> frequency shifts to 4.2Hz, micro-fracture noise added)
# This requires the model to rely on the FFT feature, not just RMS!
damage_reading = 2.0 * np.sin(2 * np.pi * 4.2 * t_damage) + 0.8 * np.sin(2 * np.pi * 12 * t_damage) + np.random.normal(0, 1.2, 2000)
damage_status = np.ones(2000)

# 3. Heavy Traffic (Healthy bridge, but HUGE amplitude to trick the AI)
traffic_reading = 3.0 * np.sin(2 * np.pi * 5 * t_traffic) + np.random.normal(0, 0.9, 2000)
traffic_status = np.zeros(2000) # Still healthy!

# 4. Return to Normal
normal_reading2 = np.sin(2 * np.pi * 5 * t_normal2) + np.random.normal(0, 0.4, 2000)
normal_status2 = np.zeros(2000)

# Combine all phases sequentially
sensor_readings = np.concatenate([normal_reading1, damage_reading, traffic_reading, normal_reading2])
status = np.concatenate([normal_status1, damage_status, traffic_status, normal_status2])

# Create DataFrame and save
df = pd.DataFrame({'Sensor_Reading': sensor_readings, 'Status': status})
df.to_csv('bridge_data_v2.csv', index=False)
print("bridge_data_v2.csv generated successfully! Ready for AI ingestion.")