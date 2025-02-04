import os
import numpy as np
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the NetCDF file
file_path = "data/NA_data.nc"
ds = xr.open_dataset(file_path)

# Extract relevant variables
time = ds["time"].values
lat = ds["lat"].values
lon = ds["lon"].values
season = ds["season"].values
storm_speed = ds["storm_speed"].values
storm_dir = ds["storm_dir"].values

# Prepare data for XGBoost
track_data = []
target_lat = []
target_lon = []

for i in range(lat.shape[0]):  # Loop over storms
    for j in range(1, lat.shape[1]):  # Skip the first observation for Δ calculation
        if np.isnan(lat[i, j]) or np.isnan(lon[i, j]) or np.isnan(lat[i, j - 1]) or np.isnan(lon[i, j - 1]):
            continue  # Skip missing data

        # Compute latitude and longitude shifts (target variables)
        delta_lat = lat[i, j] - lat[i, j - 1]
        delta_lon = lon[i, j] - lon[i, j - 1]

        # Extract input features
        track_data.append([
            lat[i, j - 1], lon[i, j - 1],  # Previous position
            storm_speed[i, j - 1] if not np.isnan(storm_speed[i, j - 1]) else 0,  # Speed
            storm_dir[i, j - 1] if not np.isnan(storm_dir[i, j - 1]) else 0,  # Direction
            season[i]  # Year/season as a feature
        ])

        # Store corresponding target values
        target_lat.append(delta_lat)
        target_lon.append(delta_lon)

# Convert to NumPy arrays
X = np.array(track_data)
y_lat = np.array(target_lat)
y_lon = np.array(target_lon)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train_lat, y_test_lat, y_train_lon, y_test_lon = train_test_split(
    X, y_lat, y_lon, test_size=0.2, random_state=42
)

# Train XGBoost models for latitude and longitude shifts
xgb_lat = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_lon = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, learning_rate=0.1)

xgb_lat.fit(X_train, y_train_lat)
xgb_lon.fit(X_train, y_train_lon)

# Predict on test set
y_pred_lat = xgb_lat.predict(X_test)
y_pred_lon = xgb_lon.predict(X_test)

# Evaluate model performance
mse_lat = mean_squared_error(y_test_lat, y_pred_lat)
mse_lon = mean_squared_error(y_test_lon, y_pred_lon)

print(f"Mean Squared Error for Latitude Shift Prediction: {mse_lat}")
print(f"Mean Squared Error for Longitude Shift Prediction: {mse_lon}")

# Visualization - Compare actual vs predicted shifts
plt.figure(figsize=(10, 5))
plt.scatter(y_test_lat, y_pred_lat, alpha=0.5, label="Latitude Shift", color="blue")
plt.scatter(y_test_lon, y_pred_lon, alpha=0.5, label="Longitude Shift", color="red")
plt.axline((0, 0), slope=1, color="black", linestyle="--")  # y = x reference line
plt.xlabel("Actual Shift")
plt.ylabel("Predicted Shift")
plt.legend()
plt.title("Actual vs Predicted Hurricane Track Shifts")
plt.show()
