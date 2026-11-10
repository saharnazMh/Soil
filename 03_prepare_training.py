
import h5py
import numpy as np

file_path = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

# First experiment: L + NDVI
features = [
    "NDVI",
    "LIA_L",
    "sigma0_db_L_vv",
    "sigma0_db_L_vh"
]

# Number of sampled pixels per map
n_samples = 1000

rng = np.random.default_rng(42)

X_list = []
y_list = []

with h5py.File(file_path, "r") as f:

    for map_number in range(1, 101):

        data = []

        for feature in features:

            if feature.startswith("LIA_"):
                variable_name = feature
            else:
                variable_name = f"{feature}_{map_number}"

            values = f[variable_name][:].ravel()
            data.append(values)

        # Combine features into columns
        X_map = np.column_stack(data)

        # Read target SMC
        y_map = f[f"SMC_{map_number}"][:].ravel()

        # Keep only finite input and target values
        valid_mask = (
            np.isfinite(X_map).all(axis=1)
            & np.isfinite(y_map)
        )

        valid_indices = np.where(valid_mask)[0]

        # Sample valid pixels without replacement
        sample_size = min(n_samples, len(valid_indices))

        chosen = rng.choice(
            valid_indices,
            size=sample_size,
            replace=False
        )

        X_list.append(X_map[chosen])
        y_list.append(y_map[chosen])

        print(
            f"Map {map_number}: "
            f"{sample_size} samples selected"
        )

# Combine samples from all training maps
X_train = np.vstack(X_list)
y_train = np.concatenate(y_list)

print("\nFinal training data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)



from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)


import joblib
from pathlib import Path

# Create folder for trained models
Path("models").mkdir(exist_ok=True)

# Save the trained RF model
joblib.dump(rf_model, "models/rf_L_NDVI.joblib")

print("Model saved successfully!")

print("Random Forest training completed!")



from sklearn.metrics import mean_squared_error

# Test on synthetic map 192
test_map = 192

with h5py.File(file_path, "r") as f:
    test_data = []

    for feature in features:
        if feature.startswith("LIA_"):
            variable_name = feature
        else:
            variable_name = f"{feature}_{test_map}"

        values = f[variable_name][:].ravel()
        test_data.append(values)

    X_test = np.column_stack(test_data)
    y_test = f[f"SMC_{test_map}"][:].ravel()

# Keep only valid pixels
valid_mask = (
    np.isfinite(X_test).all(axis=1)
    & np.isfinite(y_test)
)

X_test_valid = X_test[valid_mask]
y_test_valid = y_test[valid_mask]

# Predict soil moisture
y_pred = rf_model.predict(X_test_valid)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred))

print("\nTest results for Map 192:")
print("Valid test pixels:", len(y_test_valid))
print("RMSE:", rmse)



import matplotlib.pyplot as plt

# Reconstruct the predicted SMC map
pred_map = np.full(
    (1000 * 1000,),
    np.nan
)

pred_map[valid_mask] = y_pred
pred_map = pred_map.reshape(1000, 1000)

# Reconstruct the true SMC map
true_map = y_test.reshape(1000, 1000)

# Plot true and predicted SMC
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(
    true_map,
    cmap="jet",
    vmin=0,
    vmax=0.5
)
axes[0].set_title("True SMC - Map 192")
plt.colorbar(im1, ax=axes[0], label="SMC (m³/m³)")

im2 = axes[1].imshow(
    pred_map,
    cmap="jet",
    vmin=0,
    vmax=0.5
)
axes[1].set_title("RF Predicted SMC - Map 192")
plt.colorbar(im2, ax=axes[1], label="SMC (m³/m³)")

plt.tight_layout()
plt.show()



# Calculate prediction errors
error_map = pred_map - true_map

# Plot spatial error map
plt.figure(figsize=(7, 6))

im = plt.imshow(
    error_map,
    cmap="RdBu_r",
    vmin=-0.15,
    vmax=0.15
)

plt.colorbar(
    im,
    label="Prediction Error (m³/m³)"
)

plt.title("RF Prediction Error - Map 192")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.tight_layout()
plt.show()


from sklearn.metrics import r2_score

bias = np.mean(y_pred - y_test_valid)
r2 = r2_score(y_test_valid, y_pred)

print("\nAdditional evaluation metrics:")
print("Bias:", bias)
print("R2:", r2)