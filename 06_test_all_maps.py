
import h5py
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, r2_score

file_path = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

# Load the trained model (no new training)
rf_model = joblib.load("models/rf_LCX_NDVI.joblib")

features = [
    "NDVI",
    "LIA_L",
    "sigma0_db_L_vv",
    "sigma0_db_L_vh",
    "LIA_C",
    "sigma0_db_C_vv",
    "sigma0_db_C_vh",
    "LIA_X",
    "sigma0_db_X_hh"
]

results = []

with h5py.File(file_path, "r") as f:

    for map_number in range(191, 201):

        columns = []

        for feature in features:

            if feature.startswith("LIA_"):
                variable_name = feature
            else:
                variable_name = f"{feature}_{map_number}"

            values = f[variable_name][:].ravel()
            columns.append(values)

        X_test = np.column_stack(columns)
        y_test = f[f"SMC_{map_number}"][:].ravel()

        # Keep valid pixels
        valid_mask = (
            np.isfinite(X_test).all(axis=1)
            & np.isfinite(y_test)
        )

        X_valid = X_test[valid_mask]
        y_valid = y_test[valid_mask]

        # Predict using the saved model
        y_pred = rf_model.predict(X_valid)

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        bias = np.mean(y_pred - y_valid)
        r2 = r2_score(y_valid, y_pred)

        results.append({
            "Map": map_number,
            "Valid_pixels": len(y_valid),
            "RMSE": rmse,
            "Bias": bias,
            "R2": r2
        })

        print(
            f"Map {map_number}: "
            f"RMSE={rmse:.5f}, "
            f"Bias={bias:.5f}, "
            f"R2={r2:.4f}"
        )

# Create results table
df_results = pd.DataFrame(results)

print("\nFinal test results:")
print(df_results)

# Save results
df_results.to_csv(
    "RF_LCX_NDVI_test_results.csv",
    index=False
)

print("\nResults saved successfully!")
