
import h5py
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score


# 1. File paths
file_path = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"
model_folder = Path("models_full_pipeline")


# 2. Define radar features
groups = {
    "L": ["LIA_L", "sigma0_db_L_vv", "sigma0_db_L_vh"],
    "C": ["LIA_C", "sigma0_db_C_vv", "sigma0_db_C_vh"],
    "X": ["LIA_X", "sigma0_db_X_hh"]
}

combinations = ["L", "C", "X", "LCX", "LC", "LX", "CX"]


# 3. Get features for each combination
def get_features(combo):

    features = ["NDVI"]

    for band in combo:
        features.extend(groups[band])

    return features


# 4. Read one map
def read_map(f, map_number, features):

    columns = []

    for feature in features:

        if feature.startswith("LIA_"):
            name = feature
        else:
            name = f"{feature}_{map_number}"

        columns.append(f[name][:].ravel())

    X = np.column_stack(columns)
    y = f[f"SMC_{map_number}"][:].ravel()

    return X, y


# 5. Evaluate all models
results = []

with h5py.File(file_path, "r") as f:

    for combo in combinations:

        print(f"\nTesting {combo} + NDVI")

        features = get_features(combo)

        model_path = model_folder / f"rf_{combo}_NDVI.joblib"
        model = joblib.load(model_path)

        for map_number in range(191, 201):

            X, y = read_map(f, map_number, features)

            # Native mask: only the features of this model
            valid_mask = (
                np.isfinite(X).all(axis=1)
                & np.isfinite(y)
            )

            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) == 0:
                raise ValueError(
                    f"No valid pixels in map {map_number}"
                )

            # Predict SMC
            y_pred = model.predict(X_valid)

            # Metrics
            rmse = np.sqrt(
                mean_squared_error(y_valid, y_pred)
            )

            bias = np.mean(y_pred - y_valid)

            r2 = r2_score(y_valid, y_pred)

            results.append({
                "Combination": combo + "+NDVI",
                "Map": map_number,
                "Valid_pixels": len(y_valid),
                "RMSE": rmse,
                "Bias": bias,
                "R2": r2
            })

            print(
                f"Map {map_number}: "
                f"Pixels={len(y_valid)}, "
                f"RMSE={rmse:.5f}, "
                f"Bias={bias:.5f}, "
                f"R2={r2:.4f}"
            )


# 6. Save per-map results
df = pd.DataFrame(results)

df.to_csv(
    "RF_native_mask_per_map.csv",
    index=False
)


# 7. Calculate overall metrics
summary = []

for combo, group in df.groupby("Combination", sort=False):

    n = group["Valid_pixels"]
    total = n.sum()

    overall_rmse = np.sqrt(
        np.sum(n * group["RMSE"]**2) / total
    )

    overall_bias = np.sum(n * group["Bias"]) / total

    summary.append({
        "Combination": combo,
        "Overall_RMSE": overall_rmse,
        "Overall_Bias": overall_bias,
        "Total_valid_pixels": int(total)
    })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "RF_native_mask_summary.csv",
    index=False
)

print("\nFINAL NATIVE MASK RESULTS:")
print(summary_df.to_string(index=False))

print("\nNative-mask evaluation completed!")
