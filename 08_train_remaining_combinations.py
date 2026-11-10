
import h5py
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

Path("models").mkdir(exist_ok=True)

# Define band features
groups = {
    "L": ["LIA_L", "sigma0_db_L_vv", "sigma0_db_L_vh"],
    "C": ["LIA_C", "sigma0_db_C_vv", "sigma0_db_C_vh"],
    "X": ["LIA_X", "sigma0_db_X_hh"]
}

# All seven combinations
combinations = ["L", "C", "X", "LCX", "LC", "LX", "CX"]

# Models that have already been trained
existing_models = {
    "L": "models/rf_L_NDVI.joblib",
    "LCX": "models/rf_LCX_NDVI.joblib"
}

# Build feature lists
def get_features(combo):
    features = ["NDVI"]

    for band in combo:
        features.extend(groups[band])

    return features

# Read one map
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

# Train and save five remaining models
models = {}

with h5py.File(file_path, "r") as f:

    for combo in combinations:

        if combo in existing_models:
            models[combo] = joblib.load(
                existing_models[combo]
            )
            print(f"Loaded existing model: {combo}")
            continue

        print(f"\nTraining {combo} + NDVI")

        features = get_features(combo)

        rng = np.random.default_rng(42)

        X_list = []
        y_list = []

        for map_number in range(1, 101):

            X_map, y_map = read_map(
                f, map_number, features
            )

            valid_mask = (
                np.isfinite(X_map).all(axis=1)
                & np.isfinite(y_map)
            )

            valid_indices = np.where(valid_mask)[0]

            sample_size = min(1000, len(valid_indices))

            if sample_size == 0:
                continue

            chosen = rng.choice(
                valid_indices,
                size=sample_size,
                replace=False
            )

            X_list.append(X_map[chosen])
            y_list.append(y_map[chosen])

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        model_path = f"models/rf_{combo}_NDVI.joblib"
        joblib.dump(model, model_path)

        models[combo] = model

        print(f"Saved: {model_path}")

# Evaluate all seven models on a common mask
print("\nEvaluating on common valid pixels")

all_features = get_features("LCX")
results = []

with h5py.File(file_path, "r") as f:

    for map_number in range(191, 201):

        # Read all 9 features
        X_all, y_true = read_map(
            f, map_number, all_features
        )

        # Same valid pixels for every model
        common_mask = (
            np.isfinite(X_all).all(axis=1)
            & np.isfinite(y_true)
        )

        y_valid = y_true[common_mask]

        print(
            f"\nMap {map_number}: "
            f"{len(y_valid)} common valid pixels"
        )

        if len(y_valid) == 0:
            raise ValueError(
                f"No valid pixels in map {map_number}"
            )

        for combo in combinations:

            features = get_features(combo)

            # Select the appropriate columns
            feature_indices = [
                all_features.index(name)
                for name in features
            ]

            X_valid = X_all[
                common_mask
            ][:, feature_indices]

            model = models[combo]

            prediction = model.predict(X_valid)

            rmse = np.sqrt(
                mean_squared_error(y_valid, prediction)
            )

            bias = np.mean(prediction - y_valid)

            r2 = r2_score(y_valid, prediction)

            results.append({
                "Combination": combo + "+NDVI",
                "Map": map_number,
                "Valid_pixels": len(y_valid),
                "RMSE": rmse,
                "Bias": bias,
                "R2": r2
            })

            print(
                f"{combo}+NDVI: "
                f"RMSE={rmse:.5f}, "
                f"Bias={bias:.5f}, "
                f"R2={r2:.4f}"
            )

# Save per-map results
df = pd.DataFrame(results)

df.to_csv(
    "RF_all_combinations_per_map.csv",
    index=False
)

# Calculate overall metrics
summary = []

for combo, group in df.groupby(
    "Combination", sort=False
):

    n = group["Valid_pixels"]
    total = n.sum()

    overall_rmse = np.sqrt(
        np.sum(n * group["RMSE"]**2) / total
    )

    overall_bias = np.sum(
        n * group["Bias"]
    ) / total

    summary.append({
        "Combination": combo,
        "Overall_RMSE": overall_rmse,
        "Overall_Bias": overall_bias,
        "Total_valid_pixels": int(total)
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "RF_all_combinations_summary.csv",
    index=False
)

print("\nFinal comparison:")
print(summary_df.to_string(index=False))
print("\nAll results saved successfully!")
