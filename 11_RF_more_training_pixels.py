
import h5py
import numpy as np
import pandas as pd
import joblib
import gc

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# =====================================
# 1. SETTINGS
# =====================================

DATA_PATH = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

SAMPLES_PER_MAP = 10000

TRAIN_MAPS = range(1, 101)
TEST_MAPS = range(191, 201)

BATCH_SIZE = 100000

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / f"models_{SAMPLES_PER_MAP}_pixels"
MODEL_DIR.mkdir(exist_ok=True)

# =====================================
# 2. FEATURE COMBINATIONS
# =====================================

groups = {
    "L": [
        "LIA_L",
        "sigma0_db_L_vv",
        "sigma0_db_L_vh"
    ],
    "C": [
        "LIA_C",
        "sigma0_db_C_vv",
        "sigma0_db_C_vh"
    ],
    "X": [
        "LIA_X",
        "sigma0_db_X_hh"
    ]
}

combinations = [
    "L", "C", "X", "LCX",
    "LC", "LX", "CX"
]


def get_features(combo):

    features = ["NDVI"]

    for band in combo:
        features.extend(groups[band])

    return features


# =====================================
# 3. READ ONE MAP
# =====================================

def read_map(f, map_number, features):

    columns = []

    for feature in features:

        if feature.startswith("LIA_"):
            key = feature
        else:
            key = f"{feature}_{map_number}"

        columns.append(f[key][:].ravel())

    X = np.column_stack(columns)
    y = f[f"SMC_{map_number}"][:].ravel()

    return X, y


# =====================================
# 4. TRAINING
# =====================================

def train_model(f, combo):

    print(f"\nTraining {combo} + NDVI")

    features = get_features(combo)

    rng = np.random.default_rng(42)

    X_list = []
    y_list = []

    for map_number in TRAIN_MAPS:

        X, y = read_map(
            f, map_number, features
        )

        # Native mask
        mask = (
            np.isfinite(X).all(axis=1)
            & np.isfinite(y)
        )

        valid_indices = np.flatnonzero(mask)

        if len(valid_indices) < SAMPLES_PER_MAP:
            raise ValueError(
                f"Not enough pixels in map {map_number}"
            )

        chosen = rng.choice(
            valid_indices,
            size=SAMPLES_PER_MAP,
            replace=False
        )

        X_list.append(X[chosen])
        y_list.append(y[chosen])

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    print("Training X:", X_train.shape)
    print("Training y:", y_train.shape)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=2
    )

    model.fit(X_train, y_train)

    model_path = (
        MODEL_DIR / f"rf_{combo}_NDVI.joblib"
    )

    joblib.dump(model, model_path)

    print("Saved:", model_path)

    del X_train, y_train, X_list, y_list
    gc.collect()

    return model


# =====================================
# 5. TEST WITH NATIVE MASK
# =====================================

def test_model(f, model, combo):

    print(f"\nTesting {combo} + NDVI")

    features = get_features(combo)

    results = []

    for map_number in TEST_MAPS:

        X, y = read_map(
            f, map_number, features
        )

        mask = (
            np.isfinite(X).all(axis=1)
            & np.isfinite(y)
        )

        valid_indices = np.flatnonzero(mask)
        n = len(valid_indices)

        if n == 0:
            raise ValueError(
                f"No valid pixels in map {map_number}"
            )

        sse = 0.0
        error_sum = 0.0

        y_valid = y[valid_indices]

        sst = np.sum(
            (y_valid - np.mean(y_valid)) ** 2
        )

        for start in range(0, n, BATCH_SIZE):

            idx = valid_indices[
                start:start + BATCH_SIZE
            ]

            prediction = model.predict(X[idx])

            error = prediction - y[idx]

            sse += np.sum(error ** 2)
            error_sum += np.sum(error)

        rmse = np.sqrt(sse / n)
        bias = error_sum / n

        r2 = (
            1 - sse / sst
            if sst > 0 else np.nan
        )

        results.append({
            "Combination": combo + "+NDVI",
            "Map": map_number,
            "Valid_pixels": n,
            "RMSE": rmse,
            "Bias": bias,
            "R2": r2
        })

        print(
            f"Map {map_number}: "
            f"Pixels={n}, "
            f"RMSE={rmse:.5f}, "
            f"Bias={bias:.5f}, "
            f"R2={r2:.4f}"
        )

    return results


# =====================================
# 6. TRAIN AND TEST ALL MODELS
# =====================================

all_results = []

with h5py.File(DATA_PATH, "r") as f:

    for combo in combinations:

        model = train_model(f, combo)

        results = test_model(
            f, model, combo
        )

        all_results.extend(results)

        del model
        gc.collect()


# =====================================
# 7. SAVE RESULTS
# =====================================

df = pd.DataFrame(all_results)

df.to_csv(
    PROJECT_DIR / f"RF_{SAMPLES_PER_MAP}_test_per_map.csv",
    index=False
)

summary = []

for combo, group in df.groupby(
    "Combination", sort=False
):

    n = group["Valid_pixels"]
    total = n.sum()

    overall_rmse = np.sqrt(
        np.sum(n * group["RMSE"] ** 2) / total
    )

    overall_bias = (
        np.sum(n * group["Bias"]) / total
    )

    summary.append({
        "Combination": combo,
        "Overall_RMSE": overall_rmse,
        "Overall_Bias": overall_bias,
        "Total_valid_pixels": int(total)
    })

summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    "Overall_RMSE"
)

summary_df.to_csv(
    PROJECT_DIR / f"RF_{SAMPLES_PER_MAP}_test_summary.csv",
    index=False
)

print("\nFINAL TEST RESULTS:")
print(summary_df.to_string(index=False))

print("\nTraining and testing completed!")
