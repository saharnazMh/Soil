"""
Random Forest experiment using ALL finite/valid pixels from each training map.

Based on:
- 11_RF_more_training_pixels.py
- 02_prepare_rf_data.py

Maps:
- Training: 1-100
- Testing: 191-200

Mask definition is intentionally the same as the uploaded script:
all selected features and SMC must be finite (NaN/Inf excluded).
Numeric sentinel values such as -99 are NOT excluded by this script.
Confirm the meaning of -99 in the source data before the production run.
"""

import gc
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. SETTINGS -- CHECK BEFORE RUNNING
# ============================================================

# Change this to the actual location of the MAT file on the server.
DATA_PATH = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

TRAIN_MAPS = range(1, 101)
TEST_MAPS = range(191, 201)

TEST_BATCH_SIZE = 100_000

# Start moderately; increase only after checking server resources.
N_JOBS = 16

N_ESTIMATORS = 100
MAX_DEPTH = 15
RANDOM_STATE = 42

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models_all_pixels_native_mask"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PER_MAP_CSV = PROJECT_DIR / "RF_all_pixels_test_per_map.csv"
SUMMARY_CSV = PROJECT_DIR / "RF_all_pixels_test_summary.csv"


# ============================================================
# 2. FEATURES AND SEVEN COMBINATIONS
# ============================================================

GROUPS = {
    "L": ["LIA_L", "sigma0_db_L_vv", "sigma0_db_L_vh"],
    "C": ["LIA_C", "sigma0_db_C_vv", "sigma0_db_C_vh"],
    "X": ["LIA_X", "sigma0_db_X_hh"],
}

COMBINATIONS = ["L", "C", "X", "LCX", "LC", "LX", "CX"]


def get_features(combo):
    features = ["NDVI"]
    for band in combo:
        features.extend(GROUPS[band])
    return features


# ============================================================
# 3. READ ONE MAP
# ============================================================

def read_map(h5_file, map_number, features):
    columns = []
    expected_size = None

    for feature in features:
        key = feature if feature.startswith("LIA_") else f"{feature}_{map_number}"

        if key not in h5_file:
            raise KeyError(
                f"Dataset '{key}' was not found in the MAT file. "
                "Check DATA_PATH and the variable names."
            )

        values = np.asarray(h5_file[key][:], dtype=np.float32).reshape(-1)

        if expected_size is None:
            expected_size = values.size
        elif values.size != expected_size:
            raise ValueError(
                f"Different array sizes in map {map_number}: "
                f"'{key}' has {values.size}; expected {expected_size}."
            )

        columns.append(values)

    target_key = f"SMC_{map_number}"
    if target_key not in h5_file:
        raise KeyError(f"Dataset '{target_key}' was not found in the MAT file.")

    y = np.asarray(h5_file[target_key][:], dtype=np.float32).reshape(-1)

    if y.size != expected_size:
        raise ValueError(
            f"SMC size mismatch in map {map_number}: "
            f"{y.size} values versus {expected_size} feature values."
        )

    X = np.column_stack(columns).astype(np.float32, copy=False)
    return X, y


def make_native_mask(X, y):
    # Same validity rule as the uploaded experiment 11:
    # finite selected input features AND finite target.
    return np.isfinite(X).all(axis=1) & np.isfinite(y)


# ============================================================
# 4. FIRST PASS: COUNT VALID PIXELS PER TRAINING MAP
# ============================================================

def count_training_pixels(h5_file, combo, features):
    print(f"\nCounting valid training pixels for {combo}+NDVI ...")
    counts = []
    total = 0

    for map_number in TRAIN_MAPS:
        X, y = read_map(h5_file, map_number, features)
        mask = make_native_mask(X, y)
        n_valid = int(np.count_nonzero(mask))

        if n_valid == 0:
            raise ValueError(
                f"Map {map_number} has zero valid pixels for {combo}+NDVI."
            )

        counts.append(n_valid)
        total += n_valid
        print(f"Map {map_number}: {n_valid:,} valid training pixels")

        del X, y, mask
        gc.collect()

    print(f"Total training pixels for {combo}+NDVI: {total:,}")
    return counts, total


# ============================================================
# 5. SECOND PASS: LOAD ALL VALID PIXELS INTO PREALLOCATED ARRAYS
# ============================================================

def load_all_training_pixels(h5_file, combo, features, counts, total):
    n_features = len(features)

    x_gb = total * n_features * 4 / (1024 ** 3)
    y_gb = total * 4 / (1024 ** 3)

    print(f"\nAllocating arrays for {combo}+NDVI")
    print(f"Number of features: {n_features}")
    print(f"X array: approximately {x_gb:.2f} GB")
    print(f"y array: approximately {y_gb:.2f} GB")
    print("Random Forest fitting requires additional RAM.")

    X_train = np.empty((total, n_features), dtype=np.float32)
    y_train = np.empty(total, dtype=np.float32)

    position = 0

    for map_number, expected_count in zip(TRAIN_MAPS, counts):
        X, y = read_map(h5_file, map_number, features)
        mask = make_native_mask(X, y)
        valid_count = int(np.count_nonzero(mask))

        if valid_count != expected_count:
            raise RuntimeError(
                f"Valid-pixel count changed for map {map_number}, "
                f"{combo}+NDVI: first pass={expected_count}, "
                f"second pass={valid_count}. Check whether the MAT file changed."
            )

        end = position + valid_count
        X_train[position:end] = X[mask]
        y_train[position:end] = y[mask]
        position = end

        print(
            f"Loaded map {map_number}: {valid_count:,} pixels "
            f"({position:,}/{total:,})"
        )

        del X, y, mask
        gc.collect()

    if position != total:
        raise RuntimeError(
            f"Filled {position:,} rows, but expected {total:,}."
        )

    return X_train, y_train


# ============================================================
# 6. TRAIN ONE MODEL
# ============================================================

def train_model(h5_file, combo):
    print("\n" + "=" * 65)
    print(f"TRAINING {combo}+NDVI USING ALL VALID TRAINING PIXELS")
    print("=" * 65)

    features = get_features(combo)
    counts, total = count_training_pixels(h5_file, combo, features)

    X_train, y_train = load_all_training_pixels(
        h5_file, combo, features, counts, total
    )

    print("\nTraining array shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        # sklearn defaults for bootstrap and max_features are retained.
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_minutes = (time.time() - start_time) / 60.0

    model_path = MODEL_DIR / f"rf_{combo}_NDVI_all_pixels.joblib"
    joblib.dump(model, model_path)

    print(f"Training time: {training_minutes:.2f} minutes")
    print("Saved model:", model_path)

    del X_train, y_train
    gc.collect()

    return model


# ============================================================
# 7. TEST ONE MODEL ON MAPS 191-200
# ============================================================

def test_model(h5_file, model, combo):
    print(f"\nTesting {combo}+NDVI on maps 191-200 ...")
    features = get_features(combo)
    results = []

    for map_number in TEST_MAPS:
        X, y = read_map(h5_file, map_number, features)
        mask = make_native_mask(X, y)
        valid_indices = np.flatnonzero(mask)
        n_valid = valid_indices.size

        if n_valid == 0:
            raise ValueError(
                f"No valid test pixels in map {map_number} for {combo}+NDVI."
            )

        y_valid = y[valid_indices]
        y_mean = float(np.mean(y_valid))
        sst = float(np.sum((y_valid - y_mean) ** 2))

        sse = 0.0
        error_sum = 0.0

        for start in range(0, n_valid, TEST_BATCH_SIZE):
            idx = valid_indices[start:start + TEST_BATCH_SIZE]
            prediction = model.predict(X[idx])
            error = prediction - y[idx]

            sse += float(np.sum(error ** 2, dtype=np.float64))
            error_sum += float(np.sum(error, dtype=np.float64))

            del prediction, error

        rmse = float(np.sqrt(sse / n_valid))
        bias = float(error_sum / n_valid)
        r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

        results.append({
            "Combination": combo + "+NDVI",
            "Map": map_number,
            "Valid_pixels": int(n_valid),
            "RMSE": rmse,
            "Bias": bias,
            "R2": r2,
        })

        print(
            f"Map {map_number}: Pixels={n_valid:,}, "
            f"RMSE={rmse:.5f}, Bias={bias:.5f}, R2={r2:.4f}"
        )

        del X, y, mask, valid_indices, y_valid
        gc.collect()

    return results


# ============================================================
# 8. MAIN: RUN ALL SEVEN COMBINATIONS
# ============================================================

def main():
    if not Path(DATA_PATH).is_file():
        raise FileNotFoundError(
            f"MAT file not found:\n{DATA_PATH}\n"
            "Edit DATA_PATH at the top of this script before running."
        )

    all_results = []
    total_start = time.time()

    print("ALL-PIXEL RANDOM FOREST EXPERIMENT")
    print(f"Training maps: {TRAIN_MAPS.start}-{TRAIN_MAPS.stop - 1}")
    print(f"Test maps: {TEST_MAPS.start}-{TEST_MAPS.stop - 1}")
    print("Combinations:", COMBINATIONS)
    print("Mask: all selected features and SMC must be finite")
    print("N_JOBS:", N_JOBS)
    print("Model folder:", MODEL_DIR)

    with h5py.File(DATA_PATH, "r") as h5_file:
        for combo in COMBINATIONS:
            combo_start = time.time()

            model = train_model(h5_file, combo)
            all_results.extend(test_model(h5_file, model, combo))

            print(
                f"\nCompleted {combo}+NDVI in "
                f"{(time.time() - combo_start) / 60.0:.2f} minutes."
            )

            del model
            gc.collect()

    df = pd.DataFrame(all_results)
    df.to_csv(PER_MAP_CSV, index=False)

    summary = []

    for combo_name, group in df.groupby("Combination", sort=False):
        n = group["Valid_pixels"].to_numpy(dtype=np.float64)
        total_valid = int(n.sum())

        rmse_values = group["RMSE"].to_numpy(dtype=np.float64)
        bias_values = group["Bias"].to_numpy(dtype=np.float64)

        overall_rmse = float(np.sqrt(np.sum(n * rmse_values ** 2) / total_valid))
        overall_bias = float(np.sum(n * bias_values) / total_valid)

        summary.append({
            "Combination": combo_name,
            "Overall_RMSE": overall_rmse,
            "Overall_Bias": overall_bias,
            "Total_valid_pixels": total_valid,
        })

    summary_df = pd.DataFrame(summary).sort_values("Overall_RMSE")
    summary_df.to_csv(SUMMARY_CSV, index=False)

    total_hours = (time.time() - total_start) / 3600.0

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS -- ALL TRAINING PIXELS")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("\nPer-map results:", PER_MAP_CSV)
    print("Summary results:", SUMMARY_CSV)
    print("Models folder:", MODEL_DIR)
    print(f"Total elapsed time: {total_hours:.2f} hours")
    print("\nAll-pixel training and testing completed successfully!")


if __name__ == "__main__":
    main()
