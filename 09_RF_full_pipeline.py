
import h5py
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

FILE_PATH = (
    r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"
)

MODEL_DIR = PROJECT_DIR / "models_full_pipeline"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_MAPS = range(1, 101)
VAL_MAPS = range(101, 121)
TEST_MAPS = range(191, 201)

SAMPLES_PER_MAP = 1000
BATCH_SIZE = 100000
RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 15


# ==========================================
# 2. DEFINE FEATURES
# ==========================================

GROUPS = {
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

COMBINATIONS = [
    "L", "C", "X", "LCX", "LC", "LX", "CX"
]


def get_features(combo):
    features = ["NDVI"]

    for band in combo:
        features.extend(GROUPS[band])

    return features


ALL_FEATURES = get_features("LCX")

FEATURE_INDICES = {
    combo: [
        ALL_FEATURES.index(name)
        for name in get_features(combo)
    ]
    for combo in COMBINATIONS
}


# ==========================================
# 3. READ DATASET
# ==========================================

def read_map(f, map_number):

    columns = []

    for feature in ALL_FEATURES:

        if feature.startswith("LIA_"):
            variable_name = feature
        else:
            variable_name = f"{feature}_{map_number}"

        values = f[variable_name][:].ravel()
        columns.append(values)

    X = np.column_stack(columns)

    y = f[f"SMC_{map_number}"][:].ravel()

    return X, y


def make_common_mask(X, y):

    return (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y)
    )


# ==========================================
# 4. PREPARE TRAINING DATA
# ==========================================

def prepare_training_data():

    print("\nPreparing training data...")

    rng = np.random.default_rng(RANDOM_STATE)

    X_lists = {
        combo: [] for combo in COMBINATIONS
    }

    y_list = []

    with h5py.File(FILE_PATH, "r") as f:

        for map_number in TRAIN_MAPS:

            X_map, y_map = read_map(f, map_number)

            # Common mask for all nine features
            valid_mask = make_common_mask(
                X_map, y_map
            )

            valid_indices = np.flatnonzero(valid_mask)

            sample_size = min(
                SAMPLES_PER_MAP,
                len(valid_indices)
            )

            if sample_size == 0:
                raise ValueError(
                    f"No valid pixels in map {map_number}"
                )

            chosen = rng.choice(
                valid_indices,
                size=sample_size,
                replace=False
            )

            # Same training pixels for all models
            X_sample = X_map[chosen]
            y_sample = y_map[chosen]

            y_list.append(y_sample)

            for combo in COMBINATIONS:

                cols = FEATURE_INDICES[combo]

                X_lists[combo].append(
                    X_sample[:, cols]
                )

            print(
                f"Training map {map_number}: "
                f"{sample_size} pixels selected"
            )

    y_train = np.concatenate(y_list)

    X_train_dict = {}

    for combo in COMBINATIONS:

        X_train_dict[combo] = np.vstack(
            X_lists[combo]
        )

    print("\nTraining target:", y_train.shape)

    return X_train_dict, y_train


# ==========================================
# 5. TRAIN ALL SEVEN MODELS
# ==========================================

def train_all_models():

    X_train_dict, y_train = prepare_training_data()

    models = {}

    for combo in COMBINATIONS:

        print(f"\nTraining RF: {combo} + NDVI")

        X_train = X_train_dict[combo]

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

        rf_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        model_path = (
            MODEL_DIR / f"rf_{combo}_NDVI.joblib"
        )

        joblib.dump(rf_model, model_path)

        models[combo] = rf_model

        print(f"Model saved: {model_path.name}")

    return models


# ==========================================
# 6. VALIDATION / TEST
# ==========================================

def evaluate_models(models, map_range, split_name):

    print(f"\nEvaluating {split_name}...")

    results = []

    with h5py.File(FILE_PATH, "r") as f:

        for map_number in map_range:

            X_all, y_true = read_map(
                f, map_number
            )

            # Same mask for every combination
            common_mask = make_common_mask(
                X_all, y_true
            )

            valid_indices = np.flatnonzero(
                common_mask
            )

            n_valid = len(valid_indices)

            if n_valid == 0:
                raise ValueError(
                    f"No valid pixels in map {map_number}"
                )

            print(
                f"\n{split_name} Map {map_number}: "
                f"{n_valid} valid pixels"
            )

            # Ground-truth statistics for R2
            y_valid = y_true[valid_indices]

            y_mean = np.mean(y_valid)

            sst = np.sum(
                (y_valid - y_mean) ** 2
            )

            for combo in COMBINATIONS:

                model = models[combo]
                cols = FEATURE_INDICES[combo]

                sse = 0.0
                error_sum = 0.0

                # Predict in batches to save RAM
                for start in range(
                    0, n_valid, BATCH_SIZE
                ):

                    batch_indices = valid_indices[
                        start:start + BATCH_SIZE
                    ]

                    X_batch = X_all[
                        np.ix_(batch_indices, cols)
                    ]

                    y_batch = y_true[batch_indices]

                    prediction = model.predict(
                        X_batch
                    )

                    error = prediction - y_batch

                    sse += np.sum(error ** 2)

                    error_sum += np.sum(error)

                rmse = np.sqrt(sse / n_valid)

                bias = error_sum / n_valid

                r2 = (
                    1 - sse / sst
                    if sst > 0
                    else np.nan
                )

                results.append({
                    "Split": split_name,
                    "Combination": combo + "+NDVI",
                    "Map": map_number,
                    "Valid_pixels": n_valid,
                    "SSE": sse,
                    "SST": sst,
                    "Error_sum": error_sum,
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

    return pd.DataFrame(results)


# ==========================================
# 7. CALCULATE OVERALL RESULTS
# ==========================================

def summarize_results(df):

    summary = []

    for combo, group in df.groupby(
        "Combination", sort=False
    ):

        total_n = group["Valid_pixels"].sum()

        total_sse = group["SSE"].sum()

        total_error = group["Error_sum"].sum()

        overall_rmse = np.sqrt(
            total_sse / total_n
        )

        overall_bias = total_error / total_n

        summary.append({
            "Combination": combo,
            "Overall_RMSE": overall_rmse,
            "Overall_Bias": overall_bias,
            "Total_valid_pixels": int(total_n)
        })

    return pd.DataFrame(summary)


# ==========================================
# 8. MAIN PIPELINE
# ==========================================

if __name__ == "__main__":

    # STEP 1: TRAINING
    print("\n========== TRAINING ==========")

    models = train_all_models()

    # STEP 2: VALIDATION
    print("\n========== VALIDATION ==========")

    val_results = evaluate_models(
        models, VAL_MAPS, "Validation"
    )

    val_summary = summarize_results(val_results)

    val_summary = val_summary.sort_values(
        "Overall_RMSE"
    ).reset_index(drop=True)

    print("\nVALIDATION SUMMARY:")
    print(val_summary.to_string(index=False))

    # Select model based ONLY on validation
    best_combo = val_summary.iloc[0]["Combination"]

    print("\nSelected model:", best_combo)

    # STEP 3: TEST
    print("\n========== TEST ==========")

    test_results = evaluate_models(
        models, TEST_MAPS, "Test"
    )

    test_summary = summarize_results(test_results)

    print("\nTEST SUMMARY:")
    print(test_summary.to_string(index=False))

    # STEP 4: SAVE RESULTS
    val_results.to_csv(
        PROJECT_DIR / "RF_validation_per_map.csv",
        index=False
    )

    val_summary.to_csv(
        PROJECT_DIR / "RF_validation_summary.csv",
        index=False
    )

    test_results.to_csv(
        PROJECT_DIR / "RF_test_per_map.csv",
        index=False
    )

    test_summary.to_csv(
        PROJECT_DIR / "RF_test_summary.csv",
        index=False
    )

    # STEP 5: FINAL COMPARISON TABLE
    comparison = val_summary[
        ["Combination", "Overall_RMSE"]
    ].rename(
        columns={
            "Overall_RMSE": "Validation_RMSE"
        }
    )

    comparison = comparison.merge(
        test_summary[
            ["Combination", "Overall_RMSE"]
        ].rename(
            columns={
                "Overall_RMSE": "Test_RMSE"
            }
        ),
        on="Combination"
    )

    comparison.to_csv(
        PROJECT_DIR / "RF_final_comparison.csv",
        index=False
    )

    print("\n========== FINAL COMPARISON ==========")
    print(comparison.to_string(index=False))

    print("\nBest combination by Validation:")
    print(best_combo)

    print("\nFull pipeline completed successfully!")
