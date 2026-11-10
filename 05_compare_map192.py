
import h5py
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

file_path = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

# Load the already trained models
rf_l = joblib.load("models/rf_L_NDVI.joblib")
rf_lcx = joblib.load("models/rf_LCX_NDVI.joblib")

map_number = 192

features_l = [
    "NDVI",
    "LIA_L",
    "sigma0_db_L_vv",
    "sigma0_db_L_vh"
]

features_lcx = [
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

def read_features(f, features, map_number):
    columns = []

    for feature in features:
        if feature.startswith("LIA_"):
            name = feature
        else:
            name = f"{feature}_{map_number}"

        columns.append(f[name][:].ravel())

    return np.column_stack(columns)

# Read the same test map for both models
with h5py.File(file_path, "r") as f:
    X_l = read_features(f, features_l, map_number)
    X_lcx = read_features(f, features_lcx, map_number)
    y_true = f[f"SMC_{map_number}"][:].ravel()

# Common valid-pixel mask
common_mask = (
    np.isfinite(X_l).all(axis=1)
    & np.isfinite(X_lcx).all(axis=1)
    & np.isfinite(y_true)
)

X_l_valid = X_l[common_mask]
X_lcx_valid = X_lcx[common_mask]
y_valid = y_true[common_mask]

print("Common valid pixels:", len(y_valid))

# Predict using both saved models
pred_l = rf_l.predict(X_l_valid)
pred_lcx = rf_lcx.predict(X_lcx_valid)

# Calculate metrics
for name, prediction in [
    ("L + NDVI", pred_l),
    ("LCX + NDVI", pred_lcx)
]:
    rmse = np.sqrt(mean_squared_error(y_valid, prediction))
    bias = np.mean(prediction - y_valid)
    r2 = r2_score(y_valid, prediction)

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("Bias:", bias)
    print("R2:", r2)


    
# Check NaN values in Map 192

print("\nNaN analysis for Map 192:")

with h5py.File(file_path, "r") as f:

    for feature in features_lcx:

        if feature.startswith("LIA_"):
            variable_name = feature
        else:
            variable_name = f"{feature}_{map_number}"

        values = f[variable_name][:]

        nan_count = np.isnan(values).sum()

        print(f"{feature}: {nan_count} NaN pixels")

    # Check SMC target
    smc = f[f"SMC_{map_number}"][:]
    print("SMC:", np.isnan(smc).sum(), "NaN pixels")

