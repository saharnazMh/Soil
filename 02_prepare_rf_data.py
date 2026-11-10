
import h5py
import numpy as np
import pandas as pd

file_path = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

with h5py.File(file_path, "r") as f:

    # Read map number 1
    ndvi = f["NDVI_1"][:]
    lia = f["LIA_L"][:]
    vv = f["sigma0_db_L_vv_1"][:]
    vh = f["sigma0_db_L_vh_1"][:]
    smc = f["SMC_1"][:]

# Create a table from the maps
df = pd.DataFrame({
    "NDVI": ndvi.ravel(),
    "LIA_L": lia.ravel(),
    "Sigma0_L_VV": vv.ravel(),
    "Sigma0_L_VH": vh.ravel(),
    "SMC": smc.ravel()
})

# Remove invalid pixels
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Number of valid pixels:", len(df))
print("\nFirst 10 rows:")
print(df.head(10))


# Define feature combinations for Random Forest

feature_groups = {
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

# Seven combinations, all including NDVI
combinations = [
    "L", "C", "X", "LCX", "LC", "LX", "CX"
]

for combo in combinations:
    selected_features = ["NDVI"]

    for band in combo:
        selected_features.extend(feature_groups[band])

    print(f"\nCombination: {combo} + NDVI")
    print("Number of features:", len(selected_features))
    print("Features:", selected_features)



# Select the first experiment
combo = "L"
map_number = 1

# Build the list of input features
selected_features = ["NDVI"]

for band in combo:
    selected_features.extend(feature_groups[band])

print("\nSelected combination:", combo, "+ NDVI")

# Read the selected features from MATLAB
with h5py.File(file_path, "r") as f:

    data = {}

    for feature in selected_features:

        if feature.startswith("LIA_"):
            variable_name = feature
        else:
            variable_name = f"{feature}_{map_number}"

        data[feature] = f[variable_name][:].ravel()

    # Read the target SMC
    data["SMC"] = f[f"SMC_{map_number}"][:].ravel()

# Create the DataFrame
df_selected = pd.DataFrame(data)

# Remove invalid pixels
df_selected = df_selected.replace(
    [np.inf, -np.inf], np.nan
).dropna()

print("Selected features:", selected_features)
print("Number of valid pixels:", len(df_selected))
print(df_selected.head())

# Select the first experiment
combo = "L"
map_number = 1

# Build the list of input features
selected_features = ["NDVI"]

for band in combo:
    selected_features.extend(feature_groups[band])

print("\nSelected combination:", combo, "+ NDVI")

# Read the selected features from MATLAB
with h5py.File(file_path, "r") as f:

    data = {}

    for feature in selected_features:

        if feature.startswith("LIA_"):
            variable_name = feature
        else:
            variable_name = f"{feature}_{map_number}"

        data[feature] = f[variable_name][:].ravel()

    # Read the target SMC
    data["SMC"] = f[f"SMC_{map_number}"][:].ravel()

# Create the DataFrame
df_selected = pd.DataFrame(data)

# Remove invalid pixels
df_selected = df_selected.replace(
    [np.inf, -np.inf], np.nan
).dropna()

print("Selected features:", selected_features)
print("Number of valid pixels:", len(df_selected))
print(df_selected.head())

# Select the first experiment
combo = "L"
map_number = 1

# Build the list of input features
selected_features = ["NDVI"]

for band in combo:
    selected_features.extend(feature_groups[band])

print("\nSelected combination:", combo, "+ NDVI")

# Read the selected features from MATLAB
with h5py.File(file_path, "r") as f:

    data = {}

    for feature in selected_features:

        if feature.startswith("LIA_"):
            variable_name = feature
        else:
            variable_name = f"{feature}_{map_number}"

        data[feature] = f[variable_name][:].ravel()

    # Read the target SMC
    data["SMC"] = f[f"SMC_{map_number}"][:].ravel()

# Create the DataFrame
df_selected = pd.DataFrame(data)

# Remove invalid pixels
df_selected = df_selected.replace(
    [np.inf, -np.inf], np.nan
).dropna()

print("Selected features:", selected_features)
print("Number of valid pixels:", len(df_selected))
print(df_selected.head())

# Select the first experiment
combo = "L"
map_number = 1

# Build the list of input features
selected_features = ["NDVI"]

for band in combo:
    selected_features.extend(feature_groups[band])

print("\nSelected combination:", combo, "+ NDVI")

# Read the selected features from MATLAB
with h5py.File(file_path, "r") as f:

    data = {}

    for feature in selected_features:

        if feature.startswith("LIA_"):
            variable_name = feature
        else:
            variable_name = f"{feature}_{map_number}"

        data[feature] = f[variable_name][:].ravel()

    # Read the target SMC
    data["SMC"] = f[f"SMC_{map_number}"][:].ravel()

# Create the DataFrame
df_selected = pd.DataFrame(data)

# Remove invalid pixels
df_selected = df_selected.replace(
    [np.inf, -np.inf], np.nan
).dropna()

print("Selected features:", selected_features)
print("Number of valid pixels:", len(df_selected))
print(df_selected.head())
