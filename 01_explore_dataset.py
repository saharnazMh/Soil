
import h5py

file_path = r"C:\Users\Saharnaz\Downloads\features_v7.3.mat"

with h5py.File(file_path, "r") as f:
    print("Number of variables:", len(f.keys()))

    print("\nFirst 20 variables:")
    for name in list(f.keys())[:20]:
        print(name)


print("\nChecking dataset shapes:")

with h5py.File(file_path, "r") as f:
    variables = [
        "SMC_1",
        "NDVI_1",
        "LIA_L",
        "LIA_C",
        "LIA_X",
        "sigma0_db_L_vv_1",
        "sigma0_db_L_vh_1",
        "sigma0_db_C_vv_1",
        "sigma0_db_C_vh_1",
        "sigma0_db_X_hh_1"
    ]

    for name in variables:
        if name in f:
            print(name, ":", f[name].shape)
        else:
            print(name, ": Not found")


print("\nChecking pixel values:")

with h5py.File(file_path, "r") as f:
    variables = [
        "SMC_1",
        "NDVI_1",
        "LIA_L",
        "sigma0_db_L_vv_1",
        "sigma0_db_L_vh_1"
    ]

    for name in variables:
        value = f[name][500, 500]
        print(name, ":", value)
