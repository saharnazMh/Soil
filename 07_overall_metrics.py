
# Calculate overall RMSE across all test maps

import pandas as pd
import numpy as np

# Read the saved test results
df = pd.read_csv("RF_LCX_NDVI_test_results.csv")

# Total number of valid pixels
total_pixels = df["Valid_pixels"].sum()

# Weighted mean of squared errors
overall_mse = np.sum(
    df["Valid_pixels"] * df["RMSE"]**2
) / total_pixels

# Overall RMSE
overall_rmse = np.sqrt(overall_mse)

# Overall Bias
overall_bias = np.sum(
    df["Valid_pixels"] * df["Bias"]
) / total_pixels

print("\nOverall Test Results (Maps 191-200)")
print("Total valid pixels:", total_pixels)
print("Overall RMSE:", overall_rmse)
print("Overall Bias:", overall_bias)
