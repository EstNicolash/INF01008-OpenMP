import numpy as np
import pandas as pd
import time
import sys

def run_reference(filename):
    print("================================================")
    print("   Reference Linear Regression - NumPy Version  ")
    print("================================================")

    # 1. Loading Data
    try:
        print(f"Loading dataset: {filename}")
        start_load = time.time()

        df = pd.read_csv(filename, header=None)

        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

        end_load = time.time()
        print(f"Data loaded in {end_load - start_load:.4f}s")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Training (Normal Equation)
    # Equation: beta = (X^T * X)^-1 * X^T * y
    print("Starting model training...")
    start_fit = time.time()


    # Step A: XtX = X.T @ X
    xtx = X.T @ X

    # Step B: Xty = X.T @ y
    xty = X.T @ y

    # beta = np.linalg.inv(xtx) @ xty
    beta = np.linalg.solve(xtx, xty)

    end_fit = time.time()

    print("\n--- Execution Summary ---")
    print(f"Fit Execution Time: {end_fit - start_fit:.6f} seconds")

    print("\nModel Coefficients (showing first 10):")
    for i, b in enumerate(beta[:10]):
        print(f"  Beta[{i}]: {b:10.6f}")
    print("  ...")
    print("================================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ref.py <dataset_path.csv>")
    else:
        run_reference(sys.argv[1])
