import numpy as np
import pandas as pd
import time
import sys

def run_gradient_descent_reference(filename, learning_rate=0.1, num_iterations=5000):
    print("================================================")
    print("   Reference Gradient Descent - NumPy Version   ")
    print("================================================")

    try:
        print(f"Loading dataset: {filename}")
        start_load = time.time()
        df = pd.read_csv(filename, header=None)

        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values.astype(np.float64)

        N, D = X.shape

        end_load = time.time()
        print(f"Data loaded in {end_load - start_load:.4f}s")
        print(f"Samples (N): {N}, Features (D): {D}")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Normalizing features...")
    start_norm = time.time()

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)

    std_dev[std_dev < 1e-8] = 1.0

    X = (X - mean) / std_dev

    end_norm = time.time()
    print(f"Normalization Time: {end_norm - start_norm:.4f}s")

    print(f"Starting model training... (LR={learning_rate}, Iters={num_iterations})")

    beta = np.zeros(D, dtype=np.float64)

    start_fit = time.time()

    for i in range(num_iterations):
        predictions = X @ beta
        errors = predictions - y
        gradients = (2.0 / N) * (X.T @ errors)
        beta -= learning_rate * gradients

    end_fit = time.time()

    final_loss = np.mean((X @ beta - y) ** 2)

    print("\n--- Execution Summary ---")
    print(f"Fit Execution Time: {end_fit - start_fit:.6f} seconds")
    print(f"Final MSE Loss: {final_loss:.6f}")

    print("\nModel Coefficients (showing first 10):")
    for i, b in enumerate(beta[:10]):
        print(f"  Beta[{i}]: {b:10.6f}")
    print("  ...")
    print("================================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ref.py <dataset_path.csv>")
    else:
        run_gradient_descent_reference(sys.argv[1])
