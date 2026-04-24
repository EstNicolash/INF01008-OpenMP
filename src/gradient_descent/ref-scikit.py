import sys
import time
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

def run_sklearn_reference(filename, learning_rate=1e-7, num_iterations=5000):
    print("================================================")
    print("   Reference Gradient Descent - Scikit-Learn    ")
    print("================================================")

    try:
        print(f"Loading dataset: {filename}")
        start_load = time.time()

        df = pd.read_csv(filename, header=None)

        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

        N, D = X.shape

        end_load = time.time()
        print(f"Data loaded in {end_load - start_load:.4f}s")
        print(f"Samples (N): {N}, Features (D): {D}")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    model = SGDRegressor(
        loss='squared_error',
        penalty=None,
        fit_intercept=False,
        max_iter=num_iterations,
        learning_rate='constant',
        eta0=learning_rate,
        tol=None,
        random_state=42
    )

    print(f"Starting Scikit-Learn training... (LR={learning_rate}, Iters={num_iterations})")

    start_fit = time.time()

    model.fit(X, y)

    end_fit = time.time()
    fit_duration = end_fit - start_fit

    beta = model.coef_
    predictions = model.predict(X)
    final_loss = mean_squared_error(y, predictions)

    print("\n--- Execution Summary ---")
    print(f"Fit Execution Time: {fit_duration:.6f} seconds")
    print(f"Final MSE Loss: {final_loss:.6f}")

    print("\nModel Coefficients (first 10):")
    for i, b in enumerate(beta[:10]):
        print(f"  Beta[{i}]: {b:10.6f}")
    print("================================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ref_sklearn.py <dataset_path.csv>")
    else:
        run_sklearn_reference(sys.argv[1])
