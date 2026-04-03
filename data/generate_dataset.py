import numpy as np
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_dataset(name, rows, cols):
    """
    Generates a synthetic dataset for multiple linear regression.
    The first column is the target (y), followed by 'cols' features (X).
    """
    print(f"Generating {name} ({rows} samples, {cols} features)...")

    X = np.random.rand(rows, cols).astype(np.float32)

    true_weights = np.random.rand(cols, 1).astype(np.float32)

    noise = np.random.normal(0, 0.01, (rows, 1)).astype(np.float32)
    y = X @ true_weights + noise

    data = np.hstack((y, X))

    file_path = os.path.join(DATA_DIR, f"{name}.csv")

    np.savetxt(file_path, data, delimiter=",", fmt="%.6f")

    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Successfully saved to: {file_path}")
    print(f"Size: {file_size:.2f} MB\n")

if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    print("-" * 30)

    generate_dataset("tiny_test", 10000, 15)

    generate_dataset("medium_bench", 100000, 70)

    generate_dataset("large_stress", 1000000, 120)

    # generate_dataset("extreme_mmap", 5000000, 120)

    print("All synthetic datasets generated successfully!")
