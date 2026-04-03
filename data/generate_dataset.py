import numpy as np
import pandas as pd
import os
import time

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_dataset(name, rows, cols):
    print(f"--- Generating {name} ({rows}x{cols}) ---")
    start_time = time.time()

    X = np.random.rand(rows, cols).astype(np.float32)
    true_weights = np.random.rand(cols, 1).astype(np.float32)
    noise = np.random.normal(0, 0.01, (rows, 1)).astype(np.float32)

    y = X @ true_weights + noise

    df = pd.DataFrame(X)
    df.insert(0, 'y', y)

    file_path = os.path.join(DATA_DIR, f"{name}.csv")

    df.to_csv(file_path, index=False, header=False, float_format='%.6f')

    end_time = time.time()
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Finished in: {end_time - start_time:.2f}s")
    print(f"Size: {file_size:.2f} MB\n")

if __name__ == "__main__":
    datasets = [
        ("tiny_test", 10000, 15),       # ~1.7 MB
        ("medium_bench", 100000, 70),   # ~55 MB
        ("standard_mid", 250000, 80),   # ~160 MB
        ("big_heavy", 500000, 100),     # ~400 MB
        ("huge_stress", 750000, 110),   # ~650 MB
        ("large_stress", 1000000, 120), # ~1 GB+
    ]

    for name, r, c in datasets:
        generate_dataset(name, r, c)
