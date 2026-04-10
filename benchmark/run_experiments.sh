#!/bin/bash


# 1. Path Configuration
EXECUTABLE="./build/normal_equations"
LOG_FILE="bench_results_$(date +%Y%m%d_%H%M).csv"
DATA_DIR="./data"

DATASETS=(
    "standard_mid.csv"
    "huge_stress.csv"
    "medium_bench.csv"
    "large_stress.csv"
    "year_prediction_msd.csv"
    "tiny_test.csv"
)

THREAD_COUNTS=(1 2 4 8 12 16 20 24 28 32 36 40)

export OMP_SCHEDULE="static"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Binary not found at $EXECUTABLE. Please run 'make' first."
    exit 1
fi

echo "------------------------------------------------"
echo "Starting performance data collection..."
echo "Output will be saved to: $LOG_FILE"
echo "------------------------------------------------"

for ds in "${DATASETS[@]}"; do
    FILEPATH="$DATA_DIR/$ds"

    if [ ! -f "$FILEPATH" ]; then
        echo "Warning: Dataset $FILEPATH not found. Skipping..."
        continue
    fi

    echo "Benchmarking Dataset: $ds"

    for threads in "${THREAD_COUNTS[@]}"; do
        echo "  -> Running with $threads thread(s)..."
        OMP_NUM_THREADS=$threads $EXECUTABLE "$FILEPATH" "$LOG_FILE"

        sleep 1
    done
    echo "Done with $ds."
    echo "------------------------------------------------"
done

echo "Experiments completed successfully!"
echo "You can now import '$LOG_FILE' into Pandas or Excel for analysis."
