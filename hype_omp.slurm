#!/bin/bash
#SBATCH --job-name=vtune_benchmark_suite
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh

rm -rf $SCRATCH/projeto-omp
mkdir -p $SCRATCH/projeto-omp/data
cp -r $HOME/projeto-omp/src $SCRATCH/projeto-omp/
cp $HOME/projeto-omp/data/*.csv $SCRATCH/projeto-omp/data/
cd $SCRATCH/projeto-omp

g++ -O3 -g -march=native -fopenmp \
    -DUSE_ITT -DSILENT \
    src/normal_equations/main.cpp src/normal_equations/linear_regression.cpp \
    -I /home/intel/oneapi/vtune/2021.1.1/include \
    -L /home/intel/oneapi/vtune/2021.1.1/lib64 -littnotify -ldl \
    -o normal_equations.out

DATASETS=("medium_bench.csv" "tiny_test.csv" "standard_mid.csv" "big_heavy.csv" "year_prediction_msd.csv" "huge_stress.csv" "large_stress.csv")
THREADS=(1 2 4 8 12 16 20 24 32 40)

FINAL_RESULTS="$HOME/projeto-omp/results_$(date +%Y%m%d_%H%M)"
mkdir -p $FINAL_RESULTS/vtune_csvs

for DS in "${DATASETS[@]}"; do
    DS_PATH="data/$DS"
    DS_NAME=$(basename $DS .csv)

    for T in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$T
        echo ">>> Rodando: Dataset=$DS_NAME | Threads=$T"

        PREFIX="${DS_NAME}_t${T}"
        R_SNAP="res_snap_$PREFIX"
        R_HOT="res_hot_$PREFIX"
        R_HPC="res_hpc_$PREFIX"

        vtune -collect performance-snapshot -r $R_SNAP -start-paused -- ./normal_equations.out $DS_PATH
        vtune -collect hotspots -knob sampling-mode=hw -r $R_HOT -start-paused -- ./normal_equations.out $DS_PATH
        vtune -collect hpc-performance -r $R_HPC -start-paused -- ./normal_equations.out $DS_PATH

        vtune -report summary -r $R_SNAP -format=csv -report-output $FINAL_RESULTS/vtune_csvs/snap_$PREFIX.csv
        vtune -report summary -r $R_HOT -format=csv -report-output $FINAL_RESULTS/vtune_csvs/hot_$PREFIX.csv
        vtune -report summary -r $R_HPC -format=csv -report-output $FINAL_RESULTS/vtune_csvs/hpc_$PREFIX.csv

    done
done

cp bench_results.csv $FINAL_RESULTS/app_bench_results.csv

echo "Benchmark finalizado. Resultados em: $FINAL_RESULTS"
