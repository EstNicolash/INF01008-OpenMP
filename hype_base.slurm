#!/bin/bash
#SBATCH --job-name=vtune_sequential_baselines
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


g++ -O0 -g \
    -DUSE_ITT -DSILENT \
    src/normal_equations/main.cpp src/normal_equations/linear_regression.cpp \
    -I /home/intel/oneapi/vtune/2021.1.1/include \
    -L /home/intel/oneapi/vtune/2021.1.1/lib64 -littnotify -ldl -lgomp \
    -o seq_raw.out

g++ -O3 -g -march=native \
    -DUSE_ITT -DSILENT \
    src/normal_equations/main.cpp src/normal_equations/linear_regression.cpp \
    -I /home/intel/oneapi/vtune/2021.1.1/include \
    -L /home/intel/oneapi/vtune/2021.1.1/lib64 -littnotify -ldl -lgomp \
    -o seq_opt.out


DATASETS=("medium_bench.csv" "tiny_test.csv" "standard_mid.csv" "big_heavy.csv" "year_prediction_msd.csv" "huge_stress.csv" "large_stress.csv")

FINAL_RESULTS="$HOME/projeto-omp/baselines_$(date +%Y%m%d_%H%M)"
mkdir -p $FINAL_RESULTS/vtune_csvs

for DS in "${DATASETS[@]}"; do
    DS_PATH="data/$DS"
    DS_NAME=$(basename $DS .csv)

    echo ">>> Rodando Baselines para Dataset=$DS_NAME"

    # ---------------------------------------------------------
    # TESTE 1: CÓDIGO RAW (O0, SEM SIMD, SEM THREADS)
    # ---------------------------------------------------------
    PREFIX_RAW="raw_${DS_NAME}"
    R_SNAP_RAW="res_snap_$PREFIX_RAW"
    R_HOT_RAW="res_hot_$PREFIX_RAW"
    R_HPC_RAW="res_hpc_$PREFIX_RAW"

    vtune -collect performance-snapshot -r $R_SNAP_RAW -start-paused -- ./seq_raw.out $DS_PATH
    vtune -collect hotspots -knob sampling-mode=hw -r $R_HOT_RAW -start-paused -- ./seq_raw.out $DS_PATH
    vtune -collect hpc-performance -r $R_HPC_RAW -start-paused -- ./seq_raw.out $DS_PATH

    vtune -report summary -r $R_SNAP_RAW -format=csv -report-output $FINAL_RESULTS/vtune_csvs/snap_$PREFIX_RAW.csv
    vtune -report summary -r $R_HOT_RAW -format=csv -report-output $FINAL_RESULTS/vtune_csvs/hot_$PREFIX_RAW.csv
    vtune -report summary -r $R_HPC_RAW -format=csv -report-output $FINAL_RESULTS/vtune_csvs/hpc_$PREFIX_RAW.csv

    # ---------------------------------------------------------
    # TESTE 2: CÓDIGO OPT (O3, COM SIMD, SEM THREADS)
    # ---------------------------------------------------------
    PREFIX_OPT="opt_${DS_NAME}"
    R_SNAP_OPT="res_snap_$PREFIX_OPT"
    R_HOT_OPT="res_hot_$PREFIX_OPT"
    R_HPC_OPT="res_hpc_$PREFIX_OPT"

    vtune -collect performance-snapshot -r $R_SNAP_OPT -start-paused -- ./seq_opt.out $DS_PATH
    vtune -collect hotspots -knob sampling-mode=hw -r $R_HOT_OPT -start-paused -- ./seq_opt.out $DS_PATH
    vtune -collect hpc-performance -r $R_HPC_OPT -start-paused -- ./seq_opt.out $DS_PATH

    vtune -report summary -r $R_SNAP_OPT -format=csv -report-output $FINAL_RESULTS/vtune_csvs/snap_$PREFIX_OPT.csv
    vtune -report summary -r $R_HOT_OPT -format=csv -report-output $FINAL_RESULTS/vtune_csvs/hot_$PREFIX_OPT.csv
    vtune -report summary -r $R_HPC_OPT -format=csv -report-output $FINAL_RESULTS/vtune_csvs/hpc_$PREFIX_OPT.csv

done

cp bench_results.csv $FINAL_RESULTS/app_bench_results.csv

echo "Baselines finalizados. Resultados em: $FINAL_RESULTS"
