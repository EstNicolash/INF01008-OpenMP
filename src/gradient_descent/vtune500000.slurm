#!/bin/bash
#SBATCH --job-name=gd_vtune
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "Node:"
hostname

# ===== VTune =====
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh

# ===== Preparar ambiente =====
rm -rf $SCRATCH/gd_project_500000
mkdir -p $SCRATCH/gd_project_500000

cp -r $HOME/advanced_vtuned-LinearRegression-ParallelGradientDescent/* $SCRATCH/gd_project_500000/
cd $SCRATCH/gd_project_500000

# ===== Compilar =====
make release

# ===== Parâmetros =====
N=500000
FINAL_RESULTS=$HOME/gd_results_500000

mkdir -p $FINAL_RESULTS/vtune_csvs

# ===== Execução =====
for t in 1 2 4 8 12 16 20 24 32 40
do
    echo "=============================="
    echo "Running with $t threads"
    echo "=============================="

    export OMP_NUM_THREADS=$t

    PREFIX="t${t}_N${N}"

    R_SNAP="res_snapshot_$PREFIX"
    R_HOT="res_hotspots_$PREFIX"
    R_HPC="res_hpc_$PREFIX"

    # ===== COLETA (com ITT controlando) =====

    vtune -collect performance-snapshot \
        -start-paused \
        -r $R_SNAP \
        -- ./build/bin/benchmark $N $t

    vtune -collect hotspots -knob sampling-mode=hw \
        -start-paused \
        -r $R_HOT \
        -- ./build/bin/benchmark $N $t

    vtune -collect hpc-performance \
        -start-paused \
        -r $R_HPC \
        -- ./build/bin/benchmark $N $t

    # ===== EXPORT CSV =====

    vtune -report summary \
        -r $R_SNAP \
        -format=csv \
        -report-output $FINAL_RESULTS/vtune_csvs/snap_$PREFIX.csv

    vtune -report summary \
        -r $R_HOT \
        -format=csv \
        -report-output $FINAL_RESULTS/vtune_csvs/hot_$PREFIX.csv

    vtune -report summary \
        -r $R_HPC \
        -format=csv \
        -report-output $FINAL_RESULTS/vtune_csvs/hpc_$PREFIX.csv

done

# ===== Salvar resultados completos =====
cp -r res_* $FINAL_RESULTS/

echo "Done!"
