#!/bin/bash
#SBATCH --job-name=vtune_full_analysis
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
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

export OMP_NUM_THREADS=20
DATASET="data/year_prediction_msd.csv"


vtune -collect performance-snapshot -r res_snapshot -- ./normal_equations.out $DATASET

vtune -collect hotspots -knob sampling-mode=hw -r res_hotspots -- ./normal_equations.out $DATASET
vtune -collect hpc-performance -r res_hpc -- ./normal_equations.out $DATASET

mkdir -p $HOME/projeto-omp/results
cp -r res_snapshot res_hotspots res_hpc $HOME/projeto-omp/results/
