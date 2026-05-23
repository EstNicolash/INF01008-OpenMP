#!/bin/bash
#SBATCH --job-name=vtune_benchmark_suite
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Carrega as variáveis de ambiente do Intel VTune
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh

# Prepara o diretório de execução no SCRATCH do cluster
rm -rf $SCRATCH/projeto-omp
mkdir -p $SCRATCH/projeto-omp/data
cp -r $HOME/projeto-omp/src $SCRATCH/projeto-omp/
cp $HOME/projeto-omp/data/*.csv $SCRATCH/projeto-omp/data/
cd $SCRATCH/projeto-omp

# Compilação com OpenMP (-fopenmp) e otimização máxima (-O3)
g++ -O3 -g -march=native -fopenmp \
    -DUSE_ITT -DSILENT \
    src/normal_equations/main.cpp src/normal_equations/linear_regression.cpp \
    -I /home/intel/oneapi/vtune/2021.1.1/include \
    -L /home/intel/oneapi/vtune/2021.1.1/lib64 -littnotify -ldl \
    -o normal_equations.out

# Crossover é o único leve o bastante para rodar sem timeout com poucas threads
DATASETS_NORMAL=("crossover_search")

# Casos críticos: com poucas threads (1, 2, 4), a inversão O(D³) vai estourar o tempo útil
DATASETS_TIMED=("fat_matrix_trap" "memory_killer" "gd_wins_ne")

TIMEOUT_SECS=30 # 3 minutos de teto por coletor para proteger o escalonamento

THREADS=(1 2 4 8 12 16 20 24 32 40)
FINAL_RESULTS="$HOME/projeto-omp/results_$(date +%Y%m%d_%H%M)"
mkdir -p $FINAL_RESULTS/vtune_csvs

# Função principal de coleta de métricas
run_vtune() {
    local DS_PATH=$1
    local DS_NAME=$2
    local T=$3
    local USE_TIMEOUT=$4

    export OMP_NUM_THREADS=$T
    echo ">>> Rodando: Dataset=$DS_NAME | Threads=$T | Timeout=${USE_TIMEOUT}s (0=sem limite)"

    local PREFIX="${DS_NAME}_t${T}"
    local CMD_PREFIX=""
    if [ "$USE_TIMEOUT" -gt 0 ]; then
        CMD_PREFIX="timeout ${USE_TIMEOUT}"
    fi

    for COLLECTOR in "performance-snapshot" "hotspots -knob sampling-mode=hw" "hpc-performance"; do
        case $COLLECTOR in
            "performance-snapshot") TYPE="snap" ;;
            "hotspots"*)            TYPE="hot"  ;;
            "hpc-performance")      TYPE="hpc"  ;;
        esac

        RDIR="res_${TYPE}_${PREFIX}"

        # Executa o binário apontando corretamente para o arquivo .csv
        $CMD_PREFIX vtune -collect $COLLECTOR -r $RDIR -start-paused \
            -- ./normal_equations.out $DS_PATH

        EXIT=$?
        if [ $EXIT -eq 124 ]; then
            echo "⚠️  TIMEOUT atingido: $DS_NAME | Threads=$T | Coletor=$TYPE — pulando relatório"
            continue
        fi

        # Só gera o relatório CSV se a coleta terminou com sucesso
        vtune -report summary -r $RDIR -format=csv \
            -report-output $FINAL_RESULTS/vtune_csvs/${TYPE}_${PREFIX}.csv
    done
}

# 1. Execução do loop estável (Sem limite de tempo)
for DS in "${DATASETS_NORMAL[@]}"; do
    for T in "${THREADS[@]}"; do
        run_vtune "data/${DS}.csv" "${DS}" $T 0
    done
done

# 2. Execução do loop crítico (Protegido por limite de tempo)
for DS in "${DATASETS_TIMED[@]}"; do
    for T in "${THREADS[@]}"; do
        run_vtune "data/${DS}.csv" "${DS}" $T $TIMEOUT_SECS
    done
done

# Salvaguarda para cópia do arquivo de resultados do benchmark
if [ -f "bench_results.csv" ]; then
    cp bench_results.csv $FINAL_RESULTS/app_bench_results.csv
fi

echo "Benchmark finalizado. Resultados em: $FINAL_RESULTS"
