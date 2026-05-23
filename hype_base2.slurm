#!/bin/bash
#SBATCH --job-name=vtune_sequential_baselines
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Carrega as variáveis do VTune
source /home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh

# Prepara o diretório de execução no SCRATCH
rm -rf $SCRATCH/projeto-omp
mkdir -p $SCRATCH/projeto-omp/data
cp -r $HOME/projeto-omp/src $SCRATCH/projeto-omp/
cp $HOME/projeto-omp/data/*.csv $SCRATCH/projeto-omp/data/
cd $SCRATCH/projeto-omp

# Compilação 1: Código RAW (O0, sem otimizações, sequencial)
g++ -O0 -g \
    -DUSE_ITT -DSILENT \
    src/normal_equations/main.cpp src/normal_equations/linear_regression.cpp \
    -I /home/intel/oneapi/vtune/2021.1.1/include \
    -L /home/intel/oneapi/vtune/2021.1.1/lib64 -littnotify -ldl -lgomp \
    -o seq_raw.out

# Compilação 2: Código OPT (O3, com SIMD, sequencial nativo)
g++ -O3 -g -march=native \
    -DUSE_ITT -DSILENT \
    src/normal_equations/main.cpp src/normal_equations/linear_regression.cpp \
    -I /home/intel/oneapi/vtune/2021.1.1/include \
    -L /home/intel/oneapi/vtune/2021.1.1/lib64 -littnotify -ldl -lgomp \
    -o seq_opt.out

# Separação estratégica de datasets para evitar o travamento do Job
DATASETS_NORMAL=("crossover_search")
DATASETS_TIMED=("fat_matrix_trap" "memory_killer" "gd_wins_ne")
TIMEOUT_SECS=180

# Cria a pasta de resultados finais no seu HOME
FINAL_RESULTS="$HOME/projeto-omp/baselines_$(date +%Y%m%d_%H%M)"
mkdir -p $FINAL_RESULTS/vtune_csvs

# Função mestre de execução dos benchmarks
run_baseline() {
    local DS_PATH=$1
    local DS_NAME=$2
    local USE_TIMEOUT=$3

    echo ">>> Rodando Baselines para Dataset=$DS_NAME | Timeout=${USE_TIMEOUT}s (0=sem limite)"

    local CMD_PREFIX=""
    if [ "$USE_TIMEOUT" -gt 0 ]; then
        CMD_PREFIX="timeout ${USE_TIMEOUT}"
    fi

    for VARIANT in "raw" "opt"; do
        local BIN="seq_${VARIANT}.out"
        local PREFIX="${VARIANT}_${DS_NAME}"

        for COLLECTOR in "performance-snapshot" "hotspots -knob sampling-mode=hw" "hpc-performance"; do
            case $COLLECTOR in
                "performance-snapshot") TYPE="snap" ;;
                "hotspots"*)            TYPE="hot"  ;;
                "hpc-performance")      TYPE="hpc"  ;;
            esac

            RDIR="res_${TYPE}_${PREFIX}"

            # Executa o VTune respeitando o limite de tempo (se houver)
            $CMD_PREFIX vtune -collect $COLLECTOR -r $RDIR -start-paused \
                -- ./$BIN $DS_PATH

            EXIT=$?
            if [ $EXIT -eq 124 ]; then
                echo "⚠️  TIMEOUT: $DS_NAME | Variante=$VARIANT | Coletor=$TYPE — pulando relatório"
                continue
            fi

            # Se terminou a tempo, gera o relatório CSV consolidado
            vtune -report summary -r $RDIR -format=csv \
                -report-output $FINAL_RESULTS/vtune_csvs/${TYPE}_${PREFIX}.csv
        done
    done
}

# 1. Roda o dataset leve (sem limite de tempo)
for DS in "${DATASETS_NORMAL[@]}"; do
    run_baseline "data/${DS}.csv" "${DS}" 0
done

# 2. Roda os monstros de alta dimensionalidade (protegidos contra loops infinitos)
for DS in "${DATASETS_TIMED[@]}"; do
    run_baseline "data/${DS}.csv" "${DS}" $TIMEOUT_SECS
done

# Salva o log acumulado do benchmark se ele foi gerado por alguma instância
if [ -f "bench_results.csv" ]; then
    cp bench_results.csv $FINAL_RESULTS/app_bench_results.csv
fi

echo "Baselines finalizados. Resultados salvos em: $FINAL_RESULTS"
