#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <ittnotify.h>

// ===== FUNÇÃO =====
void runGradientDescent(const std::vector<std::vector<float>>& trainingData,
                        float learningRate,
                        int numIterations,
                        bool verbose);

// ===== GERADOR =====
std::vector<std::vector<float>> generateData(int N) {
    std::vector<std::vector<float>> data;
    data.reserve(N);

    for (int i = 0; i < N; i++) {
        float x = static_cast<float>(i);
        float y = x;
        data.push_back({x, y});
    }

    return data;
}

// ===== MAIN =====
int main(int argc, char* argv[]) {
    float learningRate = 0.0000001f;

    int numIterations = 5000;

    // ===== VTUNE =====
    if (argc == 3) {
        int N = std::stoi(argv[1]);
        int threads = std::stoi(argv[2]);

        omp_set_num_threads(threads);

        std::cout << "VTune mode\n";
        std::cout << "N = " << N << ", Threads = " << threads << "\n";

        auto data = generateData(N);

        __itt_resume();

        runGradientDescent(data, learningRate, numIterations, false);

        __itt_pause();

        return 0;
    }

    // ===== BENCHMARK =====
    std::vector<int> threadCounts = {1, 2, 4, 8, 16};
    std::vector<int> sizes = {1000, 10000, 100000, 500000};

    std::ofstream file("results.csv");
    file << "N,threads,time\n";

    for (int N : sizes) {
        std::cout << "\n===== Dataset size: " << N << " =====\n";

        auto data = generateData(N);

        for (int p : threadCounts) {
            omp_set_num_threads(p);

            // warmup
            runGradientDescent(data, learningRate, 10, false);

            double start = omp_get_wtime();

            runGradientDescent(data, learningRate, numIterations, false);

            double end = omp_get_wtime();

            double elapsed = end - start;

            std::cout << "Threads: " << p
                      << " Time: " << elapsed << " sec\n";

            file << N << "," << p << "," << elapsed << "\n";
        }
    }

    file.close();

    std::cout << "\nResultados salvos em results.csv\n";

    return 0;
}