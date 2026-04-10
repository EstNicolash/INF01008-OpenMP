#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <filesystem>
#include <omp.h>
#include "linear_regression.hpp"


namespace fs = std::filesystem;

void log_to_csv(const std::string& log_file,
                const std::string& ds_name,
                size_t n, size_t d,
                int threads,
                double load_t, double fit_t) {
    std::ofstream file(log_file, std::ios::app);
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "dataset,samples,features,threads,load_time,fit_time,total_time\n";
    }
    file << ds_name << "," << n << "," << d << "," << threads << ","
         << std::fixed << std::setprecision(6)
         << load_t << "," << fit_t << "," << (load_t + fit_t) << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path.csv> [log_file.csv]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::string log_filename = (argc > 2) ? argv[2] : "bench_results.csv";

    std::string ds_name = fs::path(filepath).filename().string();

    LinearRegression model;

    std::cout << "================================================" << std::endl;
    std::cout << "   Parallel Linear Regression - OpenMP Version  " << std::endl;
    std::cout << "================================================" << std::endl;

    // 1. Data Loading Stage
    double start_load = omp_get_wtime();
    if (!model.load_data(filepath)) {
        std::cerr << "Error: Could not open or read " << filepath << std::endl;
        return 1;
    }
    double end_load = omp_get_wtime();
    double load_duration = end_load - start_load;

    // 2. Metadata Retrieval
    size_t N = model.get_num_samples();
    size_t D = model.get_num_features();
    int threads = omp_get_max_threads();

    std::cout << "Dataset: " << ds_name << " (" << N << "x" << D << ")" << std::endl;
    std::cout << "Threads: " << threads << std::endl;
    std::cout << "Load Time: " << load_duration << "s" << std::endl;

    // 3. Training Stage
    double start_fit = omp_get_wtime();
    model.fit();
    double end_fit = omp_get_wtime();
    double fit_duration = end_fit - start_fit;

    // 4. Logging
    log_to_csv(log_filename, ds_name, N, D, threads, load_duration, fit_duration);
    std::cout << "Results logged to " << log_filename << std::endl;

    // 5. Execution Summary & Coefficients
    std::cout << "\n--- Execution Summary ---" << std::endl;
    std::cout << "Fit Execution Time: " << std::fixed << std::setprecision(6)
              << fit_duration << " seconds" << std::endl;

    const auto& beta = model.get_coefficients();
    if (beta) {
        std::cout << "\nModel Coefficients (first 10):" << std::endl;
        for (size_t i = 0; i < (D < 10 ? D : 10); ++i) {
            std::cout << "  Beta[" << i << "]: " << beta[i] << std::endl;
        }
    }
    std::cout << "================================================" << std::endl;

    return 0;
}
