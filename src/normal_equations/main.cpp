#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <filesystem>
#include <omp.h>
#include "linear_regression.hpp"

#ifdef USE_ITT
    #include <ittnotify.h>
    #define VTUNE_RESUME() __itt_resume()
    #define VTUNE_PAUSE()  __itt_pause()
#else
    #define VTUNE_RESUME()
    #define VTUNE_PAUSE()
#endif

#ifdef SILENT
    #define LOG(msg)
#else
    #define LOG(msg) std::cout << msg
#endif

namespace fs = std::filesystem;

void log_to_csv(const std::string& log_file, const std::string& ds_name,
                size_t n, size_t d, int threads,
                double load_t, double fit_t) {
    std::ofstream file(log_file, std::ios::app);
    if (file.tellp() == 0) {
        file << "dataset,samples,features,threads,load_time,fit_time,total_time\n";
    }
    file << ds_name << "," << n << "," << d << "," << threads << ","
         << std::fixed << std::setprecision(6)
         << load_t << "," << fit_t << "," << (load_t + fit_t) << "\n";
}

int main(int argc, char* argv[]) {
    VTUNE_PAUSE();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path.csv> [log_file.csv]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::string log_filename = (argc > 2) ? argv[2] : "bench_results.csv";
    std::string ds_name = fs::path(filepath).filename().string();

    LinearRegression model;

    LOG("================================================\n");
    LOG("   Parallel Linear Regression - OpenMP Version  \n");
    LOG("================================================\n");

    double start_load = omp_get_wtime();
    if (!model.load_data(filepath)) {
        std::cerr << "Error: Could not open or read " << filepath << std::endl;
        return 1;
    }
    double end_load = omp_get_wtime();
    double load_duration = end_load - start_load;

    size_t N = model.get_num_samples();
    size_t D = model.get_num_features();
    int threads = omp_get_max_threads();

    LOG("Dataset: " << ds_name << " (" << N << "x" << D << ")\n");
    LOG("Threads: " << threads << "\n");
    LOG("Load Time: " << load_duration << "s\n");

    VTUNE_RESUME();
    double start_fit = omp_get_wtime();
    model.fit();
    double end_fit = omp_get_wtime();
    VTUNE_PAUSE();

    double fit_duration = end_fit - start_fit;

    log_to_csv(log_filename, ds_name, N, D, threads, load_duration, fit_duration);
    LOG("Results logged to " << log_filename << "\n");

    LOG("\n--- Execution Summary ---\n");
    LOG("Fit Execution Time: " << std::fixed << std::setprecision(6) << fit_duration << " seconds\n");

    const auto& beta = model.get_coefficients();
    if (beta) {
        LOG("\nModel Coefficients (first 10):\n");
        for (size_t i = 0; i < (D < 10 ? D : 10); ++i) {
            LOG("  Beta[" << i << "]: " << beta[i] << "\n");
        }
    }
    LOG("================================================\n");

    return 0;
}
