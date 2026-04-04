#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>
#include "linear_regression.hpp"

int main(int argc, char* argv[]) {
    // 1. Basic argument validation
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path.csv>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    LinearRegression model;

    std::cout << "================================================" << std::endl;
    std::cout << "   Parallel Linear Regression - OpenMP Version  " << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Loading dataset: " << filename << std::endl;

    // 2. Data Loading Stage
    double start_load = omp_get_wtime();
    if (!model.load_data(filename)) {
        std::cerr << "Error: Could not open or read the file." << std::endl;
        return 1;
    }
    double end_load = omp_get_wtime();

    std::cout << "Data loaded successfully in " << (end_load - start_load) << "s" << std::endl;

    // 3. Training Stage
    std::cout << "Starting model training (Calculating Normal Equations)..." << std::endl;

    double start_fit = omp_get_wtime();
    model.fit();
    double end_fit = omp_get_wtime();

    // 4. Execution Summary
    std::cout << "\n--- Execution Summary ---" << std::endl;
    std::cout << "Fit Execution Time: " << std::fixed << std::setprecision(6)
              << (end_fit - start_fit) << " seconds" << std::endl;
    std::cout << "Available Threads:  " << omp_get_max_threads() << std::endl;

    // 5. Coefficient Output
    const auto& beta = model.get_coefficients();
    if (beta) {
        std::cout << "\nModel Coefficients (showing first 10):" << std::endl;
        for (size_t i = 0; i < 10; ++i) {
            std::cout << "  Beta[" << i << "]: " << std::setw(10) << beta[i] << std::endl;
        }
        std::cout << "  ..." << std::endl;
    }

    std::cout << "================================================" << std::endl;

    return 0;
}
