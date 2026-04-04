#include "linear_regression.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <omp.h>

std::unique_ptr<Matrix> LinearRegression::compute_XtX() const {

    auto XT = std::make_unique<Matrix>(num_features, num_samples);
    #pragma omp parallel for schedule(runtime)
    for(size_t i = 0; i < num_features; i++){
        for(size_t j = 0; j < num_samples; j++){
            (*XT)(i, j) = (*X)(j, i);
        }
    }
    std::unique_ptr<Matrix> A = std::make_unique<Matrix>(num_features, num_features);

    #pragma omp parallel for schedule(runtime)
    for(size_t i = 0; i < num_features; i++){
        for(size_t j = i; j < num_features; j++){
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for(size_t k = 0; k < num_samples; k++){
                sum += (*XT)(i,k) * (*XT)(j,k);
            }

            (*A)(i, j) = sum;
            if (i != j) {
                (*A)(j, i) = sum;
            }
        }
    }

    return A;
}

std::unique_ptr<double[]> LinearRegression::compute_Xty() const {

    std::unique_ptr<double[]> b = std::make_unique<double[]>(num_features);

    #pragma omp parallel for schedule(runtime)
    for(size_t i = 0; i < num_features; i++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for(size_t k = 0; k < num_samples; k++) {
            sum += (*X)(k, i) * y[k];
        }
        b[i] = sum;
    }
    return b;
}


std::unique_ptr<Matrix> LinearRegression::invert(const Matrix& m) const {
    size_t n = m.cols;

    // Copy matrix
    auto A = std::make_unique<Matrix>(n, n);
    for (size_t i = 0; i < n * n; ++i) {
        A->data[i] = m.data[i];
    }

    // Create identity matrix
    auto I = std::make_unique<Matrix>(n, n);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            (*I)(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan
    for (size_t i = 0; i < n; i++) {
        size_t pivot_row = i;
        double max_val = std::abs((*A)(i, i));
        for (size_t k = i + 1; k < n; k++) {
            if (std::abs((*A)(k, i)) > max_val) {
                max_val = std::abs((*A)(k, i));
                pivot_row = k;
            }
        }

        if (max_val < 1e-15) throw std::runtime_error("Matriz singular!");

        if (pivot_row != i) {
            // #pragma omp parallel for schedule(runtime)
            for (size_t j = 0; j < n; j++) {
                std::swap((*A)(i, j), (*A)(pivot_row, j));
                std::swap((*I)(i, j), (*I)(pivot_row, j));
            }
        }

        double pivot = (*A)(i, i);
        // #pragma omp parallel for schedule(runtime)
        for (size_t j = 0; j < n; j++) {
            (*A)(i, j) /= pivot;
            (*I)(i, j) /= pivot;
        }

        #pragma omp parallel for schedule(runtime)
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                double factor = (*A)(k, i);
                #pragma omp simd
                for (size_t j = 0; j < n; j++) {
                    (*A)(k, j) -= factor * (*A)(i, j);
                    (*I)(k, j) -= factor * (*I)(i, j);
                }
            }
        }
    }

    return I;
}

bool LinearRegression::load_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    num_samples = 0;
    num_features = 0;

    if (std::getline(file, line)) {
        num_samples++;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) num_features++;
    }
    while (std::getline(file, line)) num_samples++;

    num_features--;

    X = std::make_unique<Matrix>(num_samples, num_features);
    y = std::make_unique<double[]>(num_samples);

    file.clear();
    file.seekg(0);

    size_t current_row = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        y[current_row] = std::stod(value);

        for (size_t j = 0; j < num_features; ++j) {
            std::getline(ss, value, ',');
            (*X)(current_row, j) = std::stod(value);
        }
        current_row++;
    }

    return true;
}

void LinearRegression::fit() {
    if (num_samples == 0) return;

    auto XtX = compute_XtX();
    auto Xty = compute_Xty();
    auto XtX_inv = invert(*XtX);

    // (XtX_inv) * (Xty)
    beta = std::make_unique<double[]>(num_features);

    #pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < num_features; i++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t j = 0; j < num_features; j++) {
            sum += (*XtX_inv)(i, j) * Xty[j];
        }
        beta[i] = sum;
    }

}
