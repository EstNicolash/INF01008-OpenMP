#include "linear_regression.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <charconv>
#include <vector>
/*
std::unique_ptr<Matrix> LinearRegression::compute_XtX() const {

    auto XT = std::make_unique<Matrix>(num_features, num_samples);

    #pragma omp parallel for schedule(runtime)
    for(size_t i = 0; i < num_features; i++){
        #pragma omp simd
        for(size_t j = 0; j < num_samples; j++){
            (*XT)(i, j) = (*X)(j, i);
        }
    }
    std::unique_ptr<Matrix> A = std::make_unique<Matrix>(num_features, num_features);


    #pragma omp parallel for schedule(runtime)
    for(size_t i = 0; i < num_features; i++){
        const double* __restrict row_i = &(*XT)(i, 0);

        for(size_t j = i; j < num_features; j++){
            const double* __restrict row_j = &(*XT)(j, 0);
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for(size_t k = 0; k < num_samples; k++){
                sum += row_i[k] * row_j[k];
            }

            (*A)(i, j) = sum;
            if (i != j) (*A)(j, i) = sum;
        }
    }

    return A;
    }*/

std::unique_ptr<Matrix> LinearRegression::compute_XtX() const {
    size_t D = num_features;
    size_t N = num_samples;
    size_t total_size = D * D;

    auto A = std::make_unique<Matrix>(D, D);
    double* __restrict__ raw_A = A->data.get();
    const double* __restrict__ raw_X = X->data.get();

    std::fill(raw_A, raw_A + total_size, 0.0);

    #pragma omp parallel for schedule(runtime) reduction(+:raw_A[0:total_size])
    for (size_t k = 0; k < N; k++) {
        const double* __restrict__ row_k = &raw_X[k * D];

        for (size_t i = 0; i < D; i++) {
            double val_i = row_k[i];

            #pragma omp simd
            for (size_t j = i; j < D; j++) {
                raw_A[i * D + j] += val_i * row_k[j];
            }
        }
    }

    for (size_t i = 0; i < D; i++) {
        for (size_t j = i + 1; j < D; j++) {
            raw_A[j * D + i] = raw_A[i * D + j];
        }
    }

    return A;
}
std::unique_ptr<double[]> LinearRegression::compute_Xty() const {
    size_t D = num_features;
    size_t N = num_samples;
    auto b = std::make_unique<double[]>(D);

    const double* __restrict__ ptr_y = y.get();
    const double* __restrict__ ptr_X = X->data.get();
    double* __restrict__ raw_b = b.get();

    std::fill(raw_b, raw_b + D, 0.0);

    #pragma omp parallel for schedule(runtime) reduction(+:raw_b[0:D])
    for(size_t k = 0; k < N; k++) {
        double y_k = ptr_y[k];

        const double* __restrict__ row_k = &ptr_X[k * D];

        #pragma omp simd
        for(size_t i = 0; i < D; i++) {
            raw_b[i] += row_k[i] * y_k;
        }
    }

    return b;
}
/*
std::unique_ptr<double[]> LinearRegression::compute_Xty() const {
    auto b = std::make_unique<double[]>(num_features);

    const double* __restrict ptr_y = y.get();
    const double* __restrict ptr_X = X->data.get();
    size_t cols = X->cols;

    #pragma omp parallel for schedule(runtime)
    for(size_t i = 0; i < num_features; i++) {
        double sum = 0.0;

        #pragma omp simd reduction(+:sum)
        for(size_t k = 0; k < num_samples; k++) {
            sum += ptr_X[k * cols + i] * ptr_y[k];
        }
        b[i] = sum;
    }
    return b;
    }*/


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

                double* __restrict__ rK_A = &(*A)(k, 0);
                const double* __restrict__ rI_A = &(*A)(i, 0);

                double* __restrict__ rK_I = &(*I)(k, 0);
                const double* __restrict__ rI_I = &(*I)(i, 0);

                #pragma omp simd
                for (size_t j = 0; j < n; j++) {
                    rK_A[j] -= factor * rI_A[j];
                    rK_I[j] -= factor * rI_I[j];
                }
            }
        }
    }

    return I;
}

bool LinearRegression::load_data(const std::string& filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) return false;

    struct stat sb;
    fstat(fd, &sb);
    size_t length = sb.st_size;
    const char* addr = static_cast<const char*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0));
    if (addr == MAP_FAILED) { close(fd); return false; }

    std::vector<size_t> line_offsets;
    line_offsets.push_back(0);

    for (size_t i = 0; i < length; ++i) {
        if (addr[i] == '\n') line_offsets.push_back(i + 1);
    }
    if (addr[length-1] != '\n') line_offsets.push_back(length);

    num_samples = line_offsets.size() - 1;

    num_features = 0;
    size_t first_line_end = line_offsets[1] - 1;
    for (size_t i = 0; i < first_line_end; ++i) {
        if (addr[i] == ',') num_features++;
    }

    X = std::make_unique<Matrix>(num_samples, num_features);
    y = std::make_unique<double[]>(num_samples);
    double* raw_X = X->data.get();
    double* raw_y = y.get();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_samples; ++i) {
        const char* current = addr + line_offsets[i];
        const char* line_end = addr + line_offsets[i+1];

        std::from_chars_result res = std::from_chars(current, line_end, raw_y[i]);
        current = res.ptr + 1;

        for (size_t j = 0; j < num_features; ++j) {
            res = std::from_chars(current, line_end, raw_X[i * num_features + j]);
            current = res.ptr + 1;
        }
    }

    munmap((void*)addr, length);
    close(fd);
    return true;
}

void LinearRegression::fit() {
    if (num_samples == 0) return;

    auto XtX = compute_XtX();
    auto Xty = compute_Xty();
    auto XtX_inv = invert(*XtX);

    // (XtX_inv) * (Xty)
    beta = std::make_unique<double[]>(num_features);



    const double* __restrict__ ptr_inv = XtX_inv->data.get();
    const double* __restrict__ ptr_Xty = Xty.get();
    double* __restrict__ ptr_beta = beta.get();

    #pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < num_features; i++) {
        double sum = 0.0;
        const double* row_i = &ptr_inv[i * num_features];

        #pragma omp simd reduction(+:sum)
        for (size_t j = 0; j < num_features; j++) {
            sum += row_i[j] * ptr_Xty[j];
        }
        ptr_beta[i] = sum;
    }

}
