#include "gradient_descent.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>

#include "gradient_descent.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>

void normalizeFeatures(Matrix& data) {
    size_t N    = data.rows;
    size_t cols = data.cols;

    const float inv_N = 1.0f / static_cast<float>(N);

    #pragma omp parallel for schedule(runtime)
    for (size_t j = 1; j < cols; j++) {

        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < N; i++) {
            sum += data(i, j);
        }
        const float mean = sum * inv_N;

        float variance_sum = 0.0f;
        #pragma omp simd reduction(+:variance_sum)
        for (size_t i = 0; i < N; i++) {
            const float diff = data(i, j) - mean;
            variance_sum += diff * diff;
        }

        float std_dev = std::sqrt(variance_sum * inv_N);
        if (std_dev < 1e-8f) std_dev = 1.0f;

        const float inv_std = 1.0f / std_dev;

        #pragma omp simd
        for (size_t i = 0; i < N; i++) {
            data(i, j) = (data(i, j) - mean) * inv_std;
        }
    }
}
std::vector<float> calculateLossAndGradient(const Matrix& trainingData,
                                            const std::vector<float>& weights,
                                            float& out_loss) {
    size_t N = trainingData.rows;
    size_t D = trainingData.cols - 1;

    std::vector<float> grads(D, 0.0f);
    float total_loss = 0.0f;
    const float inv_N = 1.0f / static_cast<float>(N);

    float* __restrict__       raw_grads = grads.data();
    const float* __restrict__ raw_w     = weights.data();
    const float* __restrict__ raw_data  = trainingData.data.get();
    const size_t              cols      = trainingData.cols;

    #pragma omp parallel for schedule(static)  reduction(+:total_loss)  reduction(+:raw_grads[0:D])
    for (size_t i = 0; i < N; i++) {

        const float* __restrict__ row = raw_data + i * cols;

        const float y = row[0];

        float score = 0.0f;
        #pragma omp simd reduction(+:score)
        for (size_t j = 0; j < D; j++) {
            score += raw_w[j] * row[j + 1];
        }

        const float residual     = score - y;
        total_loss              += residual * residual;
        const float two_residual = 2.0f * residual;

        #pragma omp simd
        for (size_t j = 0; j < D; j++) {
            raw_grads[j] += two_residual * row[j + 1];
        }
    }

    out_loss = total_loss * inv_N;

    #pragma omp simd
    for (size_t j = 0; j < D; j++) {
        raw_grads[j] *= inv_N;
    }

    return grads;
}



// === GRADIENT DESCENT ===
std::vector<float> runGradientDescent(const Matrix& trainingData,
                                      float learningRate,
                                      int numIterations,
                                      bool verbose) {

    size_t D = trainingData.cols - 1;
    std::vector<float> weights(D, 0.0f);

    for (int i = 0; i < numIterations; i++) {
        float loss = 0.0f;

        std::vector<float> grads = calculateLossAndGradient(trainingData, weights, loss);

        #pragma omp simd
        for (size_t j = 0; j < D; j++) {
            weights[j] -= learningRate * grads[j];
        }

        if (verbose && (i % 100 == 0 || i == numIterations - 1)) {
            std::cout << "Iteration: " << i << " | Loss: " << loss << "\n";
        }
    }

    return weights;
}
