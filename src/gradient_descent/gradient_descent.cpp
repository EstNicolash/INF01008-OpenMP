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
    size_t N = data.rows;
    size_t cols = data.cols;

    #pragma omp parallel for schedule(runtime)
    for (size_t j = 1; j < cols; j++) {

        float sum = 0.0f;
        for (size_t i = 0; i < N; i++) {
            sum += data(i, j);
        }
        float mean = sum / static_cast<float>(N);

        float variance_sum = 0.0f;
        for (size_t i = 0; i < N; i++) {
            float diff = data(i, j) - mean;
            variance_sum += diff * diff;
        }
        float variance = variance_sum / static_cast<float>(N);
        float std_dev = std::sqrt(variance);

        if (std_dev < 1e-8f) {
            std_dev = 1.0f;
        }

        for (size_t i = 0; i < N; i++) {
            data(i, j) = (data(i, j) - mean) / std_dev;
        }
    }
}
std::vector<float> calculateLossAndGradient(const Matrix& trainingData,
                                            const std::vector<float>& weights,
                                            float& out_loss) {
    size_t N = trainingData.rows;
    size_t D = trainingData.cols - 1;

    std::vector<float> global_grads(D, 0.0f);
    float global_loss = 0.0f;

    #pragma omp parallel
    {
        std::vector<float> local_grads(D, 0.0f);
        float local_loss = 0.0f;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < N; i++) {
            float y = trainingData(i, 0);

            float score = 0.0f;
            #pragma omp simd
            for (size_t j = 0; j < D; j++) {
                score += weights[j] * trainingData(i, j + 1);
            }

            float residual = score - y;
            local_loss += residual * residual;

            #pragma omp simd
            for (size_t j = 0; j < D; j++) {
                local_grads[j] += 2.0f * residual * trainingData(i, j + 1);
            }
        }

        #pragma omp critical
        {
            global_loss += local_loss;
            for (size_t j = 0; j < D; j++) {
                global_grads[j] += local_grads[j];
            }
        }
    }

    out_loss = global_loss / static_cast<float>(N);
    for (size_t j = 0; j < D; j++) {
        global_grads[j] /= static_cast<float>(N);
    }

    return global_grads;
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

        for (size_t j = 0; j < D; j++) {
            weights[j] -= learningRate * grads[j];
        }

        if (verbose && (i % 100 == 0 || i == numIterations - 1)) {
            std::cout << "Iteration: " << i << " | Loss: " << loss << "\n";
        }
    }

    return weights;
}
