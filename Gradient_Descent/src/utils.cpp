#include "utils.h"
#include <cmath>
#include <omp.h>
#include <iostream>

const int NUM_FEATURES = 2;
const int DEFAULT_WEIGHT_VALUE = 0;

// INICIALIZAÇÃO
std::vector<float> initializeWeights() {
    return std::vector<float>(NUM_FEATURES, DEFAULT_WEIGHT_VALUE);
}

// LOSS
float calculateTrainingLoss(const std::vector<float>& weights,
                           const std::vector<std::vector<float>>& trainingData) {

    float totalLoss = 0.0f;
    int N = trainingData.size();

    #pragma omp parallel for reduction(+:totalLoss)
    for (int i = 0; i < N; i++) {
        float x = trainingData[i][0];
        float y = trainingData[i][1];

        float score = weights[0] + weights[1] * x;
        float residual = y - score;

        totalLoss += residual * residual;
    }

    return totalLoss / N;
}

// GRADIENT
void calculateGradient(const std::vector<float>& weights,
                       const std::vector<std::vector<float>>& trainingData,
                       float& grad0, float& grad1) {

    int N = trainingData.size();

    float g0 = 0.0f;
    float g1 = 0.0f;

    #pragma omp parallel for reduction(+:g0,g1)
    for (int i = 0; i < N; i++) {
        float x = trainingData[i][0];
        float y = trainingData[i][1];

        float score = weights[0] + weights[1] * x;
        float residual = (score - y) * 2.0f;

        g0 += residual;
        g1 += residual * x;
    }

    grad0 = g0 / N;
    grad1 = g1 / N;
}

// GRADIENT DESCENT
void runGradientDescent(const std::vector<std::vector<float>>& trainingData,
                        float learningRate,
                        int numIterations,
                        bool verbose) {

    std::vector<float> weights = initializeWeights();

    for (int i = 0; i < numIterations; i++) {

        float loss = calculateTrainingLoss(weights, trainingData);

        float g0, g1;
        calculateGradient(weights, trainingData, g0, g1);

        weights[0] -= learningRate * g0;
        weights[1] -= learningRate * g1;

        if (verbose) {
            std::cout << "Iteration: " << i
                      << " Loss: " << loss
                      << " Weights: " << weights[0] << " " << weights[1]
                      << " Gradient: " << g0 << " " << g1
                      << "\n";
        }
    }
}
