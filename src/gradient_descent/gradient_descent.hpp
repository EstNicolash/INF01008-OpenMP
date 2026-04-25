#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include <iostream>
#include <vector>
#include <memory>


struct Matrix {
    size_t rows;
    size_t cols;
    std::unique_ptr<float[]> data;

    Matrix(size_t r, size_t c)
        : rows(r), cols(c), data(std::make_unique<float[]>(r * c)) {}

    inline float& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    inline float operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
};

void normalizeFeatures(Matrix& data);

/// @brief Translates MIPS 32 bit instruction into assembly
/// @param x 32 bit binary instruction
/// @return
std::vector<float> convertToFeatureVector(int x);

/// @brief
/// @return
std::vector<float> initializeWeights();

/// @brief
/// @param trainingData
/// @param trainingData
/// @return
void calculateLossAndGradient(const Matrix& trainingData,
                               const std::vector<float>& weights,
                               float& loss,
                               std::vector<float>& grads);


// Function to add two vectors and store the result in the first vector
void addVectors(std::vector<float>& vec1, const std::vector<float>& vec2);

/// @brief
/// @param trainingData
/// @param learningRate
/// @param numIterations
std::vector<float> runGradientDescent(const Matrix& trainingData, float learningRate, int numIterations, bool verbose = false);

#endif // GRADIENT_DESCENT_HPP
