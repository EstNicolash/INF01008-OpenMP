#include <iostream>
#include <vector>

/// @brief Translates MIPS 32 bit instruction into assembly
/// @param x 32 bit binary instruction
/// @return
std::vector<float> convertToFeatureVector(int x);

/// @brief 
/// @return 
std::vector<float> initializeWeights(); 

/// @brief 
/// @param weights
/// @param trainingData
/// @return 
float calculateTrainingLoss(const std::vector<float>& weights,
                            const std::vector<std::vector<float>>& trainingData);

/// @brief 
/// @param weights
/// @param trainingData
/// @return 
std::vector<float> calculateGradient(const std::vector<float>& weights,
                                     const std::vector<std::vector<float>>& trainingData);

// Function to add two vectors and store the result in the first vector
void addVectors(std::vector<float>& vec1, const std::vector<float>& vec2);

/// @brief
/// @param trainingData
/// @param learningRate
/// @param numIterations
void runGradientDescent(std::vector<std::vector<float>> trainingData, float learningRate, int numIterations, bool verbose = false);