#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct Matrix {
    size_t rows;
    size_t cols;
    std::unique_ptr<double[]> data;

    Matrix(size_t r, size_t c)
        : rows(r), cols(c), data(std::make_unique<double[]>(r * c)) {}

    inline double& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    inline double operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
};

class LinearRegression {
public:
    LinearRegression() = default;

    bool load_data(const std::string& filename);

    void fit();

    std::unique_ptr<double[]> predict(const Matrix& input_X) const;

    const std::unique_ptr<double[]>& get_coefficients() const { return beta; }

private:
    std::unique_ptr<Matrix> X;
    std::unique_ptr<double[]> y;
    size_t num_samples = 0;
    size_t num_features = 0;

    std::unique_ptr<double[]> beta;

    std::unique_ptr<Matrix> transpose(const Matrix& m) const;
    std::unique_ptr<Matrix> multiply(const Matrix& A, const Matrix& B) const;
    std::unique_ptr<Matrix> invert(const Matrix& m) const;
};

#endif
