#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "matrix.hpp"

// ============================================================================
// Activation functions — free functions and functors
// Demonstrates: template functions, callable objects (functors), in-place ops
// ============================================================================

namespace activation {

// ---------- ReLU ----------
template<typename T>
inline T relu(T x) {
    return x > T(0) ? x : T(0);
}

template<typename T>
inline T reluDerivative(T y) {
    // y is the output of relu: y>0 means the input was >0
    return y > T(0) ? T(1) : T(0);
}

// ReLU functor — callable object
template<typename T>
struct ReLU {
    Matrix<T> operator()(const Matrix<T>& x) const {
        Matrix<T> result(x.rows(), x.cols());
        for (int i = 0; i < x.size(); ++i)
            result.data()[i] = relu(x.data()[i]);
        return result;
    }

    Matrix<T> derivative(const Matrix<T>& y) const {
        Matrix<T> result(y.rows(), y.cols());
        for (int i = 0; i < y.size(); ++i)
            result.data()[i] = reluDerivative(y.data()[i]);
        return result;
    }
};

// ---------- Sigmoid (for reference / alternative) ----------
template<typename T>
inline T sigmoid(T x) {
    return T(1) / (T(1) + std::exp(-x));
}

template<typename T>
inline T sigmoidDerivative(T y) {
    return y * (T(1) - y);
}

// ---------- Softmax (operates on a single row vector) ----------
template<typename T>
Matrix<T> softmax(const Matrix<T>& logits) {
    // logits must be 1 x N
    if (logits.rows() != 1)
        throw std::invalid_argument("softmax expects a 1xN row vector");

    Matrix<T> result(1, logits.cols());

    // Find max for numerical stability
    T maxVal = logits.data()[0];
    for (int i = 1; i < logits.cols(); ++i)
        maxVal = std::max(maxVal, logits.data()[i]);

    T sum = T(0);
    for (int i = 0; i < logits.cols(); ++i) {
        T val = std::exp(logits.data()[i] - maxVal);
        result.data()[i] = val;
        sum += val;
    }

    for (int i = 0; i < logits.cols(); ++i)
        result.data()[i] /= sum;

    return result;
}

} // namespace activation

#endif // ACTIVATIONS_HPP
