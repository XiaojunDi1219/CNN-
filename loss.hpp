#ifndef LOSS_HPP
#define LOSS_HPP

#include <cmath>
#include <stdexcept>
#include "matrix.hpp"

// ============================================================================
// 损失函数
// ============================================================================

namespace loss {

template<typename T>
T crossEntropy(const Matrix<T>& predicted, const Matrix<T>& target) {
    if (predicted.rows() != 1 || target.rows() != 1 || predicted.cols() != target.cols())
        throw std::invalid_argument("crossEntropy: both must be 1xN row vectors");

    T loss = T(0);
    const T eps = T(1e-10);
    for (int i = 0; i < predicted.cols(); ++i) {
        if (target.data()[i] > T(0)) {
            loss -= target.data()[i] * std::log(std::max(predicted.data()[i], eps));
        }
    }
    return loss;
}

// 交叉熵对 logits（softmax 之前）的梯度:
//   dL/d(logits) = softmax(logits) - target  = predicted - target
template<typename T>
Matrix<T> crossEntropyGradient(const Matrix<T>& softmaxOutput, const Matrix<T>& target) {
    if (softmaxOutput.rows() != 1 || target.rows() != 1)
        throw std::invalid_argument("crossEntropyGradient: both must be 1xN row vectors");
    Matrix<T> grad = softmaxOutput;
    for (int i = 0; i < grad.cols(); ++i)
        grad.data()[i] -= target.data()[i];
    return grad;
}

} // namespace loss

#endif // LOSS_HPP
