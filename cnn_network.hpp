#ifndef CNN_NETWORK_HPP
#define CNN_NETWORK_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <string>
#include "matrix.hpp"
#include "layer.hpp"
#include "conv_layer.hpp"
#include "pool_layer.hpp"
#include "fully_connected_layer.hpp"
#include "activations.hpp"
#include "loss.hpp"

// ============================================================================
// CNN — 5-layer LeNet-5-style network for MNIST
//   C1: Conv  1x28x28 → 6x24x24  (5x5 kernel, Valid, ReLU)
//   S2: MaxPool 6x24x24 → 6x12x12  (2x2 window)
//   C3: Conv  6x12x12 → 12x8x8   (5x5 kernel, Valid, ReLU)
//   S4: MaxPool 12x8x8  → 12x4x4   (2x2 window)
//   O5: FC  192 → 10  (Affine + Softmax)
// ============================================================================
template<typename T = float>
class CNN {
public:
    CNN()
        : c1_(28, 28, 5, 1, 6)
        , s2_(24, 24, 2, 6, 6, PoolType::Max)
        , c3_(12, 12, 5, 6, 12)
        , s4_(8, 8, 2, 12, 12, PoolType::Max)
        , o5_(192, 10)
    {}

    // Forward pass: input is 1×784 row vector (flattened 28×28 image)
    // Returns 1×10 probability vector
    Matrix<T> forward(const Matrix<T>& input) {
        // Reshape 1×784 to 28×28 for C1
        Matrix<T> img(28, 28);
        for (int r = 0; r < 28; ++r)
            for (int c = 0; c < 28; ++c)
                img(r, c) = input.data()[r * 28 + c];

        // C1
        c1Inputs_ = { img };
        c1_.setLastInputs(c1Inputs_);
        c1Outputs_ = c1_.forward(c1Inputs_);

        // S2
        s2Outputs_ = s2_.forward(c1Outputs_);

        // C3
        c3_.setLastInputs(s2Outputs_);
        c3Outputs_ = c3_.forward(s2Outputs_);

        // S4
        s4Outputs_ = s4_.forward(c3Outputs_);

        // O5 — flatten S4 outputs (12×4×4) into 1×192
        o5Outputs_ = o5_.forward(s4Outputs_);

        return o5Outputs_[0];
    }

    // Backward pass using cross-entropy loss gradient
    void backward(const Matrix<T>& target) {
        // Compute combined cross-entropy + softmax gradient: Y - target
        Matrix<T> outputGrad = loss::crossEntropyGradient(o5_.output()[0], target);

        // O5 backward — returns flat gradient
        auto o5Grad = o5_.backward({ outputGrad });

        // Reshape flat gradient (1×192) back to 12×4×4 for S4
        int fmSize = 4 * 4;
        int numFM = 12;
        std::vector<Matrix<T>> s4Grad(numFM);
        for (int i = 0; i < numFM; ++i) {
            s4Grad[i] = Matrix<T>(4, 4);
            for (int j = 0; j < fmSize; ++j)
                s4Grad[i].data()[j] = o5Grad[0].data()[i * fmSize + j];
        }

        // S4 backward
        auto s4BackGrad = s4_.backward(s4Grad);

        // C3 backward
        auto c3BackGrad = c3_.backward(s4BackGrad);

        // S2 backward
        auto s2BackGrad = s2_.backward(c3BackGrad);

        // C1 backward
        c1_.backward(s2BackGrad);
    }

    // Update all parameters
    void updateParams(T learningRate) {
        c1_.updateParams(learningRate);
        c3_.updateParams(learningRate);
        o5_.updateParams(learningRate);
    }

    // Clear intermediate values between training samples
    void clearGradients() {
        c1_.clearGradients();
        s2_.clearGradients();
        c3_.clearGradients();
        s4_.clearGradients();
        o5_.clearGradients();
    }

    // Compute cross-entropy loss
    T computeLoss(const Matrix<T>& predicted, const Matrix<T>& target) const {
        return loss::crossEntropy(predicted, target);
    }

    // Layer access for debugging / weight saving
    const ConvLayer<T>&        C1() const { return c1_; }
    const PoolLayer<T>&        S2() const { return s2_; }
    const ConvLayer<T>&        C3() const { return c3_; }
    const PoolLayer<T>&        S4() const { return s4_; }
    const FullyConnectedLayer<T>& O5() const { return o5_; }

    // Return number of correct predictions for testing
    static int argmax(const Matrix<T>& probs) {
        int best = 0;
        T bestVal = probs.data()[0];
        for (int i = 1; i < probs.cols(); ++i) {
            if (probs.data()[i] > bestVal) {
                bestVal = probs.data()[i];
                best = i;
            }
        }
        return best;
    }

private:
    ConvLayer<T> c1_;
    PoolLayer<T> s2_;
    ConvLayer<T> c3_;
    PoolLayer<T> s4_;
    FullyConnectedLayer<T> o5_;

    // Intermediate storage for forward pass (needed for backward)
    std::vector<Matrix<T>> c1Inputs_;
    std::vector<Matrix<T>> c1Outputs_;
    std::vector<Matrix<T>> s2Outputs_;
    std::vector<Matrix<T>> c3Outputs_;
    std::vector<Matrix<T>> s4Outputs_;
    std::vector<Matrix<T>> o5Outputs_;
};

#endif // CNN_NETWORK_HPP
