#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include "matrix.hpp"
#include "layer.hpp"
#include "activations.hpp"
#include "loss.hpp"

// ============================================================================
// FullyConnectedLayer — Affine transform + Softmax activation
// Input:  multiple feature maps which are flattened into a 1D vector
// Output: 1 x outputNum probability vector (via Softmax)
// ============================================================================
template<typename T = float>
class FullyConnectedLayer : public Layer<T> {
public:
    FullyConnectedLayer(int inputNum, int outputNum)
        : inputNum_(inputNum), outputNum_(outputNum)
    {
        initWeights();

        v_ = Matrix<T>(1, outputNum_, T(0));
        this->output_ = { Matrix<T>(1, outputNum_, T(0)) };
        this->gradient_ = { Matrix<T>(1, outputNum_, T(0)) };
        weightGradients_ = Matrix<T>(outputNum_, inputNum_, T(0));
    }

    const char* name() const override { return "FullyConnectedLayer"; }

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& inputs) override {
        // Flatten all input feature maps into a 1×inputNum_ row vector
        Matrix<T> flat(1, inputNum_);
        int idx = 0;
        for (const auto& fm : inputs) {
            for (int r = 0; r < fm.rows(); ++r)
                for (int c = 0; c < fm.cols(); ++c)
                    flat.data()[idx++] = fm(r, c);
        }
        lastFlatInput_ = flat;

        // Affine: v = flat * W^T + b  (W is outputNum_ × inputNum_)
        for (int i = 0; i < outputNum_; ++i) {
            T sum = biases_[i];
            for (int j = 0; j < inputNum_; ++j)
                sum += flat.data()[j] * weights_(i, j);
            v_.data()[i] = sum;
        }

        // Softmax
        this->output_[0] = activation::softmax(v_);
        return this->output_;
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& gradOutput) override {
        // For cross-entropy + softmax, the combined gradient is (Y - target)
        // This is computed externally and passed as gradOutput
        // Store as local gradient
        this->gradient_[0] = gradOutput[0];

        // Accumulate weight gradients: dW = d * x^T  (outer product)
        for (int i = 0; i < outputNum_; ++i) {
            for (int j = 0; j < inputNum_; ++j) {
                weightGradients_(i, j) += gradOutput[0].data()[i] * lastFlatInput_.data()[j];
            }
        }
        // Accumulate bias gradients
        for (int i = 0; i < outputNum_; ++i)
            biasGradients_[i] += gradOutput[0].data()[i];

        // Compute gradient w.r.t. flat input: d_input = d * W
        Matrix<T> gradFlat(1, inputNum_, T(0));
        for (int j = 0; j < inputNum_; ++j) {
            T sum = T(0);
            for (int i = 0; i < outputNum_; ++i)
                sum += gradOutput[0].data()[i] * weights_(i, j);
            gradFlat.data()[j] = sum;
        }

        // Reshape back to feature maps — caller knows the target shape
        // For now, just return as a single-element vector containing the flat gradient
        return { gradFlat };
    }

    void updateParams(T learningRate) override {
        for (int i = 0; i < outputNum_; ++i) {
            for (int j = 0; j < inputNum_; ++j)
                weights_(i, j) -= learningRate * weightGradients_(i, j);
            biases_[i] -= learningRate * biasGradients_[i];
        }
    }

    void clearGradients() override {
        weightGradients_.setZero();
        std::fill(biasGradients_.begin(), biasGradients_.end(), T(0));
        v_.setZero();
        this->output_[0].setZero();
        this->gradient_[0].setZero();
    }

    int inputNum()  const { return inputNum_; }
    int outputNum() const { return outputNum_; }

private:
    void initWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        T limit = std::sqrt(T(6.0) / (inputNum_ + outputNum_));
        std::uniform_real_distribution<T> dist(-limit, limit);

        weights_ = Matrix<T>(outputNum_, inputNum_);
        for (int i = 0; i < outputNum_; ++i)
            for (int j = 0; j < inputNum_; ++j)
                weights_(i, j) = dist(gen);

        biases_.resize(outputNum_, T(0));
        biasGradients_.resize(outputNum_, T(0));
    }

    int inputNum_, outputNum_;
    Matrix<T> weights_;           // outputNum_ × inputNum_
    std::vector<T> biases_;
    Matrix<T> v_;                 // pre-activation (logits)
    Matrix<T> weightGradients_;   // outputNum_ × inputNum_
    std::vector<T> biasGradients_;
    Matrix<T> lastFlatInput_;     // stored for backward pass
};

#endif // FULLY_CONNECTED_LAYER_HPP
