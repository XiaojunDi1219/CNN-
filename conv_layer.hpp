#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "matrix.hpp"
#include "layer.hpp"

// Convolution mode: Valid (no padding, output shrinks by kernel_size-1)
enum class ConvMode { Valid, Full };

// 2D correlation (no kernel rotation) — used in forward pass and kernel gradient
// Output dimensions depend on mode:
//   Valid: out_rows = in_rows - k_rows + 1, out_cols = in_cols - k_cols + 1
//   Full:  out_rows = in_rows + k_rows - 1, out_cols = in_cols + k_cols - 1
template<typename T>
Matrix<T> correlate2D(const Matrix<T>& input, const Matrix<T>& kernel, ConvMode mode) {
    int inRows = input.rows(), inCols = input.cols();
    int kRows  = kernel.rows(),  kCols  = kernel.cols();

    int outRows, outCols;
    int padRows, padCols;

    if (mode == ConvMode::Valid) {
        if (inRows < kRows || inCols < kCols)
            throw std::invalid_argument("correlate2D Valid: input must be >= kernel");
        outRows = inRows - kRows + 1;
        outCols = inCols - kCols + 1;
        padRows = 0;
        padCols = 0;
    } else { // Full
        outRows = inRows + kRows - 1;
        outCols = inCols + kCols - 1;
        padRows = kRows - 1;
        padCols = kCols - 1;
    }

    Matrix<T> result(outRows, outCols, T(0));

    // For Full mode, we zero-pad the input
    // Iterate over output positions
    for (int or_ = 0; or_ < outRows; ++or_) {
        for (int oc = 0; oc < outCols; ++oc) {
            T sum = T(0);
            for (int kr = 0; kr < kRows; ++kr) {
                int ir = or_ - padRows + kr;
                if (ir < 0 || ir >= inRows) continue;
                for (int kc = 0; kc < kCols; ++kc) {
                    int ic = oc - padCols + kc;
                    if (ic < 0 || ic >= inCols) continue;
                    sum += input(ir, ic) * kernel(kr, kc);
                }
            }
            result(or_, oc) = sum;
        }
    }
    return result;
}

// Convolution (with kernel rotation) — used in backward pass for input gradient
template<typename T>
Matrix<T> convolve2D(const Matrix<T>& kernel, const Matrix<T>& grad, ConvMode mode) {
    // Rotate kernel 180 degrees
    Matrix<T> rotated = kernel.rotate180();
    return correlate2D(grad, rotated, mode);
}

// ============================================================================
// ConvLayer — convolutional layer with ReLU activation
// Forward:  Y = ReLU( sum_j(input[j] * kernel[i][j]) + bias[i] )
// Backward: computes grad w.r.t. input and accumulates kernel/bias gradients
// ============================================================================
template<typename T = float>
class ConvLayer : public Layer<T> {
public:
    ConvLayer(int inputWidth, int inputHeight,
              int kernelSize, int inChannels, int outChannels)
        : inputWidth_(inputWidth), inputHeight_(inputHeight),
          kernelSize_(kernelSize),
          inChannels_(inChannels), outChannels_(outChannels),
          outputWidth_(inputWidth - kernelSize + 1),
          outputHeight_(inputHeight - kernelSize + 1)
    {
        if (outputWidth_ <= 0 || outputHeight_ <= 0)
            throw std::invalid_argument("ConvLayer: input smaller than kernel");

        initWeights();

        // Allocate intermediate storage
        int outW = outputWidth_, outH = outputHeight_;
        v_.resize(outChannels_, Matrix<T>(outH, outW, T(0)));
        this->output_.resize(outChannels_, Matrix<T>(outH, outW, T(0)));
        this->gradient_.resize(outChannels_, Matrix<T>(outH, outW, T(0)));
        kernelGradients_.resize(inChannels_,
            std::vector<Matrix<T>>(outChannels_, Matrix<T>(kernelSize_, kernelSize_, T(0))));
    }

    const char* name() const override { return "ConvLayer"; }

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& inputs) override {
        if (static_cast<int>(inputs.size()) != inChannels_)
            throw std::invalid_argument("ConvLayer::forward: wrong input channel count");

        for (int oc = 0; oc < outChannels_; ++oc) {
            v_[oc].setZero();
            for (int ic = 0; ic < inChannels_; ++ic) {
                Matrix<T> corr = correlate2D(inputs[ic], kernels_[ic][oc], ConvMode::Valid);
                v_[oc] += corr;
            }
            // Add bias and apply ReLU
            for (int r = 0; r < outputHeight_; ++r)
                for (int c = 0; c < outputWidth_; ++c) {
                    T val = v_[oc](r, c) + biases_[oc];
                    v_[oc](r, c) = val;
                    this->output_[oc](r, c) = val > T(0) ? val : T(0);
                }
        }
        return this->output_;
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& gradOutput) override {
        if (static_cast<int>(gradOutput.size()) != outChannels_)
            throw std::invalid_argument("ConvLayer::backward: wrong grad output channel count");

        // Compute local gradient: gradOutput ⊙ ReLU'(output)
        for (int oc = 0; oc < outChannels_; ++oc) {
            for (int r = 0; r < outputHeight_; ++r) {
                for (int c = 0; c < outputWidth_; ++c) {
                    T reluDeriv = this->output_[oc](r, c) > T(0) ? T(1) : T(0);
                    this->gradient_[oc](r, c) = gradOutput[oc](r, c) * reluDeriv;
                }
            }
        }

        // Accumulate kernel gradients
        for (int oc = 0; oc < outChannels_; ++oc) {
            for (int ic = 0; ic < inChannels_; ++ic) {
                // kernelGradient = correlate(input, localGradient, Valid)
                // Need inputs — stored separately
                Matrix<T> kg = correlate2D(lastInputs_[ic], this->gradient_[oc], ConvMode::Valid);
                kernelGradients_[ic][oc] += kg;
            }
            // Accumulate bias gradient: sum of local gradient
            biasGradients_[oc] += this->gradient_[oc].sum();
        }

        // Compute gradient w.r.t. input (to pass to previous layer)
        std::vector<Matrix<T>> gradInput(inChannels_);
        for (int ic = 0; ic < inChannels_; ++ic) {
            gradInput[ic] = Matrix<T>(inputHeight_, inputWidth_, T(0));
            for (int oc = 0; oc < outChannels_; ++oc) {
                // Full convolution of rotated kernel with local gradient
                Matrix<T> contrib = convolve2D(kernels_[ic][oc], this->gradient_[oc], ConvMode::Full);
                gradInput[ic] += contrib;
            }
        }

        return gradInput;
    }

    void updateParams(T learningRate) override {
        for (int oc = 0; oc < outChannels_; ++oc) {
            for (int ic = 0; ic < inChannels_; ++ic) {
                for (int r = 0; r < kernelSize_; ++r)
                    for (int c = 0; c < kernelSize_; ++c)
                        kernels_[ic][oc](r, c) -= learningRate * kernelGradients_[ic][oc](r, c);
            }
            biases_[oc] -= learningRate * biasGradients_[oc];
        }
    }

    void clearGradients() override {
        for (auto& row : kernelGradients_)
            for (auto& kg : row)
                kg.setZero();
        std::fill(biasGradients_.begin(), biasGradients_.end(), T(0));
        for (auto& m : v_) m.setZero();
        for (auto& m : this->output_) m.setZero();
        for (auto& m : this->gradient_) m.setZero();
    }

    // Store inputs for kernel gradient computation in backward pass
    void setLastInputs(const std::vector<Matrix<T>>& inputs) {
        lastInputs_ = inputs;
    }

private:
    void initWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        T limit = std::sqrt(T(6.0) / (kernelSize_ * kernelSize_ * (inChannels_ + outChannels_)));
        std::uniform_real_distribution<T> dist(-limit, limit);

        kernels_.resize(inChannels_);
        kernelGradients_.resize(inChannels_);
        for (int ic = 0; ic < inChannels_; ++ic) {
            kernels_[ic].resize(outChannels_);
            kernelGradients_[ic].resize(outChannels_);
            for (int oc = 0; oc < outChannels_; ++oc) {
                kernels_[ic][oc] = Matrix<T>(kernelSize_, kernelSize_);
                for (int r = 0; r < kernelSize_; ++r)
                    for (int c = 0; c < kernelSize_; ++c)
                        kernels_[ic][oc](r, c) = dist(gen);
                kernelGradients_[ic][oc] = Matrix<T>(kernelSize_, kernelSize_, T(0));
            }
        }

        biases_.resize(outChannels_, T(0));
        biasGradients_.resize(outChannels_, T(0));
    }

    int inputWidth_, inputHeight_;
    int kernelSize_;
    int inChannels_, outChannels_;
    int outputWidth_, outputHeight_;

    // weights
    std::vector<std::vector<Matrix<T>>> kernels_;  // [inChannel][outChannel]
    std::vector<T> biases_;

    // gradient accumulators
    std::vector<std::vector<Matrix<T>>> kernelGradients_;
    std::vector<T> biasGradients_;

    // intermediate values
    std::vector<Matrix<T>> v_;           // pre-activation values
    std::vector<Matrix<T>> lastInputs_;  // stored for kernel gradient computation
};

#endif // CONV_LAYER_HPP
