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
// FullyConnectedLayer — 仿射变换 + Softmax 激活
// 输入:  多个特征图，被展平为一维向量
// 输出: 1 x outputNum 概率向量（通过 Softmax）
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
        // 将所有输入特征图展平为 1×inputNum_ 行向量
        Matrix<T> flat(1, inputNum_);
        int idx = 0;
        for (const auto& fm : inputs) {
            for (int r = 0; r < fm.rows(); ++r)
                for (int c = 0; c < fm.cols(); ++c)
                    flat.data()[idx++] = fm(r, c);
        }
        lastFlatInput_ = flat;

        // 仿射: v = flat * W^T + b  (W 为 outputNum_ × inputNum_)
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
        // 对于交叉熵 + softmax，合并梯度为 (Y - target)
        // 该梯度在外部计算并以 gradOutput 形式传入
        // 存储为局部梯度
        this->gradient_[0] = gradOutput[0];

        // 累积权重梯度: dW = d * x^T  (外积)
        for (int i = 0; i < outputNum_; ++i) {
            for (int j = 0; j < inputNum_; ++j) {
                weightGradients_(i, j) += gradOutput[0].data()[i] * lastFlatInput_.data()[j];
            }
        }
        // 累积偏置梯度
        for (int i = 0; i < outputNum_; ++i)
            biasGradients_[i] += gradOutput[0].data()[i];

        // 计算相对于展平输入的梯度: d_input = d * W
        Matrix<T> gradFlat(1, inputNum_, T(0));
        for (int j = 0; j < inputNum_; ++j) {
            T sum = T(0);
            for (int i = 0; i < outputNum_; ++i)
                sum += gradOutput[0].data()[i] * weights_(i, j);
            gradFlat.data()[j] = sum;
        }

        // 重塑回特征图 — 调用者知道目标形状
        // 目前仅作为包含展平梯度的单元素向量返回
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
    Matrix<T> v_;                 // 预激活值 (logits)
    Matrix<T> weightGradients_;   // outputNum_ × inputNum_
    std::vector<T> biasGradients_;
    Matrix<T> lastFlatInput_;     // 存储的展平输入，供反向传播使用
};

#endif // FULLY_CONNECTED_LAYER_HPP
