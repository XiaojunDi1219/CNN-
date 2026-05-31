#ifndef POOL_LAYER_HPP
#define POOL_LAYER_HPP

#include <vector>
#include <stdexcept>
#include <limits>
#include "matrix.hpp"
#include "layer.hpp"

enum class PoolType { Max, Avg };

// ============================================================================
// PoolLayer — 2x2 最大池化或平均池化
// 前向:  按 poolSize 因子下采样（默认 2）
// 反向: 上采样 — 将梯度分配回输入位置
// ============================================================================
template<typename T = float>
class PoolLayer : public Layer<T> {
public:
    PoolLayer(int inputWidth, int inputHeight, int poolSize,
              int inChannels, int outChannels, PoolType poolType = PoolType::Max)
        : inputWidth_(inputWidth), inputHeight_(inputHeight),
          poolSize_(poolSize),
          inChannels_(inChannels), outChannels_(outChannels),
          poolType_(poolType),
          outputWidth_(inputWidth / poolSize),
          outputHeight_(inputHeight / poolSize)
    {
        if (inputWidth % poolSize != 0 || inputHeight % poolSize != 0)
            throw std::invalid_argument("PoolLayer: input dimensions must be divisible by pool size");

        this->output_.resize(outChannels_, Matrix<T>(outputHeight_, outputWidth_, T(0)));
        this->gradient_.resize(outChannels_, Matrix<T>(outputHeight_, outputWidth_, T(0)));

        if (poolType_ == PoolType::Max)
            maxPositions_.resize(outChannels_, Matrix<int>(outputHeight_, outputWidth_, 0));
    }

    const char* name() const override {
        return poolType_ == PoolType::Max ? "MaxPoolLayer" : "AvgPoolLayer";
    }

    std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& inputs) override {
        if (static_cast<int>(inputs.size()) != inChannels_)
            throw std::invalid_argument("PoolLayer::forward: wrong input channel count");

        for (int ch = 0; ch < outChannels_; ++ch) {
            const Matrix<T>& in = inputs[ch];
            Matrix<T>& out = this->output_[ch];

            for (int or_ = 0; or_ < outputHeight_; ++or_) {
                for (int oc = 0; oc < outputWidth_; ++oc) {
                    int startR = or_ * poolSize_;
                    int startC = oc * poolSize_;

                    if (poolType_ == PoolType::Max) {
                        T maxVal = -std::numeric_limits<T>::max();
                        int maxIdx = 0;
                        for (int pr = 0; pr < poolSize_; ++pr) {
                            for (int pc = 0; pc < poolSize_; ++pc) {
                                T val = in(startR + pr, startC + pc);
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxIdx = pr * poolSize_ + pc;
                                }
                            }
                        }
                        out(or_, oc) = maxVal;
                        maxPositions_[ch](or_, oc) = maxIdx;
                    } else { // Avg
                        T sum = T(0);
                        for (int pr = 0; pr < poolSize_; ++pr)
                            for (int pc = 0; pc < poolSize_; ++pc)
                                sum += in(startR + pr, startC + pc);
                        out(or_, oc) = sum / static_cast<T>(poolSize_ * poolSize_);
                    }
                }
            }
        }
        return this->output_;
    }

    std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& gradOutput) override {
        if (static_cast<int>(gradOutput.size()) != outChannels_)
            throw std::invalid_argument("PoolLayer::backward: wrong grad output channel count");

        // 存储局部梯度
        for (int ch = 0; ch < outChannels_; ++ch)
            this->gradient_[ch] = gradOutput[ch];

        // 上采样到输入尺寸
        std::vector<Matrix<T>> gradInput(inChannels_);
        for (int ch = 0; ch < inChannels_; ++ch) {
            gradInput[ch] = Matrix<T>(inputHeight_, inputWidth_, T(0));

            for (int or_ = 0; or_ < outputHeight_; ++or_) {
                for (int oc = 0; oc < outputWidth_; ++oc) {
                    int startR = or_ * poolSize_;
                    int startC = oc * poolSize_;
                    T gVal = gradOutput[ch](or_, oc);

                    if (poolType_ == PoolType::Max) {
                        int pos = maxPositions_[ch](or_, oc);
                        int pr = pos / poolSize_;
                        int pc = pos % poolSize_;
                        gradInput[ch](startR + pr, startC + pc) = gVal;
                    } else { // 平均池化: 均匀分配
                        T avgGrad = gVal / static_cast<T>(poolSize_ * poolSize_);
                        for (int pr = 0; pr < poolSize_; ++pr)
                            for (int pc = 0; pc < poolSize_; ++pc)
                                gradInput[ch](startR + pr, startC + pc) = avgGrad;
                    }
                }
            }
        }
        return gradInput;
    }

    void updateParams(T /*learningRate*/) override {
        // 池化层没有可训练参数
    }

    void clearGradients() override {
        for (auto& m : this->output_) m.setZero();
        for (auto& m : this->gradient_) m.setZero();
    }

private:
    int inputWidth_, inputHeight_;
    int poolSize_;
    int inChannels_, outChannels_;
    PoolType poolType_;
    int outputWidth_, outputHeight_;
    std::vector<Matrix<int>> maxPositions_;  // 用于最大池化的反向传播
};

#endif // POOL_LAYER_HPP
