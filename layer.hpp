#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <memory>
#include "matrix.hpp"

// ============================================================================
// Layer - 抽象基类，演示:
//  - 纯虚接口（C++ 多态）
//  - 基于数值类型 T 的模板
//  - 通过虚析构函数实现 RAII
//  - 派生类可访问的受保护状态
// ============================================================================
template<typename T = float>
class Layer {
public:
    virtual ~Layer() = default;

    // 纯虚函数 — 每一层必须实现自己的前向/反向逻辑
    virtual std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& inputs) = 0;
    virtual std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& gradOutput) = 0;
    virtual void updateParams(T learningRate) = 0;

    // 在训练样本之间重置中间值（派生类重写）
    virtual void clearGradients() = 0;

    // 访问器
    const std::vector<Matrix<T>>& output()  const { return output_; }
    const std::vector<Matrix<T>>& gradient() const { return gradient_; }

    // 描述性名称（多态查询）
    virtual const char* name() const = 0;

protected:
    std::vector<Matrix<T>> output_;    // 前向传播结果
    std::vector<Matrix<T>> gradient_;  // 反向传播局部梯度
};

#endif // LAYER_HPP
