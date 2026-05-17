#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <memory>
#include "matrix.hpp"

// ============================================================================
// Layer - Abstract base class demonstrating:
//  - Pure virtual interface (C++ polymorphism)
//  - Template on numeric type T
//  - RAII via virtual destructor
//  - Protected state accessible to derived classes
// ============================================================================
template<typename T = float>
class Layer {
public:
    virtual ~Layer() = default;

    // Pure virtual — each layer must implement its own forward/backward logic
    virtual std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& inputs) = 0;
    virtual std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& gradOutput) = 0;
    virtual void updateParams(T learningRate) = 0;

    // Reset intermediate values between training samples (derived classes override)
    virtual void clearGradients() = 0;

    // Accessors
    const std::vector<Matrix<T>>& output()  const { return output_; }
    const std::vector<Matrix<T>>& gradient() const { return gradient_; }

    // Descriptive name (polymorphic query)
    virtual const char* name() const = 0;

protected:
    std::vector<Matrix<T>> output_;    // forward-pass results
    std::vector<Matrix<T>> gradient_;  // backward-pass local gradients
};

#endif // LAYER_HPP
