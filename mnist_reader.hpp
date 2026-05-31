#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <vector>
#include <string>
#include "matrix.hpp"

// 纯 C++ MNIST 数据加载器 — 无 OpenCV 依赖
// 处理 IDX 文件格式，执行大端序到小端序的转换

class MnistReader {
public:
    // 每张图像以 1x784 形状的 Matrix<float> 返回（展平后的 28x28）
    // 像素值被归一化到 [0, 1]
    static std::vector<Matrix<float>> readImages(const std::string& filepath);

    // 每个标签以 1x10 Matrix<float> 返回（one-hot 编码）
    static std::vector<Matrix<float>> readLabels(const std::string& filepath);

private:
    static int reverseInt(int i);
};

#endif // MNIST_READER_HPP
