#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <vector>
#include <string>
#include "matrix.hpp"

// Pure C++ MNIST data loader — no OpenCV dependency
// Handles IDX file format with big-endian to little-endian conversion

class MnistReader {
public:
    // Each image is returned as a Matrix<float> with shape 1x784 (flattened 28x28)
    // Pixel values are normalized to [0, 1]
    static std::vector<Matrix<float>> readImages(const std::string& filepath);

    // Each label is returned as a 1x10 Matrix<float> (one-hot encoded)
    static std::vector<Matrix<float>> readLabels(const std::string& filepath);

private:
    static int reverseInt(int i);
};

#endif // MNIST_READER_HPP
