#include "mnist_reader.hpp"

#include <fstream>
#include <stdexcept>
#include <cstdint>

int MnistReader::reverseInt(int i) {
    unsigned char c1 = i & 0xff;
    unsigned char c2 = (i >> 8) & 0xff;
    unsigned char c3 = (i >> 16) & 0xff;
    unsigned char c4 = (i >> 24) & 0xff;
    return (static_cast<int>(c1) << 24) +
           (static_cast<int>(c2) << 16) +
           (static_cast<int>(c3) << 8) +
           static_cast<int>(c4);
}

std::vector<Matrix<float>> MnistReader::readImages(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filepath);

    int magic = 0, numImages = 0, rows = 0, cols = 0;

    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    magic = reverseInt(magic);
    if (magic != 2051)
        throw std::runtime_error("Invalid image file magic number: " + std::to_string(magic));

    file.read(reinterpret_cast<char*>(&numImages), sizeof(int));
    numImages = reverseInt(numImages);

    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    rows = reverseInt(rows);

    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    cols = reverseInt(cols);

    int imgSize = rows * cols;
    std::vector<unsigned char> buffer(imgSize);
    std::vector<Matrix<float>> images;
    images.reserve(numImages);

    for (int i = 0; i < numImages; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), imgSize);
        Matrix<float> img(1, imgSize);  // flattened row vector
        for (int j = 0; j < imgSize; ++j)
            img.data()[j] = static_cast<float>(buffer[j]) / 255.0f;
        images.push_back(std::move(img));
    }

    return images;
}

std::vector<Matrix<float>> MnistReader::readLabels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filepath);

    int magic = 0, numLabels = 0;

    file.read(reinterpret_cast<char*>(&magic), sizeof(int));
    magic = reverseInt(magic);
    if (magic != 2049)
        throw std::runtime_error("Invalid label file magic number: " + std::to_string(magic));

    file.read(reinterpret_cast<char*>(&numLabels), sizeof(int));
    numLabels = reverseInt(numLabels);

    std::vector<Matrix<float>> labels;
    labels.reserve(numLabels);

    for (int i = 0; i < numLabels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        Matrix<float> oneHot(1, 10, 0.0f);
        if (label < 10)
            oneHot.data()[label] = 1.0f;
        labels.push_back(std::move(oneHot));
    }

    return labels;
}
