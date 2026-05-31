#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "matrix.hpp"
#include "cnn_network.hpp"
#include "mnist_reader.hpp"

// ============================================================================
// MNIST 手写数字识别 — 纯 C++ CNN
// 5 层 LeNet-5 风格网络:
//   C1(1→6, 5×5) → S2(2×2 最大池化) → C3(6→12, 5×5) → S4(2×2 最大池化) → O5(192→10)
// ============================================================================

struct TrainingConfig {
    std::string trainImages = "mnist/train-images.idx3-ubyte";
    std::string trainLabels = "mnist/train-labels.idx1-ubyte";
    std::string testImages  = "mnist/t10k-images.idx3-ubyte";
    std::string testLabels  = "mnist/t10k-labels.idx1-ubyte";
    int epochs = 1;
    int trainLimit = 60000;   // 每 epoch 最大训练样本数
    int testLimit  = 10000;   // 最大测试样本数
    float initialAlpha = 0.03f;
    float minAlpha = 0.001f;
    bool verbose = true;
};

float trainEpoch(CNN<float>& cnn,
                 const std::vector<Matrix<float>>& trainImages,
                 const std::vector<Matrix<float>>& trainLabels,
                 int numSamples, float initialAlpha, float minAlpha)
{
    float totalLoss = 0.0f;
    int correct = 0;

    for (int n = 0; n < numSamples; ++n) {
        // 线性学习率衰减
        float alpha = initialAlpha - (initialAlpha - minAlpha) * n / (numSamples - 1);

        // 前向传播
        Matrix<float> output = cnn.forward(trainImages[n]);

        // 损失
        float loss = cnn.computeLoss(output, trainLabels[n]);
        totalLoss += loss;

        // 准确率跟踪
        if (CNN<float>::argmax(output) == CNN<float>::argmax(trainLabels[n]))
            correct++;

        // 反向传播
        cnn.backward(trainLabels[n]);

        // 更新参数
        cnn.updateParams(alpha);

        // 清除中间值，为下一个样本做准备
        cnn.clearGradients();

        if ((n + 1) % 5000 == 0) {
            std::cout << "  Processed " << (n + 1) << "/" << numSamples
                      << " | loss=" << std::fixed << std::setprecision(6) << loss
                      << " | alpha=" << std::setprecision(4) << alpha
                      << " | acc=" << std::setprecision(2)
                      << (100.0f * correct / (n + 1)) << "%" << std::endl;
        }
    }

    std::cout << "  Epoch complete: avg_loss=" << std::fixed << std::setprecision(6)
              << (totalLoss / numSamples)
              << " | accuracy=" << std::setprecision(2)
              << (100.0f * correct / numSamples) << "%" << std::endl;

    return totalLoss / numSamples;
}

float testEvaluate(CNN<float>& cnn,
                   const std::vector<Matrix<float>>& testImages,
                   const std::vector<Matrix<float>>& testLabels,
                   int numSamples)
{
    int correct = 0;
    float totalLoss = 0.0f;

    for (int i = 0; i < numSamples; ++i) {
        Matrix<float> output = cnn.forward(testImages[i]);
        totalLoss += cnn.computeLoss(output, testLabels[i]);

        if (CNN<float>::argmax(output) == CNN<float>::argmax(testLabels[i]))
            correct++;

        cnn.clearGradients();
    }

    float accuracy = 100.0f * correct / numSamples;
    std::cout << "  Test: " << correct << "/" << numSamples
              << " correct (" << std::fixed << std::setprecision(2)
              << accuracy << "%)"
              << " | avg_loss=" << std::setprecision(6)
              << (totalLoss / numSamples) << std::endl;

    return accuracy;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  CNN MNIST Digit Recognition (Pure C++)" << std::endl;
    std::cout << "  5-layer LeNet-5 style network" << std::endl;
    std::cout << "========================================" << std::endl;

    TrainingConfig cfg;

    // 简单的命令行参数覆盖
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--epochs" && i + 1 < argc)
            cfg.epochs = std::stoi(argv[++i]);
        else if (arg == "--train-limit" && i + 1 < argc)
            cfg.trainLimit = std::stoi(argv[++i]);
        else if (arg == "--test-limit" && i + 1 < argc)
            cfg.testLimit = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc)
            cfg.initialAlpha = std::stof(argv[++i]);
        else if (arg == "--train-images" && i + 1 < argc)
            cfg.trainImages = argv[++i];
        else if (arg == "--train-labels" && i + 1 < argc)
            cfg.trainLabels = argv[++i];
        else if (arg == "--test-images" && i + 1 < argc)
            cfg.testImages = argv[++i];
        else if (arg == "--test-labels" && i + 1 < argc)
            cfg.testLabels = argv[++i];
        else if (arg == "--quiet")
            cfg.verbose = false;
    }

    // 加载数据
    std::cout << "\n[1/4] Loading MNIST dataset..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<Matrix<float>> trainImages, trainLabels;
    std::vector<Matrix<float>> testImages, testLabels;

    try {
        trainImages = MnistReader::readImages(cfg.trainImages);
        trainLabels = MnistReader::readLabels(cfg.trainLabels);
        testImages  = MnistReader::readImages(cfg.testImages);
        testLabels  = MnistReader::readLabels(cfg.testLabels);
    } catch (const std::exception& e) {
        std::cerr << "Error loading MNIST data: " << e.what() << std::endl;
        std::cerr << "Please download the MNIST dataset (4 .gz files) from:" << std::endl;
        std::cerr << "  http://yann.lecun.com/exdb/mnist/" << std::endl;
        std::cerr << "Extract them and place in a 'mnist/' subdirectory." << std::endl;
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "  Loaded " << trainImages.size() << " training images, "
              << testImages.size() << " test images"
              << " (" << loadMs << " ms)" << std::endl;

    // 限制在实际数据大小范围内
    int trainN = std::min(cfg.trainLimit, static_cast<int>(trainImages.size()));
    int testN  = std::min(cfg.testLimit,  static_cast<int>(testImages.size()));
    std::cout << "  Using " << trainN << " training, " << testN << " testing" << std::endl;

    // 初始化网络
    std::cout << "\n[2/4] Initializing 5-layer CNN..." << std::endl;
    CNN<float> cnn;

    std::cout << "  Architecture:" << std::endl;
    std::cout << "    C1: Conv  1x28x28 -> 6x24x24  (5x5 kernel, ReLU)" << std::endl;
    std::cout << "    S2: MaxPool 6x24x24 -> 6x12x12  (2x2 window)" << std::endl;
    std::cout << "    C3: Conv  6x12x12 -> 12x8x8   (5x5 kernel, ReLU)" << std::endl;
    std::cout << "    S4: MaxPool 12x8x8  -> 12x4x4   (2x2 window)" << std::endl;
    std::cout << "    O5: FC    192 -> 10  (Affine + Softmax)" << std::endl;

    // 训练
    std::cout << "\n[3/4] Training (" << cfg.epochs << " epoch(s), "
              << trainN << " samples/epoch)..." << std::endl;
    auto t2 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::cout << "\n  --- Epoch " << (epoch + 1) << "/" << cfg.epochs << " ---" << std::endl;
        float avgLoss = trainEpoch(cnn, trainImages, trainLabels,
                                   trainN, cfg.initialAlpha, cfg.minAlpha);
        std::cout << "  Epoch " << (epoch + 1)
                  << " done, avg_loss=" << std::fixed << std::setprecision(6)
                  << avgLoss << std::endl;
    }

    auto t3 = std::chrono::steady_clock::now();
    auto trainMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    std::cout << "  Training time: " << (trainMs / 1000.0) << " s" << std::endl;

    // 测试
    std::cout << "\n[4/4] Evaluating on test set (" << testN << " samples)..." << std::endl;
    float accuracy = testEvaluate(cnn, testImages, testLabels, testN);

    auto t4 = std::chrono::steady_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t0).count();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Final Test Accuracy: " << std::fixed
              << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "  Total time: " << (totalMs / 1000.0) << " s" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
