# Version 2: Pure C++ CNN 手写数字识别 — 使用手册

## 1. 概述

本版本使用**纯 C++ 标准库**实现了一个 5 层 LeNet-5 风格卷积神经网络，用于 MNIST 手写数字（0~9）识别。**不依赖任何第三方库**（无 OpenCV，无 Eigen，无 BLAS），仅使用 C++17 标准库。

### 网络结构

| 层 | 类型 | 输入 | 输出 | 参数 |
|----|------|------|------|------|
| C1 | 卷积层 (Valid, ReLU) | 1×28×28 | 6×24×24 | 5×5 卷积核×6, 偏置×6 |
| S2 | 最大池化层 | 6×24×24 | 6×12×12 | 2×2 窗口 |
| C3 | 卷积层 (Valid, ReLU) | 6×12×12 | 12×8×8 | 5×5 卷积核×72, 偏置×12 |
| S4 | 最大池化层 | 12×8×8 | 12×4×4 | 2×2 窗口 |
| O5 | 全连接层 (Softmax) | 192 | 10 | 权重 192×10, 偏置×10 |

### 理论准确率

参照源文章，该网络在 MNIST 测试集上可达 **约 98.3%** 的识别准确率。

---

## 2. 项目文件结构

```
version2_pure_cpp/
├── matrix.hpp                  # 矩阵类模板 (RAII, 移动语义, 运算符重载)
├── layer.hpp                   # 层抽象基类 (运行时多态)
├── activations.hpp             # 激活函数 (ReLU, Sigmoid, Softmax)
├── loss.hpp                    # 损失函数 (交叉熵 + 梯度)
├── conv_layer.hpp              # 卷积层实现 (前向/反向传播, 参数更新)
├── pool_layer.hpp              # 池化层实现 (最大池化/均值池化)
├── fully_connected_layer.hpp   # 全连接层实现 (仿射变换 + Softmax)
├── cnn_network.hpp             # 5层 CNN 网络组装
├── mnist_reader.hpp            # MNIST 数据读取器 (纯二进制解析)
├── mnist_reader.cpp            # MNIST 数据读取器实现
├── main.cpp                    # 训练与测试主程序
├── CMakeLists.txt              # CMake 构建配置
└── VERSION2_USAGE_MANUAL.md    # 本使用手册
```

---

## 3. 环境要求

### 编译器

- **C++17** 或更高版本
- 支持的编译器:
  - **GCC** ≥ 8.0
  - **Clang** ≥ 7.0
  - **MSVC** ≥ 2017 (Visual Studio 2017+)

### 构建工具

- **CMake** ≥ 3.16

### 依赖库

- **无** — 仅使用 C++ 标准库 (`<memory>`, `<vector>`, `<random>`, `<cmath>`, `<fstream>` 等)

---

## 4. 获取 MNIST 数据集

### 4.1 下载

从 Yann LeCun 官网下载 MNIST 数据集（4个文件）：

```
http://yann.lecun.com/exdb/mnist/
```

需要下载以下文件：

| 文件名 | 用途 | 大小 |
|--------|------|------|
| `train-images-idx3-ubyte.gz` | 训练图像 (60000张) | ~9.5 MB |
| `train-labels-idx1-ubyte.gz` | 训练标签 (60000个) | ~29 KB |
| `t10k-images-idx3-ubyte.gz` | 测试图像 (10000张) | ~1.6 MB |
| `t10k-labels-idx1-ubyte.gz` | 测试标签 (10000个) | ~4.5 KB |

### 4.2 解压并放置

将下载的 `.gz` 文件解压，得到 `.idx3-ubyte` 和 `.idx1-ubyte` 文件，放入项目目录下的 `mnist/` 文件夹中：

```
version2_pure_cpp/
└── mnist/
    ├── train-images.idx3-ubyte
    ├── train-labels.idx1-ubyte
    ├── t10k-images.idx3-ubyte
    └── t10k-labels.idx1-ubyte
```

解压命令（Windows 可使用 7-Zip，Linux/macOS 使用 gunzip）：

```bash
# Linux / macOS / MSYS2
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

# 重命名 (去掉 -idx 中的 -)
mv train-images-idx3-ubyte train-images.idx3-ubyte
mv train-labels-idx1-ubyte train-labels.idx1-ubyte
mv t10k-images-idx3-ubyte t10k-images.idx3-ubyte
mv t10k-labels-idx1-ubyte t10k-labels.idx1-ubyte
```

---

## 5. 编译与构建

### 5.1 使用 CMake (推荐)

```bash
# 进入 version2_pure_cpp 目录
cd version2_pure_cpp

# 创建构建目录
mkdir build && cd build

# 配置 (Windows MinGW)
cmake .. -G "MinGW Makefiles"

# 配置 (Linux / macOS)
cmake ..

# 编译
cmake --build . -j$(nproc)
```

### 5.2 使用 Visual Studio

```bash
cd version2_pure_cpp
mkdir build && cd build

# 生成 Visual Studio 解决方案
cmake .. -G "Visual Studio 17 2022"

# 编译
cmake --build . --config Release
```

### 5.3 手动编译 (无 CMake)

```bash
# GCC
g++ -std=c++17 -O3 -march=native -o cnn_mnist main.cpp mnist_reader.cpp

# MSVC
cl /std:c++17 /O2 /arch:AVX2 /Fe:cnn_mnist.exe main.cpp mnist_reader.cpp
```

编译后在 `build/bin/` 目录下生成 `cnn_mnist.exe`（Windows）或 `cnn_mnist`（Linux/macOS）。

---

## 6. 运行

### 6.1 基本运行

```bash
# 在 version2_pure_cpp 目录下运行 (确保 mnist/ 文件夹存在)
cd version2_pure_cpp
./build/bin/cnn_mnist
```

### 6.2 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs <N>` | 训练轮数 | 1 |
| `--train-limit <N>` | 每轮训练样本数 | 60000 |
| `--test-limit <N>` | 测试样本数 | 10000 |
| `--lr <value>` | 初始学习率 | 0.03 |
| `--train-images <path>` | 训练图像文件路径 | `mnist/train-images.idx3-ubyte` |
| `--train-labels <path>` | 训练标签文件路径 | `mnist/train-labels.idx1-ubyte` |
| `--test-images <path>` | 测试图像文件路径 | `mnist/t10k-images.idx3-ubyte` |
| `--test-labels <path>` | 测试标签文件路径 | `mnist/t10k-labels.idx1-ubyte` |
| `--quiet` | 减少输出 | false |

### 6.3 示例运行

```bash
# 训练 3 轮，使用全部 60000 张训练图像
./build/bin/cnn_mnist --epochs 3

# 快速测试：只用 5000 张图像训练
./build/bin/cnn_mnist --train-limit 5000 --epochs 1

# 自定义学习率
./build/bin/cnn_mnist --lr 0.01 --epochs 2

# 指定自定义数据路径
./build/bin/cnn_mnist \
    --train-images /path/to/train-images.idx3-ubyte \
    --train-labels /path/to/train-labels.idx1-ubyte \
    --test-images  /path/to/t10k-images.idx3-ubyte \
    --test-labels  /path/to/t10k-labels.idx1-ubyte
```

### 6.4 预期输出

```
========================================
  CNN MNIST Digit Recognition (Pure C++)
  5-layer LeNet-5 style network
========================================

[1/4] Loading MNIST dataset...
  Loaded 60000 training images, 10000 test images (1234 ms)
  Using 60000 training, 10000 testing

[2/4] Initializing 5-layer CNN...
  Architecture:
    C1: Conv  1x28x28 -> 6x24x24  (5x5 kernel, ReLU)
    S2: MaxPool 6x24x24 -> 6x12x12  (2x2 window)
    C3: Conv  6x12x12 -> 12x8x8   (5x5 kernel, ReLU)
    S4: MaxPool 12x8x8  -> 12x4x4   (2x2 window)
    O5: FC    192 -> 10  (Affine + Softmax)

[3/4] Training (1 epoch(s), 60000 samples/epoch)...
  --- Epoch 1/1 ---
  Processed 5000/60000 | loss=0.234567 | alpha=0.0276 | acc=87.34%
  Processed 10000/60000 | loss=0.123456 | alpha=0.0251 | acc=89.12%
  ...
  Epoch complete: avg_loss=0.089234 | accuracy=95.67%

[4/4] Evaluating on test set (10000 samples)...
  Test: 9830/10000 correct (98.30%) | avg_loss=0.092145

========================================
  Final Test Accuracy: 98.30%
  Total time: 342.5 s
========================================
```

> **注意**：纯 C++ 实现不使用任何加速库（如 BLAS、OpenCV），训练 60000 张图像可能需要较长时间（取决于 CPU 性能）。建议首次测试时使用 `--train-limit 5000` 先快速验证。

---

## 7. 代码架构说明

### 7.1 核心类

#### Matrix\<T\> (`matrix.hpp`)

自定义矩阵类，体现了以下 C++ 核心概念：

- **RAII**: 使用 `std::unique_ptr<T[]>` 管理动态内存，自动释放
- **移动语义**: 高效的资源所有权转移
- **运算符重载**: 自然语法的矩阵运算
- **模板**: 泛型元素类型
- **const 正确性**: 分离 const/non-const 访问路径

```cpp
// 创建 3x3 矩阵
Matrix<float> m(3, 3, 0.0f);
m(0, 0) = 1.0f;           // 元素访问
Matrix<float> m2 = m * 2.0f;  // 标量乘法
Matrix<float> m3 = m.apply([](float x) { return x > 0 ? x : 0; });  // ReLU
```

#### Layer\<T\> (`layer.hpp`)

抽象基类，定义层接口（运行时多态）：

```cpp
template<typename T>
class Layer {
public:
    virtual std::vector<Matrix<T>> forward(const std::vector<Matrix<T>>& inputs) = 0;
    virtual std::vector<Matrix<T>> backward(const std::vector<Matrix<T>>& gradOutput) = 0;
    virtual void updateParams(T learningRate) = 0;
    virtual void clearGradients() = 0;
};
```

#### ConvLayer\<T\> (`conv_layer.hpp`)

卷积层，核心算法：

- **前向传播**: 输入与卷积核进行 Valid 模式相关运算 → 加偏置 → ReLU
- **反向传播**:
  - 局部梯度 = 上游梯度 ⊙ ReLU'(输出)
  - 卷积核梯度 = 输入 ⊗ 局部梯度 (Valid 相关)
  - 传递给前层的梯度 = rotate180(卷积核) ⊛ 局部梯度 (Full 卷积)
- **参数更新**: W = W - α·dE/dW, b = b - α·dE/db

#### PoolLayer\<T\> (`pool_layer.hpp`)

池化层：

- **前向传播**: 2×2 窗口下采样（最大池化记录最大值位置）
- **反向传播**: 上采样（最大池化将梯度放回最大值位置；均值池化均匀分配）

#### FullyConnectedLayer\<T\> (`fully_connected_layer.hpp`)

全连接层 (Affine + Softmax)：

- **前向传播**: 展平输入 → 仿射变换 (y = Wx + b) → Softmax
- **反向传播**: 交叉熵 + Softmax 联合梯度 (Y - t)

### 7.2 CNN 网络 (`cnn_network.hpp`)

组装 5 层网络，管理：

1. 前向传播顺序: C1 → S2 → C3 → S4 → O5
2. 反向传播顺序: O5 → S4 → C3 → S2 → C1
3. S4↔O5 之间的数据展平/重塑
4. 参数更新和梯度清零

### 7.3 MNIST 数据读取 (`mnist_reader.hpp/cpp`)

纯 C++ 二进制文件解析：

- 处理 IDX 文件格式（magic number 校验）
- **大端序 → 小端序** 字节序转换（Intel 处理器必需）
- 图像像素值归一化到 [0, 1]
- 标签转换为 one-hot 编码 (0 → [1,0,0,...,0])

---

## 8. 训练算法

### 8.1 优化方法

- **梯度下降法** (Stochastic Gradient Descent, 单样本模式)
- **学习率衰减**: α 从初始值线性递减至最小值
  ```
  α(n) = α_initial - (α_initial - α_min) * n / (N - 1)
  ```
  默认: α_initial = 0.03, α_min = 0.001

### 8.2 权重初始化

使用 Xavier/Glorot 均匀分布初始化：

```
limit = sqrt(6.0 / (fan_in + fan_out))
W ~ Uniform(-limit, +limit)
```

### 8.3 损失函数

交叉熵误差 (Cross-Entropy Loss):

```
L = -Σ t_i * log(Y_i)
```

其中 t 为 one-hot 标签，Y 为 Softmax 输出概率。

---

## 9. 性能说明

### 9.1 时间复杂度

纯 C++ 实现（无 BLAS 加速）:
- 单张图像前向传播: ~2-3 百万次浮点运算
- 60000 张图像训练 1 轮: 约 5-10 分钟（取决于 CPU）

### 9.2 优化建议

如需加速，可以考虑：
- 启用编译器优化 (`-O3 -march=native` / `/O2 /arch:AVX2`)
- 减少训练样本数 (`--train-limit 10000`)
- 后续版本可使用 OpenMP 并行化卷积运算

---

## 10. 与 Version1 (OpenCV) 版本对比

| 特性 | Version 2 (本版本) | Version 1 |
|------|-------------------|-----------|
| 依赖 | 仅 C++ 标准库 | C++ 标准库 + OpenCV |
| 图像读取 | 自定义二进制解析 | cv::imread / 自定义解析 |
| 矩阵运算 | 自定义 Matrix 类 | cv::Mat |
| 卷积运算 | 纯 C++ 四重循环 | cv::filter2D |
| 编译复杂度 | 低 (无外部依赖) | 中 (需安装 OpenCV) |
| 运行速度 | 较慢 (无加速) | 较快 (OpenCV 优化) |
| 代码可读性 | 算法细节清晰 | 调用 OpenCV API |
| 学习价值 | 高 (理解底层原理) | 中 (理解框架使用) |

---

## 11. 常见问题

### Q: 运行时提示 "Cannot open file"

**A:** 确保已下载 MNIST 数据集并解压到 `mnist/` 目录。也可通过 `--train-images` 等参数指定自定义路径。

### Q: 训练速度慢

**A:** 纯 C++ CPU 运算天然较慢。建议：
1. 使用 `--train-limit 5000` 先验证功能
2. 确保编译器优化已开启
3. 等待完整训练 60000 张图像（约 5-10 分钟）

### Q: 准确率低于预期

**A:** 可能原因：
1. 训练样本太少 → 增加 `--train-limit` 或 `--epochs`
2. 学习率不合适 → 调整 `--lr`（建议范围 0.001~0.1）

### Q: 编译错误 "C++17 required"

**A:** 确保编译器支持 C++17:
- GCC: `g++ -std=c++17`
- MSVC: VS 2017 或更新版本
- Clang: `clang++ -std=c++17`

---

## 12. 技术要点总结

本实现涵盖了以下 C++ 和深度学习核心知识点：

**C++ 方面**:
- RAII 资源管理 (`std::unique_ptr`)
- 移动语义与完美转发
- 模板元编程
- 运算符重载
- 运行时多态 (虚函数)
- const 正确性
- 智能指针
- 随机数生成 (`<random>`)

**深度学习方面**:
- 卷积运算 (Valid/Full/Same 模式)
- ReLU 激活函数
- 最大池化与均值池化
- Softmax 多分类
- 交叉熵损失函数
- 误反向传播算法
- 链式求导法则
- 梯度下降优化
- Xavier 权重初始化
