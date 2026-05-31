# Version 2: Pure C++ CNN — VS Code 操作手册 (Windows + MinGW)

## 你的环境

| 工具 | 路径 | 版本 |
|------|------|------|
| 编译器 (g++) | `C:\msys64\mingw64\bin\g++.exe` | 15.2.0 |
| CMake | `D:\cmake\bin\cmake.exe` | 4.2.0 |
| MNIST 数据 | `mnist/` 目录 | 已就绪 ✅ |

---

## 第一步：清理旧的 VS 构建（重要！）

旧的 `build/` 目录是用 Visual Studio 2022 生成的，和 MinGW g++ 不兼容，**必须先删掉**。

打开 **PowerShell**，逐条执行：

```powershell
cd "D:\c++\project test\version2_pure_cpp"

# 删除旧的 build 目录（用 VS 生成的，换个 MinGW 不能用）
Remove-Item -Recurse -Force build
```

---

## 第二步：确认工具可用

在同一个 PowerShell 窗口里，验证编译器能正常被 CMake 找到：

```powershell
g++ --version
```

应该显示版本号 `15.2.0`。

```powershell
cmake --version
```

应该显示 `4.2.0`。

---

## 第三步：用 MinGW 重新构建

> ⚠️ **注意**：下面的代码块是一个整体，请**一条一条**执行，不要全部复制粘贴。每次粘贴一条，等它执行完再粘下一条。

```powershell
# 1. 创建新的 build 目录并进入
mkdir build
cd build
```

```powershell
# 2. 用 MinGW Makefiles 配置项目
cmake .. -G "MinGW Makefiles"
```

这一步 CMake 会检查你的 g++ 编译器，输出类似：

```
-- The CXX compiler identification is GNU 15.2.0
-- Detecting CXX compiler ABI info - done
-- Configuring done
-- Generating done
```

如果这里报错，跳到本文末尾的"故障排查"。

```powershell
# 3. 编译（-j8 表示用 8 个线程并行编译，加快速度）
cmake --build . -j8
```

编译成功后，可执行文件在：`build\bin\cnn_mnist.exe`

---

## 第四步：运行

**编译完成后先回到项目根目录**（因为程序要从项目根目录找 `mnist/` 文件夹）：

```powershell
cd "D:\c++\project test\version2_pure_cpp"
```

### 快速测试（推荐，约 30 秒～1 分钟）

只用 5000 张图训练、1000 张图测试，先确认程序能跑通：

```powershell
.\build\bin\cnn_mnist.exe --train-limit 5000 --test-limit 1000
```

### 完整训练（约 5～15 分钟）

用全部 60000 张图训练 1 轮：

```powershell
.\build\bin\cnn_mnist.exe --epochs 1
```

### 训练 3 轮（准确率更高，约 15～45 分钟）

```powershell
.\build\bin\cnn_mnist.exe --epochs 3
```

---

## 你看到的大概是这样

```
========================================
  CNN MNIST Digit Recognition (Pure C++)
  5-layer LeNet-5 style network
========================================

[1/4] Loading MNIST dataset...
  Loaded 60000 training images, 10000 test images (XXX ms)
  Using 5000 training, 1000 testing

[2/4] Initializing 5-layer CNN...
  Architecture:
    C1: Conv  1x28x28 -> 6x24x24  (5x5 kernel, ReLU)
    S2: MaxPool 6x24x24 -> 6x12x12  (2x2 window)
    C3: Conv  6x12x12 -> 12x8x8   (5x5 kernel, ReLU)
    S4: MaxPool 12x8x8  -> 12x4x4   (2x2 window)
    O5: FC    192 -> 10  (Affine + Softmax)

[3/4] Training (1 epoch(s), 5000 samples/epoch)...
  Processed 5000/5000 | loss=0.xxxxxx | alpha=0.0010 | acc=XX.XX%

[4/4] Evaluating on test set (1000 samples)...
  Test: XXX/1000 correct (XX.XX%)

========================================
  Final Test Accuracy: XX.XX%
  Total time: XX.X s
========================================
```

---

## VS Code 里一键编译（可选，更方便）

如果你想在 VS Code 里按快捷键编译而不是每次敲命令：

1. 装 VS Code 扩展：**CMake Tools**（微软出品）
2. 按 `Ctrl+Shift+P` → 输入 `CMake: Configure`
3. 当它问选什么编译器时，选 **GCC 15.2.0**（即你的 MinGW g++）
4. 之后每次改代码，按 `F7` 就能编译
5. 编译完在 VS Code 终端里手动执行：
   ```
   .\build\bin\cnn_mnist.exe --train-limit 5000 --test-limit 1000
   ```

---

## 命令行参数速查

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--epochs N` | 训练几轮 | 1 |
| `--train-limit N` | 每轮用几张图训练 | 60000 |
| `--test-limit N` | 用几张图测试 | 10000 |
| `--lr X` | 初始学习率 | 0.03 |
| `--quiet` | 减少打印输出 | 关 |

示例：

```powershell
# 只练 2000 张，快速验证（约 20 秒）
.\build\bin\cnn_mnist.exe --train-limit 2000 --test-limit 500

# 学习率调低一点
.\build\bin\cnn_mnist.exe --train-limit 5000 --lr 0.01
```

---

## 故障排查

### 报错："CMake Error: CMake was unable to find a build program..."

MinGW 的 make 叫 `mingw32-make.exe`，CMake 可能找不到。解决：

```powershell
# 先确认它存在
where mingw32-make

# 然后用完整命令构建
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j8
```

编译成功后 exe 路径是一样的：`build\bin\cnn_mnist.exe`

### 报错："Cannot open file: mnist/train-images.idx3-ubyte"

说明你没在项目根目录运行。确保先 `cd "D:\c++\project test\version2_pure_cpp"` 再执行 `.\build\bin\cnn_mnist.exe`。

### 报错：g++ 找不到

关闭 PowerShell，重新打开一个新的 PowerShell 窗口，再试。如果还不行，手动把 MinGW 加到 PATH：

```powershell
$env:Path += ";C:\msys64\mingw64\bin"
g++ --version   # 验证
```

### 训练太慢

纯 CPU、纯 C++，本来就慢，这是正常的。先加 `--train-limit 2000` 验证能跑通，确认没问题后再跑大的。

---

## 总结：你每次重新编译的步骤

以后改了代码想重新编译运行，只需要：

```powershell
cd "D:\c++\project test\version2_pure_cpp\build"
cmake --build . -j8
cd ..
.\build\bin\cnn_mnist.exe --train-limit 5000 --test-limit 1000
```

就这三条命令。
