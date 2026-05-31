#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <memory>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <functional>

// ============================================================================
// Matrix - 自定义轻量级矩阵类，演示:
//  - RAII: 动态内存由 unique_ptr<T[]> 管理，析构时自动释放
//  - 移动语义: 高效的所有权转移
//  - 运算符重载: 算术运算的自然语法
//  - 模板: 基于元素类型 T 的泛型
//  - const 正确性: 分离的 const/非const 访问路径
// ============================================================================
template<typename T = float>
class Matrix {
public:
    // ---------- 构造函数 ----------

    Matrix()
        : rows_(0), cols_(0), size_(0)
    {}

    Matrix(int rows, int cols)
        : rows_(rows), cols_(cols), size_(rows * cols)
    {
        if (rows <= 0 || cols <= 0)
            throw std::invalid_argument("Matrix dimensions must be positive");
        data_ = std::make_unique<T[]>(size_);
        std::fill_n(data_.get(), size_, T(0));
    }

    Matrix(int rows, int cols, T fillValue)
        : rows_(rows), cols_(cols), size_(rows * cols)
    {
        if (rows <= 0 || cols <= 0)
            throw std::invalid_argument("Matrix dimensions must be positive");
        data_ = std::make_unique<T[]>(size_);
        std::fill_n(data_.get(), size_, fillValue);
    }

    Matrix(int rows, int cols, std::initializer_list<T> values)
        : rows_(rows), cols_(cols), size_(rows * cols)
    {
        if (rows <= 0 || cols <= 0)
            throw std::invalid_argument("Matrix dimensions must be positive");
        data_ = std::make_unique<T[]>(size_);
        size_t n = std::min(values.size(), static_cast<size_t>(size_));
        std::copy_n(values.begin(), n, data_.get());
        if (n < static_cast<size_t>(size_))
            std::fill_n(data_.get() + n, size_ - n, T(0));
    }

    // 拷贝构造函数
    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), size_(other.size_)
    {
        if (size_ > 0) {
            data_ = std::make_unique<T[]>(size_);
            std::copy_n(other.data_.get(), size_, data_.get());
        }
    }

    // 移动构造函数（noexcept 以保证 STL 兼容性）
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_))
        , rows_(other.rows_), cols_(other.cols_), size_(other.size_)
    {
        other.rows_ = 0;
        other.cols_ = 0;
        other.size_ = 0;
    }

    // ---------- 赋值运算符 ----------

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            size_ = other.size_;
            if (size_ > 0) {
                data_ = std::make_unique<T[]>(size_);
                std::copy_n(other.data_.get(), size_, data_.get());
            } else {
                data_.reset();
            }
        }
        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            size_ = other.size_;
            other.rows_ = 0;
            other.cols_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    // ---------- 元素访问 ----------

    T& operator()(int r, int c) {
        return data_[r * cols_ + c];
    }

    const T& operator()(int r, int c) const {
        return data_[r * cols_ + c];
    }

    T* data()             { return data_.get(); }
    const T* data() const { return data_.get(); }

    // ---------- 维度 ----------

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return size_; }
    bool empty() const { return size_ == 0; }

    // ---------- 填充 ----------

    void fill(T value) {
        std::fill_n(data_.get(), size_, value);
    }

    void setZero() {
        std::fill_n(data_.get(), size_, T(0));
    }

    // ---------- 原地算术运算 ----------

    Matrix& operator+=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Matrix::operator+= : dimension mismatch");
        for (int i = 0; i < size_; ++i)
            data_[i] += other.data_[i];
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Matrix::operator-= : dimension mismatch");
        for (int i = 0; i < size_; ++i)
            data_[i] -= other.data_[i];
        return *this;
    }

    Matrix& operator*=(T scalar) {
        for (int i = 0; i < size_; ++i)
            data_[i] *= scalar;
        return *this;
    }

    Matrix& operator/=(T scalar) {
        for (int i = 0; i < size_; ++i)
            data_[i] /= scalar;
        return *this;
    }

    // ---------- 子矩阵提取 ----------

    Matrix subMatrix(int startRow, int startCol, int numRows, int numCols) const {
        if (startRow < 0 || startCol < 0 ||
            startRow + numRows > rows_ || startCol + numCols > cols_)
            throw std::out_of_range("Matrix::subMatrix : index out of range");
        Matrix<T> result(numRows, numCols);
        for (int r = 0; r < numRows; ++r)
            for (int c = 0; c < numCols; ++c)
                result(r, c) = (*this)(startRow + r, startCol + c);
        return result;
    }

    // ---------- 填充 ----------

    Matrix pad(int topPad, int bottomPad, int leftPad, int rightPad) const {
        Matrix<T> result(rows_ + topPad + bottomPad, cols_ + leftPad + rightPad, T(0));
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(r + topPad, c + leftPad) = (*this)(r, c);
        return result;
    }

    // ---------- 重塑 ----------

    Matrix reshape(int newRows, int newCols) const {
        if (newRows * newCols != size_)
            throw std::invalid_argument("Matrix::reshape : total size must match");
        Matrix<T> result(newRows, newCols);
        std::copy_n(data_.get(), size_, result.data_.get());
        return result;
    }

    // ---------- 逐元素运算（静态辅助方法风格）----------

    static Matrix elementWiseMultiply(const Matrix& a, const Matrix& b) {
        if (a.rows_ != b.rows_ || a.cols_ != b.cols_)
            throw std::invalid_argument("elementWiseMultiply: dimension mismatch");
        Matrix<T> result(a.rows_, a.cols_);
        for (int i = 0; i < a.size_; ++i)
            result.data_[i] = a.data_[i] * b.data_[i];
        return result;
    }

    Matrix elementWiseMultiply(const Matrix& other) const {
        return elementWiseMultiply(*this, other);
    }

    // ---------- 所有元素求和 ----------

    T sum() const {
        T s = T(0);
        for (int i = 0; i < size_; ++i)
            s += data_[i];
        return s;
    }

    // ---------- 按行求和（返回列向量）----------

    Matrix sumOverRows() const {
        Matrix<T> result(1, cols_, T(0));
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(0, c) += (*this)(r, c);
        return result;
    }

    // ---------- 逐元素应用函数 ----------

    template<typename Func>
    Matrix apply(Func&& f) const {
        Matrix<T> result(rows_, cols_);
        for (int i = 0; i < size_; ++i)
            result.data_[i] = f(data_[i]);
        return result;
    }

    template<typename Func>
    void applyInPlace(Func&& f) {
        for (int i = 0; i < size_; ++i)
            data_[i] = f(data_[i]);
    }

    // ---------- 转置 ----------

    Matrix transpose() const {
        Matrix<T> result(cols_, rows_);
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(c, r) = (*this)(r, c);
        return result;
    }

    // ---------- 旋转 180 度 ----------

    Matrix rotate180() const {
        Matrix<T> result(rows_, cols_);
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(r, c) = (*this)(rows_ - 1 - r, cols_ - 1 - c);
        return result;
    }

    // ---------- 展平为一维 ----------

    Matrix flattenToRowVector() const {
        Matrix<T> result(1, size_);
        std::copy_n(data_.get(), size_, result.data());
        return result;
    }

    // ---------- 打印（调试）----------

    void print(std::ostream& os = std::cout, int maxRows = 5, int maxCols = 8) const {
        os << "Matrix(" << rows_ << "x" << cols_ << ")\n";
        int rlim = std::min(rows_, maxRows);
        int clim = std::min(cols_, maxCols);
        for (int r = 0; r < rlim; ++r) {
            os << "  ";
            for (int c = 0; c < clim; ++c) {
                os << (*this)(r, c);
                if (c < clim - 1) os << "\t";
            }
            if (clim < cols_) os << " ...";
            os << "\n";
        }
        if (rlim < rows_) os << "  ...\n";
    }

private:
    std::unique_ptr<T[]> data_;  // RAII 管理的缓冲区
    int rows_;
    int cols_;
    int size_;
};

// 自由函数运算符重载
template<typename T>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> result(a);
    result += b;
    return result;
}

template<typename T>
Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> result(a);
    result -= b;
    return result;
}

template<typename T>
Matrix<T> operator*(const Matrix<T>& m, T scalar) {
    Matrix<T> result(m);
    result *= scalar;
    return result;
}

template<typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& m) {
    return m * scalar;
}

#endif // MATRIX_HPP