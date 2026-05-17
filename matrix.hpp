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
// Matrix - Custom lightweight matrix class demonstrating:
//  - RAII: dynamic memory owned by unique_ptr<T[]>, auto-released on destruction
//  - Move semantics: efficient transfer of ownership
//  - Operator overloading: natural syntax for arithmetic
//  - Templates: generic over element type T
//  - const correctness: separate const/non-const access paths
// ============================================================================
template<typename T = float>
class Matrix {
public:
    // ---------- constructors ----------

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

    // copy constructor
    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), size_(other.size_)
    {
        if (size_ > 0) {
            data_ = std::make_unique<T[]>(size_);
            std::copy_n(other.data_.get(), size_, data_.get());
        }
    }

    // move constructor (noexcept for STL compatibility)
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_))
        , rows_(other.rows_), cols_(other.cols_), size_(other.size_)
    {
        other.rows_ = 0;
        other.cols_ = 0;
        other.size_ = 0;
    }

    // ---------- assignment operators ----------

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

    // ---------- element access ----------

    T& operator()(int r, int c) {
        return data_[r * cols_ + c];
    }

    const T& operator()(int r, int c) const {
        return data_[r * cols_ + c];
    }

    T* data()             { return data_.get(); }
    const T* data() const { return data_.get(); }

    // ---------- dimensions ----------

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return size_; }
    bool empty() const { return size_ == 0; }

    // ---------- fill ----------

    void fill(T value) {
        std::fill_n(data_.get(), size_, value);
    }

    void setZero() {
        std::fill_n(data_.get(), size_, T(0));
    }

    // ---------- in-place arithmetic ----------

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

    // ---------- sub-matrix extraction ----------

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

    // ---------- padding ----------

    Matrix pad(int topPad, int bottomPad, int leftPad, int rightPad) const {
        Matrix<T> result(rows_ + topPad + bottomPad, cols_ + leftPad + rightPad, T(0));
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(r + topPad, c + leftPad) = (*this)(r, c);
        return result;
    }

    // ---------- reshaping ----------

    Matrix reshape(int newRows, int newCols) const {
        if (newRows * newCols != size_)
            throw std::invalid_argument("Matrix::reshape : total size must match");
        Matrix<T> result(newRows, newCols);
        std::copy_n(data_.get(), size_, result.data_.get());
        return result;
    }

    // ---------- element-wise operations (static helper style) ----------

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

    // ---------- sum of all elements ----------

    T sum() const {
        T s = T(0);
        for (int i = 0; i < size_; ++i)
            s += data_[i];
        return s;
    }

    // ---------- sum over rows (return column vector) ----------

    Matrix sumOverRows() const {
        Matrix<T> result(1, cols_, T(0));
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(0, c) += (*this)(r, c);
        return result;
    }

    // ---------- apply function element-wise ----------

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

    // ---------- transpose ----------

    Matrix transpose() const {
        Matrix<T> result(cols_, rows_);
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(c, r) = (*this)(r, c);
        return result;
    }

    // ---------- rotate 180 degrees ----------

    Matrix rotate180() const {
        Matrix<T> result(rows_, cols_);
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                result(r, c) = (*this)(rows_ - 1 - r, cols_ - 1 - c);
        return result;
    }

    // ---------- flatten to 1D ----------

    Matrix flattenToRowVector() const {
        Matrix<T> result(1, size_);
        std::copy_n(data_.get(), size_, result.data());
        return result;
    }

    // ---------- print (debug) ----------

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
    std::unique_ptr<T[]> data_;  // RAII-managed buffer
    int rows_;
    int cols_;
    int size_;
};

// Free-function operator overloads
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
