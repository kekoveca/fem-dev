#pragma once

#include <stdexcept>
#include <vector>

template <typename T>
class DenseMatrix
{
public:
    DenseMatrix(std::size_t rows, std::size_t cols)
        : _rows(rows)
        , _cols(cols)
        , _data(rows * cols)
    {
    }

    std::size_t rows() const noexcept { return _rows; };
    std::size_t cols() const noexcept { return _cols; };

    T&       operator()(std::size_t row, std::size_t col) { return _data[row * _cols + col]; };
    const T& operator()(std::size_t row, std::size_t col) const { return _data[row * _cols + col]; };

private:
    std::size_t    _rows {};
    std::size_t    _cols {};
    std::vector<T> _data {};
};

template <typename T>
T dot(const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("dimensions mismatch");
    }

    T result {};

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }

    return result;
}

template <typename T>
DenseMatrix<T> operator+(const DenseMatrix<T>& A, const DenseMatrix<T>& B)
{
    if (A.rows() != B.rows() || A.cols() != B.cols())
    {
        throw std::invalid_argument("dimensions mismatch");
    }
    DenseMatrix<T> result(A.rows(), A.cols());

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            result(i, j) = A(i, j) + B(i, j);
        }
    }

    return result;
}

template <typename T>
DenseMatrix<T> operator-(const DenseMatrix<T>& A, const DenseMatrix<T>& B)
{
    if (A.rows() != B.rows() || A.cols() != B.cols())
    {
        throw std::invalid_argument("dimensions mismatch");
    }
    DenseMatrix<T> result(A.rows(), A.cols());

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            result(i, j) = A(i, j) - B(i, j);
        }
    }

    return result;
}

template <typename T>
DenseMatrix<T> operator*(const T value, const DenseMatrix<T>& A)
{
    DenseMatrix<T> result(A.rows(), A.cols());

    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
            result(i, j) = A(i, j) * value;
        }
    }

    return result;
}

template <typename T>
DenseMatrix<T> operator*(const DenseMatrix<T>& A, const DenseMatrix<T>& B)
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument("dimensions mismatch");
    }

    DenseMatrix<T> result(A.rows(), B.cols());
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        for (std::size_t k = 0; k < A.cols(); ++k)
        {
            for (std::size_t j = 0; j < B.cols(); ++j)
            {
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return result;
}
