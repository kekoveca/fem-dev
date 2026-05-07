#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"

template <typename T>
std::vector<T> CG(const DenseMatrix<T>& A, const std::vector<T>& b, double rtol = 1e-8, double atol = 1e-12,
                  std::size_t max_iter = 1000000)
{
    const std::size_t n = b.size();

    std::vector<T> x(n, T {0});
    std::vector<T> r = b;
    std::vector<T> p = r;

    T rr = dot(r, r);

    const T b_norm = norm_L2(b);
    const T tol    = static_cast<T>(atol) + static_cast<T>(rtol) * b_norm;
    const T tol2   = tol * tol;

    if (rr <= tol2)
    {
        return x;
    }

    for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
        const std::vector<T> Ap = A * p;

        const T pAp = dot(p, Ap);

        if (pAp <= T {0})
        {
            throw std::runtime_error("CG breakdown: matrix is not SPD");
        }

        const T alpha = rr / pAp;

        for (std::size_t j = 0; j < n; ++j)
        {
            x[j] += alpha * p[j];
        }

        for (std::size_t j = 0; j < n; ++j)
        {
            r[j] -= alpha * Ap[j];
        }

        const T rr_next = dot(r, r);

        if (rr_next <= tol2)
        {
            return x;
        }

        const T beta = rr_next / rr;

        for (std::size_t j = 0; j < n; ++j)
        {
            p[j] = r[j] + beta * p[j];
        }

        rr = rr_next;
    }

    throw std::runtime_error("CG did not converge");
}