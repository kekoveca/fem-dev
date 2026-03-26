#pragma once

#include "ref_T3.hpp"
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

class PoissonKernel
{
public:
    using Point         = std::array<double, 2>;
    using ElementPoints = std::array<Point, 3>;
    using LocalMatrix   = std::array<std::array<double, 3>, 3>;
    using LocalVector   = std::array<double, 3>;

    std::function<double(double, double)> f;
    std::function<double(double, double)> k;

    PoissonKernel(std::function<double(double, double)> f_fun, std::function<double(double, double)> k_fun)
        : f(std::move(f_fun))
        , k(std::move(k_fun)) {};

    struct ElementData
    {
        LocalMatrix Ke;
        LocalVector Fe;
    };

    ElementData element_matrix(const ElementPoints& element_points);
};