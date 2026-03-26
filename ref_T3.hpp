#pragma once
#include <array>
#include <vector>

struct QuadraturePoint2D
{
    double xi, eta, weight;
};

struct Tri3
{
    using GradMatrix = std::array<std::array<double, 3>, 2>;

    static constexpr int dim = 2;
    static constexpr int nen = 3;

    static std::vector<QuadraturePoint2D> quadrature();
    static std::array<double, 3>          ref_shape(double xi, double eta);
    static GradMatrix                     grad_ref_shape();
};