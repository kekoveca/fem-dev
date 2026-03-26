#include "ref_T3.hpp"

std::vector<QuadraturePoint2D> Tri3::quadrature()
{
    return {
        {1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0}
    };
}

std::array<double, 3> Tri3::ref_shape(double xi, double eta) { return {1.0 - xi - eta, xi, eta}; }

Tri3::GradMatrix Tri3::grad_ref_shape()
{
    return {
        {{-1.0, 1.0, 0.0}, {-1.0, 0.0, 1.0}}
    };
}