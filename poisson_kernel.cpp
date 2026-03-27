#include "poisson_kernel.hpp"

PoissonKernel::ElementData PoissonKernel::element_matrix(const ElementPoints& element_points)
{
    ElementData data              = {};
    const auto  quadrature_points = Tri3::quadrature();
    const auto  B                 = Tri3::grad_ref_shape();

    for (auto& row : data.Ke)
    {
        row.fill(0.0);
    }

    data.Fe.fill(0);

    double x1 = element_points[0][0];
    double x2 = element_points[1][0];
    double x3 = element_points[2][0];

    double y1 = element_points[0][1];
    double y2 = element_points[1][1];
    double y3 = element_points[2][1];

    double detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);

    if (std::abs(detJ) < 1e-14)
    {
        throw std::runtime_error("det(J) ~ 0. Degenerate triangle");
    }
    if (detJ <= 0.0)
    {
        throw std::runtime_error("det(J) <= 0. Check triangle orientation");
    }

    std::array<std::array<double, 2>, 2> invJT;
    invJT[0][0] = (y3 - y1) / detJ;
    invJT[0][1] = (x1 - x3) / detJ;
    invJT[1][0] = (y1 - y2) / detJ;
    invJT[1][1] = (x2 - x1) / detJ;

    Tri3::GradMatrix B_physical {};

    for (std::size_t i = 0; i < Tri3::dim; ++i)
    {
        for (std::size_t j = 0; j < Tri3::nen; ++j)
        {
            for (std::size_t l = 0; l < Tri3::dim; ++l)
            {

                B_physical[i][j] += invJT[i][l] * B[l][j];
            }
        }
    }

    for (const auto& qp : quadrature_points)
    {
        const double xi  = qp.xi;
        const double eta = qp.eta;
        const double w   = qp.weight;

        auto shapes = Tri3::ref_shape(xi, eta);

        double x = shapes[0] * x1 + shapes[1] * x2 + shapes[2] * x3;
        double y = shapes[0] * y1 + shapes[1] * y2 + shapes[2] * y3;

        double fq = f(x, y);
        double kq = k(x, y);

        for (std::size_t i = 0; i < Tri3::nen; ++i)
        {
            data.Fe[i] += shapes[i] * fq * detJ * w;
            for (std::size_t j = 0; j < Tri3::nen; ++j)
            {
                data.Ke[i][j] +=
                    kq * (B_physical[0][i] * B_physical[0][j] + B_physical[1][i] * B_physical[1][j]) * detJ * w;
            }
        }
    }

    return data;
};