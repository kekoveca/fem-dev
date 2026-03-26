#include "poisson_kernel.hpp"
#include <cmath>
#include <gtest/gtest.h>

static bool nearly_equal(double a, double b, double tol = 1e-12) { return std::abs(a - b) < tol; }

TEST(PoissonKernelTest, ReferenceTriangle)
{
    PoissonKernel kernel([](double x, double y) { return 1.0; }, [](double x, double y) { return 1.0; });

    PoissonKernel::ElementPoints element = {
        {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}
    };

    auto data = kernel.element_matrix(element);

    EXPECT_NEAR(data.Ke[0][0], 1.0, 1e-12);
    EXPECT_NEAR(data.Ke[0][1], -0.5, 1e-12);
    EXPECT_NEAR(data.Ke[0][2], -0.5, 1e-12);

    EXPECT_NEAR(data.Ke[1][0], -0.5, 1e-12);
    EXPECT_NEAR(data.Ke[1][1], 0.5, 1e-12);
    EXPECT_NEAR(data.Ke[1][2], 0.0, 1e-12);

    EXPECT_NEAR(data.Ke[2][0], -0.5, 1e-12);
    EXPECT_NEAR(data.Ke[2][1], 0.0, 1e-12);
    EXPECT_NEAR(data.Ke[2][2], 0.5, 1e-12);

    EXPECT_NEAR(data.Fe[0], 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(data.Fe[1], 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(data.Fe[2], 1.0 / 6.0, 1e-12);
}

TEST(PoissonKernelTest, DegenerateTriangle)
{
    PoissonKernel kernel([](double x, double y) { return 1.0; }, [](double x, double y) { return 1.0; });

    PoissonKernel::ElementPoints element = {
        {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}
    };

    EXPECT_THROW(kernel.element_matrix(element), std::runtime_error);
}

TEST(PoissonKernelTest, MatrixSymmetry)
{
    PoissonKernel kernel([](double x, double y) { return 1.0; }, [](double x, double y) { return 1.0; });

    PoissonKernel::ElementPoints element = {
        {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}
    };

    auto data = kernel.element_matrix(element);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            EXPECT_NEAR(data.Ke[i][j], data.Ke[j][i], 1e-12);
        }
    }
}