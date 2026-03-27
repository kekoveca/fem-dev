#include "assemble.hpp"
#include "mesh2d.hpp"
#include "poisson_kernel.hpp"
#include <cmath>
#include <gtest/gtest.h>

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

class MockPoissonKernel : public PoissonKernel
{
public:
    mutable ElementPoints last_coords {};
    mutable bool          called = false;

    MockPoissonKernel()
        : PoissonKernel([](double, double) { return 0.0; }, [](double, double) { return 1.0; })
    {
    }

    ElementData element_matrix(const ElementPoints& pts) override
    {
        called      = true;
        last_coords = pts;

        ElementData data {};

        data.Ke = {
            {{{2.0, -1.0, -1.0}}, {{-1.0, 2.0, -1.0}}, {{-1.0, -1.0, 2.0}}}
        };

        data.Fe = {
            {1.0, 2.0, 3.0}
        };

        return data;
    }
};

TEST(AssembleTest, SingleTriangleAssemblesLocalMatrixAndVector)
{
    Mesh2d mesh;

    mesh.coords = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    mesh.triangles = {
        {0, 1, 2}
    };

    MockPoissonKernel kernel;

    Assemble::GlobalData out = Assemble::assemble_poisson(mesh, kernel);

    ASSERT_TRUE(kernel.called);

    ASSERT_EQ(out.F.size(), 3);
    ASSERT_EQ(out.K.size(), 3);
    ASSERT_EQ(out.K[0].size(), 3);
    ASSERT_EQ(out.K[1].size(), 3);
    ASSERT_EQ(out.K[2].size(), 3);

    EXPECT_DOUBLE_EQ(out.F[0], 1.0);
    EXPECT_DOUBLE_EQ(out.F[1], 2.0);
    EXPECT_DOUBLE_EQ(out.F[2], 3.0);

    EXPECT_DOUBLE_EQ(out.K[0][0], 2.0);
    EXPECT_DOUBLE_EQ(out.K[0][1], -1.0);
    EXPECT_DOUBLE_EQ(out.K[0][2], -1.0);

    EXPECT_DOUBLE_EQ(out.K[1][0], -1.0);
    EXPECT_DOUBLE_EQ(out.K[1][1], 2.0);
    EXPECT_DOUBLE_EQ(out.K[1][2], -1.0);

    EXPECT_DOUBLE_EQ(out.K[2][0], -1.0);
    EXPECT_DOUBLE_EQ(out.K[2][1], -1.0);
    EXPECT_DOUBLE_EQ(out.K[2][2], 2.0);

    EXPECT_DOUBLE_EQ(kernel.last_coords[0][0], 0.0);
    EXPECT_DOUBLE_EQ(kernel.last_coords[0][1], 0.0);

    EXPECT_DOUBLE_EQ(kernel.last_coords[1][0], 1.0);
    EXPECT_DOUBLE_EQ(kernel.last_coords[1][1], 0.0);

    EXPECT_DOUBLE_EQ(kernel.last_coords[2][0], 0.0);
    EXPECT_DOUBLE_EQ(kernel.last_coords[2][1], 1.0);
}