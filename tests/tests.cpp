#include "assemble.hpp"
#include "bc.hpp"
#include "cg.hpp"
#include "matrix.hpp"
#include "mesh2d.hpp"
#include "poisson_kernel.hpp"

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

constexpr double F_VAL     = 5.0;
constexpr double K_VAL     = 3.0;
constexpr double BND_1_VAL = 3.0;
constexpr double BND_2_VAL = 5.0;

template <typename T>
std::vector<T> read_testfile_data(const std::string& fname)
{
    std::ifstream test_file(fname);

    if (!test_file)
    {
        throw std::runtime_error("Cannot open file: " + fname);
    }

    std::vector<T> test_data;
    T              x;

    while (test_file >> x)
    {
        test_data.push_back(x);
    }

    return test_data;
}

template <typename T>
void expect_vector_near(const std::vector<T>& actual, const std::vector<T>& expected, T eps)
{
    ASSERT_EQ(actual.size(), expected.size());

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], eps) << "Mismatch at index " << i;
    }
}

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

    EXPECT_EQ(out.F.size(), 3);
    EXPECT_EQ(out.K.rows(), 3);
    EXPECT_EQ(out.K.cols(), 3);

    EXPECT_DOUBLE_EQ(out.F[0], 1.0);
    EXPECT_DOUBLE_EQ(out.F[1], 2.0);
    EXPECT_DOUBLE_EQ(out.F[2], 3.0);

    EXPECT_DOUBLE_EQ(out.K(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(out.K(0, 1), -1.0);
    EXPECT_DOUBLE_EQ(out.K(0, 2), -1.0);
    EXPECT_DOUBLE_EQ(out.K(1, 0), -1.0);
    EXPECT_DOUBLE_EQ(out.K(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(out.K(1, 2), -1.0);
    EXPECT_DOUBLE_EQ(out.K(2, 0), -1.0);
    EXPECT_DOUBLE_EQ(out.K(2, 1), -1.0);
    EXPECT_DOUBLE_EQ(out.K(2, 2), 2.0);

    EXPECT_DOUBLE_EQ(kernel.last_coords[0][0], 0.0);
    EXPECT_DOUBLE_EQ(kernel.last_coords[0][1], 0.0);

    EXPECT_DOUBLE_EQ(kernel.last_coords[1][0], 1.0);
    EXPECT_DOUBLE_EQ(kernel.last_coords[1][1], 0.0);

    EXPECT_DOUBLE_EQ(kernel.last_coords[2][0], 0.0);
    EXPECT_DOUBLE_EQ(kernel.last_coords[2][1], 1.0);
}

TEST(DirichletBCTest, DirichletNodesFromPhysicalSelectsCorrectNodes)
{
    Mesh2d mesh;

    mesh.coords = {
        {0.0, 0.0}, // 0
        {1.0, 0.0}, // 1
        {1.0, 1.0}, // 2
        {0.0, 1.0}  // 3
    };

    mesh.lines = {
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 0}
    };

    mesh.line_tags = {10, 20, 10, 30};

    const auto nodes = DirichletBC::dirichlet_nodes_from_physical(mesh, 10);

    EXPECT_EQ(nodes.size(), 4);
    EXPECT_EQ(nodes[0], 0u);
    EXPECT_EQ(nodes[1], 1u);
    EXPECT_EQ(nodes[2], 2u);
    EXPECT_EQ(nodes[3], 3u);
}

TEST(DirichletBCTest, ApplyDirichletEliminationBuildsReducedSystem)
{
    std::vector<double>       _d = {4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0};
    const DenseMatrix<double> K  = {3, 3, _d};

    const std::vector<double> F = {1.0, 2.0, 3.0};

    std::vector<DirichletBC::NodeAndValue> fixed = {
        {0, 10.0}
    };

    const auto reduced = DirichletBC::apply_dirichlet_elimination(K, F, fixed);

    EXPECT_EQ(reduced.free_nodes.size(), 2);
    EXPECT_EQ(reduced.free_nodes[0], 1u);
    EXPECT_EQ(reduced.free_nodes[1], 2u);

    EXPECT_EQ(reduced.fixed.size(), 1);
    EXPECT_EQ(reduced.fixed[0].node, 0u);
    EXPECT_DOUBLE_EQ(reduced.fixed[0].value, 10.0);

    EXPECT_EQ(reduced.K_reduced.rows(), 2);
    EXPECT_EQ(reduced.K_reduced.cols(), 2);

    EXPECT_DOUBLE_EQ(reduced.K_reduced(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(reduced.K_reduced(0, 1), -1.0);
    EXPECT_DOUBLE_EQ(reduced.K_reduced(1, 0), -1.0);
    EXPECT_DOUBLE_EQ(reduced.K_reduced(1, 1), 4.0);

    EXPECT_EQ(reduced.F_reduced.size(), 2);

    // F_free = [2, 3]
    // K_fd = [[-1], [0]]
    // F_reduced = [2, 3] - [[-1], [0]] * 10 = [12, 3]
    EXPECT_DOUBLE_EQ(reduced.F_reduced[0], 12.0);
    EXPECT_DOUBLE_EQ(reduced.F_reduced[1], 3.0);
}

TEST(DirichletBCTest, ApplyDirichletEliminationThrowsOnDuplicateDirichletNode)
{
    std::vector<double>       _d = {1.0, 0.0, 0.0, 1.0};
    const DenseMatrix<double> K  = {2, 2, _d};

    const std::vector<double> F = {0.0, 0.0};

    std::vector<DirichletBC::NodeAndValue> fixed = {
        {1, 5.0},
        {1, 7.0}
    };

    EXPECT_THROW(DirichletBC::apply_dirichlet_elimination(K, F, fixed), std::invalid_argument);
}

TEST(DirichletBCTest, RecoverFullSolutionReconstructsCorrectVector)
{
    const std::vector<DirichletBC::NodeAndValue> free_solution = {
        {1, 2.5},
        {3, 4.5}
    };

    const std::vector<DirichletBC::NodeAndValue> fixed = {
        {0, 0.0},
        {2, 1.0}
    };

    const auto u = DirichletBC::recover_full_solution(free_solution, fixed, 4);

    EXPECT_EQ(u.size(), 4);
    EXPECT_DOUBLE_EQ(u[0], 0.0);
    EXPECT_DOUBLE_EQ(u[1], 2.5);
    EXPECT_DOUBLE_EQ(u[2], 1.0);
    EXPECT_DOUBLE_EQ(u[3], 4.5);
}

TEST(DirichletBCTest, RecoverFullSolutionThrowsWhenNodeAppearsTwice)
{
    const std::vector<DirichletBC::NodeAndValue> free_solution = {
        {1, 2.5}
    };

    const std::vector<DirichletBC::NodeAndValue> fixed = {
        {1, 0.0}
    };

    EXPECT_THROW(DirichletBC::recover_full_solution(free_solution, fixed, 2), std::invalid_argument);
}

TEST(SolverTest, FixedNodesSizeEQTest)
{
    std::string mesh_fname = "/home/alex/fem-dev/meshes/test_mesh.msh";
    auto        mesh       = Mesh2d::read_from_gmsh(mesh_fname);

    PoissonKernel kernel([](double x, double y) { return F_VAL; }, [](double x, double y) { return K_VAL; });
    auto          bc1 = DirichletBC("boundary_one", [](double x, double y) { return BND_1_VAL; });
    auto          bc2 = DirichletBC("boundary_two", [](double x, double y) { return BND_2_VAL; });

    auto fixed_nodes_1 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc1.physical_name).tag);
    auto fixed_nodes_2 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc2.physical_name).tag);
    auto fixed         = fixed_nodes_1;

    fixed.reserve(fixed_nodes_1.size() + fixed_nodes_2.size());

    for (const auto& elem : fixed_nodes_2)
    {
        fixed.push_back(elem);
    }

    std::string fixed_nodes_test    = "/home/alex/fem-dev/tests/test_data/fixed_nodes_test.txt";
    auto        fixed_from_testfile = read_testfile_data<double>(fixed_nodes_test);

    std::vector<DirichletBC::NodeAndValue> nodes_and_values = {};
    nodes_and_values.resize(fixed.size());

    for (std::size_t i = 0; i < fixed.size(); ++i)
    {
        auto node                = fixed[i];
        nodes_and_values[i].node = node;
        auto [x, y]              = mesh.coords[i];
        if (std::find(fixed_nodes_1.begin(), fixed_nodes_1.end(), node) != fixed_nodes_1.end())
        {
            nodes_and_values[i].value = bc1.value(x, y);
        }
        else
        {
            nodes_and_values[i].value = bc2.value(x, y);
        }
    }

    std::sort(nodes_and_values.begin(), nodes_and_values.end(),
              [](const DirichletBC::NodeAndValue& lhs, const DirichletBC::NodeAndValue& rhs)
              { return lhs.node < rhs.node; });

    auto it = std::unique(nodes_and_values.begin(), nodes_and_values.end(),
                          [](const DirichletBC::NodeAndValue& lhs, const DirichletBC::NodeAndValue& rhs)
                          { return lhs.node == rhs.node; });

    nodes_and_values.erase(it, nodes_and_values.end());

    EXPECT_EQ(nodes_and_values.size(), fixed_from_testfile.size());
}

TEST(SolverTest, FixedValuesEQTest)
{
    std::string mesh_fname = "/home/alex/fem-dev/meshes/test_mesh.msh";
    auto        mesh       = Mesh2d::read_from_gmsh(mesh_fname);

    PoissonKernel kernel([](double x, double y) { return F_VAL; }, [](double x, double y) { return K_VAL; });
    auto          bc1 = DirichletBC("boundary_one", [](double x, double y) { return BND_1_VAL; });
    auto          bc2 = DirichletBC("boundary_two", [](double x, double y) { return BND_2_VAL; });

    auto fixed_nodes_1 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc1.physical_name).tag);
    auto fixed_nodes_2 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc2.physical_name).tag);
    auto fixed         = fixed_nodes_1;

    fixed.reserve(fixed_nodes_1.size() + fixed_nodes_2.size());

    for (const auto& elem : fixed_nodes_2)
    {
        fixed.push_back(elem);
    }

    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());

    std::vector<DirichletBC::NodeAndValue> nodes_and_values(fixed.size());

    for (std::size_t i = 0; i < fixed.size(); ++i)
    {
        auto node                = fixed[i];
        nodes_and_values[i].node = node;
        auto [x, y]              = mesh.coords[node];
        if (std::find(fixed_nodes_1.begin(), fixed_nodes_1.end(), node) != fixed_nodes_1.end())
        {
            nodes_and_values[i].value = bc1.value(x, y);
        }
        else
        {
            nodes_and_values[i].value = bc2.value(x, y);
        }
    }

    std::string              fixed_values_test = "/home/alex/fem-dev/tests/test_data/fixed_values_test.txt";
    std::ifstream            fixed_values_file(fixed_values_test);
    std::vector<std::size_t> fixed_from_testfile;
    double                   _x;
    while (fixed_values_file >> _x)
    {
        fixed_from_testfile.push_back(static_cast<std::size_t>(_x));
    }

    int error_counter = 0;
    for (std::size_t i = 0; i < nodes_and_values.size(); ++i)
    {
        if (std::abs(nodes_and_values[i].value - fixed_from_testfile[i]) > 1e-12)
        {
            error_counter++;
        };
    }

    EXPECT_EQ(error_counter, 0);
}

TEST(SolverTest, FullMatrixTest)
{
    std::string   mesh_fname = "/home/alex/fem-dev/meshes/test_mesh.msh";
    auto          mesh       = Mesh2d::read_from_gmsh(mesh_fname);
    PoissonKernel kernel([](double x, double y) { return F_VAL; }, [](double x, double y) { return K_VAL; });

    std::string         full_matrix_test     = "/home/alex/fem-dev/tests/test_data/full_matrix_flattened_test.txt";
    std::vector<double> matrix_from_testfile = read_testfile_data<double>(full_matrix_test);

    auto assembled = Assemble::assemble_poisson(mesh, kernel);

    std::size_t m = assembled.K.rows();
    std::size_t n = assembled.K.cols();
    EXPECT_EQ(m, n);
    EXPECT_EQ(matrix_from_testfile.size(), m * n);

    int counter = 0;
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            if (std::abs(matrix_from_testfile[i * m + j] - assembled.K(i, j)) > 1.0e-12)
            {
                ++counter;
            }
        }
    }
    EXPECT_EQ(counter, 0);
}

TEST(SolverTest, ReducedMatrixTest)
{
    std::string   mesh_fname = "/home/alex/fem-dev/meshes/test_mesh.msh";
    auto          mesh       = Mesh2d::read_from_gmsh(mesh_fname);
    PoissonKernel kernel([](double x, double y) { return F_VAL; }, [](double x, double y) { return K_VAL; });
    auto          bc1  = DirichletBC("boundary_one", [](double x, double y) { return BND_1_VAL; });
    auto          bc2  = DirichletBC("boundary_two", [](double x, double y) { return BND_2_VAL; });
    auto fixed_nodes_1 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc1.physical_name).tag);

    auto fixed_nodes_2 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc2.physical_name).tag);
    auto fixed         = fixed_nodes_1;

    fixed.reserve(fixed_nodes_1.size() + fixed_nodes_2.size());

    for (const auto& elem : fixed_nodes_2)
    {
        fixed.push_back(elem);
    }

    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());

    std::vector<DirichletBC::NodeAndValue> nodes_and_values(fixed.size());

    for (std::size_t i = 0; i < fixed.size(); ++i)
    {
        auto node                = fixed[i];
        nodes_and_values[i].node = node;
        auto [x, y]              = mesh.coords[node];
        if (std::find(fixed_nodes_1.begin(), fixed_nodes_1.end(), node) != fixed_nodes_1.end())
        {
            nodes_and_values[i].value = bc1.value(x, y);
        }
        else
        {
            nodes_and_values[i].value = bc2.value(x, y);
        }
    }

    std::string         reduced_matrix_test  = "/home/alex/fem-dev/tests/test_data/reduced_matrix_flattened_test.txt";
    std::vector<double> matrix_from_testfile = read_testfile_data<double>(reduced_matrix_test);

    auto assembled      = Assemble::assemble_poisson(mesh, kernel);
    auto reduced_system = DirichletBC::apply_dirichlet_elimination(assembled.K, assembled.F, nodes_and_values);

    std::size_t m = reduced_system.K_reduced.rows();
    std::size_t n = reduced_system.K_reduced.cols();
    EXPECT_EQ(m, n);
    EXPECT_EQ(matrix_from_testfile.size(), m * n);

    int counter = 0;
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            if (std::abs(matrix_from_testfile[i * m + j] - reduced_system.K_reduced(i, j)) > 1.0e-12)
            {
                ++counter;
            }
        }
    }
    EXPECT_EQ(counter, 0);
}

TEST(SolverTest, FullRHSTest)
{
    std::string   mesh_fname = "/home/alex/fem-dev/meshes/test_mesh.msh";
    auto          mesh       = Mesh2d::read_from_gmsh(mesh_fname);
    PoissonKernel kernel([](double x, double y) { return F_VAL; }, [](double x, double y) { return K_VAL; });

    auto                assembled         = Assemble::assemble_poisson(mesh, kernel);
    std::string         full_rhs_test     = "/home/alex/fem-dev/tests/test_data/full_rhs_flattened_test.txt";
    std::vector<double> rhs_from_testfile = read_testfile_data<double>(full_rhs_test);

    std::size_t n = assembled.F.size();
    EXPECT_EQ(rhs_from_testfile.size(), n);

    int counter = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (std::abs(rhs_from_testfile[i] - assembled.F[i]) > 1.0e-12)
        {
            ++counter;
        }
    }
    EXPECT_EQ(counter, 0);
}

TEST(SolverTest, ReducedRHSTest)
{
    std::string   mesh_fname = "/home/alex/fem-dev/meshes/test_mesh.msh";
    auto          mesh       = Mesh2d::read_from_gmsh(mesh_fname);
    PoissonKernel kernel([](double x, double y) { return F_VAL; }, [](double x, double y) { return K_VAL; });
    auto          bc1  = DirichletBC("boundary_one", [](double x, double y) { return BND_1_VAL; });
    auto          bc2  = DirichletBC("boundary_two", [](double x, double y) { return BND_2_VAL; });
    auto fixed_nodes_1 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc1.physical_name).tag);
    auto fixed_nodes_2 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc2.physical_name).tag);
    auto fixed         = fixed_nodes_1;

    fixed.reserve(fixed_nodes_1.size() + fixed_nodes_2.size());

    for (const auto& elem : fixed_nodes_2)
    {
        fixed.push_back(elem);
    }

    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());

    std::vector<DirichletBC::NodeAndValue> nodes_and_values(fixed.size());

    for (std::size_t i = 0; i < fixed.size(); ++i)
    {
        auto node                = fixed[i];
        nodes_and_values[i].node = node;
        auto [x, y]              = mesh.coords[node];
        if (std::find(fixed_nodes_1.begin(), fixed_nodes_1.end(), node) != fixed_nodes_1.end())
        {
            nodes_and_values[i].value = bc1.value(x, y);
        }
        else
        {
            nodes_and_values[i].value = bc2.value(x, y);
        }
    }

    std::string         reduced_rhs_test  = "/home/alex/fem-dev/tests/test_data/reduced_rhs_flattened_test.txt";
    std::vector<double> rhs_from_testfile = read_testfile_data<double>(reduced_rhs_test);

    auto assembled      = Assemble::assemble_poisson(mesh, kernel);
    auto reduced_system = DirichletBC::apply_dirichlet_elimination(assembled.K, assembled.F, nodes_and_values);

    std::size_t m = reduced_system.F_reduced.size();
    EXPECT_EQ(rhs_from_testfile.size(), m);

    int counter = 0;
    for (std::size_t i = 0; i < m; ++i)
    {

        if (std::abs(rhs_from_testfile[i] - reduced_system.F_reduced[i]) > 1.0e-12)
        {
            ++counter;
        }
    }
    EXPECT_EQ(counter, 0);
}

TEST(SolverTest, FullSolverTest)
{
    std::string   mesh_fname = "/home/alex/fem-dev/meshes/test_mesh.msh";
    auto          mesh       = Mesh2d::read_from_gmsh(mesh_fname);
    PoissonKernel kernel([](double x, double y) { return F_VAL; }, [](double x, double y) { return K_VAL; });
    auto          bc1  = DirichletBC("boundary_one", [](double x, double y) { return BND_1_VAL; });
    auto          bc2  = DirichletBC("boundary_two", [](double x, double y) { return BND_2_VAL; });
    auto fixed_nodes_1 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc1.physical_name).tag);
    auto fixed_nodes_2 = DirichletBC::dirichlet_nodes_from_physical(mesh, mesh.fields_data.at(bc2.physical_name).tag);
    auto fixed         = fixed_nodes_1;

    fixed.reserve(fixed_nodes_1.size() + fixed_nodes_2.size());

    for (const auto& elem : fixed_nodes_2)
    {
        fixed.push_back(elem);
    }

    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());

    std::vector<DirichletBC::NodeAndValue> nodes_and_values(fixed.size());

    for (std::size_t i = 0; i < fixed.size(); ++i)
    {
        auto node                = fixed[i];
        nodes_and_values[i].node = node;
        auto [x, y]              = mesh.coords[node];
        if (std::find(fixed_nodes_1.begin(), fixed_nodes_1.end(), node) != fixed_nodes_1.end())
        {
            nodes_and_values[i].value = bc1.value(x, y);
        }
        else
        {
            nodes_and_values[i].value = bc2.value(x, y);
        }
    }

    auto assembled      = Assemble::assemble_poisson(mesh, kernel);
    auto reduced_system = DirichletBC::apply_dirichlet_elimination(assembled.K, assembled.F, nodes_and_values);
    auto solution       = CG(reduced_system.K_reduced, reduced_system.F_reduced);

    std::vector<DirichletBC::NodeAndValue> freed;
    freed.resize(solution.size());

    for (std::size_t i = 0; i < solution.size(); ++i)
    {
        freed[i] = {reduced_system.free_nodes[i], solution[i]};
    }

    auto u = DirichletBC::recover_full_solution(freed, reduced_system.fixed, mesh.coords.size());

    std::string         solution_test_fname = "/home/alex/fem-dev/tests/test_data/solution_test.txt";
    std::vector<double> test_solution_data  = read_testfile_data<double>(solution_test_fname);

    auto err = norm(test_solution_data - u);
    EXPECT_LT(err, 1e-14);
}

TEST(MatrixClassTests, DotProductTest)
{
    std::vector<double> a {1.0, 2.0, 3.0};
    std::vector<double> b {4.0, 5.0, 6.0};

    double result = dot(a, b);
    EXPECT_EQ(result, 32.0);
}

TEST(MatrixClassTests, SumTest)
{
    DenseMatrix<double> A(2, 2);
    DenseMatrix<double> B(2, 2);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            A(i, j) = i + j;
            B(i, j) = 2 * i + j;
        }
    }

    auto C = A + B;

    EXPECT_DOUBLE_EQ(C(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 5.0);
}

TEST(MatrixClassTests, DiffTest)
{
    DenseMatrix<double> A(2, 2);
    DenseMatrix<double> B(2, 2);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            A(i, j) = i + j;
            B(i, j) = 2 * i + j;
        }
    }

    auto C = A - B;

    EXPECT_DOUBLE_EQ(C(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(C(1, 0), -1.0);
    EXPECT_DOUBLE_EQ(C(1, 1), -1.0);
}

TEST(MatrixClassTests, MatmulTest)
{
    DenseMatrix<double> A(2, 2);
    DenseMatrix<double> B(2, 2);

    A(0, 0) = 3.1;
    A(0, 1) = 12.2;
    A(1, 0) = 4.2;
    A(1, 1) = 5.7;

    B(0, 0) = 35.3;
    B(0, 1) = 9.7;
    B(1, 0) = 6.3;
    B(1, 1) = 12.2;

    auto C = A * B;

    EXPECT_DOUBLE_EQ(C(0, 0), 186.29);
    EXPECT_DOUBLE_EQ(C(0, 1), 178.91);
    EXPECT_DOUBLE_EQ(C(1, 0), 184.17);
    EXPECT_DOUBLE_EQ(C(1, 1), 110.28);

    auto D = B * A;

    EXPECT_DOUBLE_EQ(D(0, 0), 150.17);
    EXPECT_DOUBLE_EQ(D(0, 1), 485.95);
    EXPECT_DOUBLE_EQ(D(1, 0), 70.77);
    EXPECT_DOUBLE_EQ(D(1, 1), 146.4);

    DenseMatrix<double> M(2, 6);
    DenseMatrix<double> N(4, 7);

    EXPECT_THROW(M * N, std::invalid_argument);
}

TEST(MatrixClassTests, MatVecProductTest)
{
    DenseMatrix<double> A(2, 3);

    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;

    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    std::vector<double> x {10.0, 20.0, 30.0};

    std::vector<double> y = A * x;

    EXPECT_DOUBLE_EQ(y[0], 140.0);
    EXPECT_DOUBLE_EQ(y[1], 320.0);
}

TEST(MatrixClassTests, VecMatProductTest)
{
    std::vector<double> x {10.0, 20.0};

    DenseMatrix<double> A(2, 3);

    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;

    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    std::vector<double> y = x * A;

    EXPECT_DOUBLE_EQ(y[0], 90.0);
    EXPECT_DOUBLE_EQ(y[1], 120.0);
    EXPECT_DOUBLE_EQ(y[2], 150.0);
}

TEST(MatrixClassTests, VectorNormTest)
{
    std::vector<double> v    = {3.0, 4.0};
    double              magn = norm(v);
    EXPECT_DOUBLE_EQ(magn, 5.0);
}

TEST(MatrixClassTests, VectorSubtractionTest)
{
    const std::vector<double> a = {6.0, 10.0, 8.0};
    const std::vector<double> b = {-6.0, -10.0, -8.0};

    const auto r = a - b;

    ASSERT_EQ(r.size(), 3);
    EXPECT_DOUBLE_EQ(r[0], 12.0);
    EXPECT_DOUBLE_EQ(r[1], 20.0);
    EXPECT_DOUBLE_EQ(r[2], 16.0);
}

TEST(CGTest, Solves2x2KnownSolution)
{
    DenseMatrix<double> A(2, 2);

    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 3.0;

    std::vector<double> b = {1.0, 2.0};

    auto x = CG(A, b, 1e-12, 1e-14, 100);

    const std::vector<double> expected = {1.0 / 11.0, 7.0 / 11.0};

    expect_vector_near(x, expected, 1e-10);
}

TEST(CGTest, ReturnsZeroForZeroRightHandSide)
{
    DenseMatrix<double> A(3, 3);

    A(0, 0) = 2.0;
    A(0, 1) = 0.0;
    A(0, 2) = 0.0;

    A(1, 0) = 0.0;
    A(1, 1) = 3.0;
    A(1, 2) = 0.0;

    A(2, 0) = 0.0;
    A(2, 1) = 0.0;
    A(2, 2) = 4.0;

    std::vector<double> b = {0.0, 0.0, 0.0};

    auto x = CG(A, b);

    const std::vector<double> expected = {0.0, 0.0, 0.0};

    expect_vector_near(x, expected, 1e-14);
}

TEST(CGTest, SolvesDiagonalMatrix)
{
    DenseMatrix<double> A(4, 4);

    A(0, 0) = 2.0;
    A(0, 1) = 0.0;
    A(0, 2) = 0.0;
    A(0, 3) = 0.0;

    A(1, 0) = 0.0;
    A(1, 1) = 4.0;
    A(1, 2) = 0.0;
    A(1, 3) = 0.0;

    A(2, 0) = 0.0;
    A(2, 1) = 0.0;
    A(2, 2) = 5.0;
    A(2, 3) = 0.0;

    A(3, 0) = 0.0;
    A(3, 1) = 0.0;
    A(3, 2) = 0.0;
    A(3, 3) = 10.0;

    std::vector<double> b = {2.0, 8.0, 15.0, 40.0};

    auto x = CG(A, b, 1e-12, 1e-14, 100);

    const std::vector<double> expected = {1.0, 2.0, 3.0, 4.0};

    expect_vector_near(x, expected, 1e-10);
}

TEST(CGTest, ProducesSmallResidual)
{
    DenseMatrix<double> A(3, 3);

    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;

    A(1, 0) = 1.0;
    A(1, 1) = 3.0;
    A(1, 2) = 1.0;

    A(2, 0) = 0.0;
    A(2, 1) = 1.0;
    A(2, 2) = 2.0;

    const std::vector<double> expected = {1.0, 2.0, 3.0};

    auto b = A * expected;

    EXPECT_NEAR(b[0], 6.0, 1e-14);
    EXPECT_NEAR(b[1], 10.0, 1e-14);
    EXPECT_NEAR(b[2], 8.0, 1e-14);

    auto x = CG(A, b, 1e-12, 1e-14, 100);

    auto Ax = A * x;
    auto r  = b - Ax;

    EXPECT_LT(norm(r), 1e-10);

    expect_vector_near(x, expected, 1e-10);
}