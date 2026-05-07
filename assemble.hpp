#pragma once
#include "matrix.hpp"
#include "mesh2d.hpp"
#include "poisson_kernel.hpp"
#include <vector>

class Assemble
{
public:
    struct GlobalData
    {
        DenseMatrix<double> K;
        std::vector<double> F;
    };

    static GlobalData assemble_poisson(const Mesh2d& mesh, PoissonKernel& kernel);
};