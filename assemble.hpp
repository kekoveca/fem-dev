#pragma once
#include "mesh2d.hpp"
#include "poisson_kernel.hpp"
#include <vector>

class Assemble
{
public:
    struct GlobalData
    {
        std::vector<std::vector<double>> K;
        std::vector<double>              F;
    };

    static GlobalData assemble_poisson(const Mesh2d& mesh, PoissonKernel& kernel);
};