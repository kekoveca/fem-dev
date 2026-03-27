#include "assemble.hpp"

Assemble::GlobalData Assemble::assemble_poisson(const Mesh2d& mesh, PoissonKernel& kernel)
{
    const auto&       conn    = mesh.triangles;
    const std::size_t n_elems = conn.size();
    const std::size_t nen     = conn[0].size();
    const std::size_t n_nodes = mesh.coords.size();
    GlobalData        out     = {};

    if (n_elems == 0)
    {
        return out;
    }

    out.K.resize(n_nodes);

    for (std::size_t i = 0; i < n_nodes; ++i)
    {
        out.K[i].resize(n_nodes);
    }
    out.F.resize(n_nodes);

    for (std::size_t i = 0; i < n_elems; ++i)
    {
        const auto&                  nodes = conn[i];
        PoissonKernel::ElementPoints coords_e;

        for (std::size_t j = 0; j < nen; ++j)
        {
            coords_e[j] = mesh.coords[nodes[j]];
        }

        PoissonKernel::ElementData data = kernel.element_matrix(coords_e);

        for (std::size_t a = 0; a < nen; ++a)
        {
            std::size_t A = nodes[a];
            out.F[A] += data.Fe[a];

            for (std::size_t b = 0; b < nen; ++b)
            {
                std::size_t B = nodes[b];
                out.K[A][B] += data.Ke[a][b];
            }
        }
    }

    return out;
}