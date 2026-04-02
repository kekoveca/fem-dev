#pragma once

#include "mesh2d.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

class DirichletBC
{
public:
    struct NodeAndValue
    {
        std::size_t node;
        double      value;
    };

    struct ReducedSystem
    {
        std::vector<std::vector<double>> K_reduced;
        std::vector<double>              F_reduced;
        std::vector<std::size_t>         free_nodes;
        std::vector<NodeAndValue>        fixed;
    };

    std::string                           physical_name;
    std::function<double(double, double)> value;

    DirichletBC(const std::string& physical_name_, std::function<double(double, double)> value_)
        : physical_name(physical_name_)
        , value(std::move(value_))
    {
    }

    static std::vector<std::size_t> dirichlet_nodes_from_physical(const Mesh2d& mesh, int physical_tag);

    static ReducedSystem apply_dirichlet_elimination(const std::vector<std::vector<double>>& K,
                                                     const std::vector<double>&              F,
                                                     std::vector<NodeAndValue>               nodes_and_values);

    static std::vector<double> recover_full_solution(const std::vector<NodeAndValue>& freed,
                                                     const std::vector<NodeAndValue>& fixed, const std::size_t n_nodes);
};