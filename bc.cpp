#include "bc.hpp"

std::vector<std::size_t> DirichletBC::dirichlet_nodes_from_physical(const Mesh2d& mesh, int physical_tag)
{
    const auto& edges     = mesh.lines;
    const auto& edge_tags = mesh.line_tags;

    if (edge_tags.empty())
    {
        throw std::runtime_error("no tags for 'line'. Check definition of physical tags");
    }

    if (edges.size() != edge_tags.size())
    {
        throw std::runtime_error("edges and edge_tags size mismatch");
    }

    std::vector<std::array<std::size_t, 2>> selected_edges;
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
        if (edge_tags[i] == physical_tag)
        {
            selected_edges.push_back(edges[i]);
        }
    }

    std::vector<std::size_t> dirichlet_nodes;

    for (const auto& line : selected_edges)
    {
        dirichlet_nodes.push_back(line[0]);
        dirichlet_nodes.push_back(line[1]);
    }

    std::sort(dirichlet_nodes.begin(), dirichlet_nodes.end());
    dirichlet_nodes.erase(std::unique(dirichlet_nodes.begin(), dirichlet_nodes.end()), dirichlet_nodes.end());

    return dirichlet_nodes;
};

DirichletBC::ReducedSystem DirichletBC::apply_dirichlet_elimination(const std::vector<std::vector<double>>& K,
                                                                    const std::vector<double>&              F,
                                                                    std::vector<NodeAndValue> fixed_nodes_and_values)
{
    const std::size_t n = F.size();

    if (K.size() != n)
    {
        throw std::invalid_argument("K and F size mismatch");
    }

    for (const auto& row : K)
    {
        if (row.size() != n)
        {
            throw std::invalid_argument("K must be square");
        }
    }

    for (const auto& bc : fixed_nodes_and_values)
    {
        if (bc.node >= n)
        {
            throw std::out_of_range("Dirichlet node out of range");
        }
    }

    std::sort(fixed_nodes_and_values.begin(), fixed_nodes_and_values.end(),
              [](const NodeAndValue& lhs, const NodeAndValue& rhs) { return lhs.node < rhs.node; });

    for (std::size_t i = 1; i < fixed_nodes_and_values.size(); ++i)
    {
        if (fixed_nodes_and_values[i].node == fixed_nodes_and_values[i - 1].node)
        {
            throw std::invalid_argument("duplicate Dirichlet node");
        }
    }

    ReducedSystem out;

    out.fixed.reserve(fixed_nodes_and_values.size());

    std::vector<bool> is_fixed(n, false);

    for (const auto& [node, value] : fixed_nodes_and_values)
    {
        out.fixed.push_back({node, value});
        is_fixed[node] = true;
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!is_fixed[i])
        {
            out.free_nodes.push_back(i);
        }
    }

    const std::size_t n_free  = out.free_nodes.size();
    const std::size_t n_fixed = out.fixed.size();

    out.K_reduced.assign(n_free, std::vector<double>(n_free, 0.0));
    out.F_reduced.assign(n_free, 0.0);

    for (std::size_t i = 0; i < n_free; ++i)
    {
        const std::size_t I = out.free_nodes[i];

        out.F_reduced[i] = F[I];

        for (std::size_t j = 0; j < n_fixed; ++j)
        {
            const std::size_t J = out.fixed[j].node;
            out.F_reduced[i] -= K[I][J] * out.fixed[j].value;
        }

        for (std::size_t j = 0; j < n_free; ++j)
        {
            const std::size_t J = out.free_nodes[j];
            out.K_reduced[i][j] = K[I][J];
        }
    }

    return out;
};

std::vector<double> DirichletBC::recover_full_solution(const std::vector<NodeAndValue>& free_nodes_and_values,
                                                       const std::vector<NodeAndValue>& fixed_nodes_and_values,
                                                       const std::size_t                n_nodes)
{
    if (n_nodes != (free_nodes_and_values.size() + fixed_nodes_and_values.size()))
    {
        throw std::invalid_argument("Invalid argumets. size(free) + size(fixed) != n_nodes");
    }

    std::unordered_set<std::size_t> nodes;

    for (const auto& [node, value] : free_nodes_and_values)
    {
        nodes.insert(node);
    }

    for (const auto& [node, value] : fixed_nodes_and_values)
    {
        if (nodes.count(node))
        {
            throw std::invalid_argument("Invalid argumets. Check duplicate nodes");
        }
    }

    std::vector<double> out(n_nodes);

    for (const auto& [node, value] : free_nodes_and_values)
    {
        out[node] = value;
    }

    for (const auto& [node, value] : fixed_nodes_and_values)
    {
        out[node] = value;
    }

    return out;
};