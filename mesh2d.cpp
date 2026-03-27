#include "mesh2d.hpp"

Mesh2d Mesh2d::read_from_gmsh(const std::string& path)
{
    Mesh2d mesh;

    gmsh::initialize();
    gmsh::open(path);

    std::vector<std::size_t> node_tags;
    std::vector<double>      node_coords;
    std::vector<double>      node_params;

    gmsh::model::mesh::getNodes(node_tags, node_coords, node_params);

    std::unordered_map<std::size_t, std::size_t> node_tag_to_index;
    std::unordered_map<std::size_t, std::size_t> line_elem_tag_to_index;
    std::unordered_map<std::size_t, std::size_t> tri_elem_tag_to_index;

    mesh.coords.resize(node_tags.size());

    for (std::size_t i = 0; i < node_tags.size(); ++i)
    {
        std::size_t tag        = node_tags[i];
        node_tag_to_index[tag] = i;
        double x               = node_coords[3 * i];
        double y               = node_coords[3 * i + 1];
        mesh.coords[i]         = {x, y};
    }

    std::vector<std::size_t> triangle_elem_node_tags;
    gmsh::model::mesh::getElementsByType(2, mesh._triangle_elem_tags, triangle_elem_node_tags, -1);

    mesh.triangles.reserve(mesh._triangle_elem_tags.size());

    for (std::size_t i = 0; i < triangle_elem_node_tags.size(); i += 3)
    {
        std::size_t n0 = node_tag_to_index[triangle_elem_node_tags[i]];
        std::size_t n1 = node_tag_to_index[triangle_elem_node_tags[i + 1]];
        std::size_t n2 = node_tag_to_index[triangle_elem_node_tags[i + 2]];

        mesh.triangles.push_back({n0, n1, n2});
    }

    mesh.triangle_tags.resize(mesh._triangle_elem_tags.size());

    for (std::size_t i = 0; i < mesh._triangle_elem_tags.size(); ++i)
    {
        std::size_t tag = mesh._triangle_elem_tags[i];

        tri_elem_tag_to_index[tag] = i;
    }

    std::vector<std::size_t> line_elem_nodes_tags;
    gmsh::model::mesh::getElementsByType(1, mesh._line_elem_tags, line_elem_nodes_tags, -1);

    mesh.lines.reserve(mesh._line_elem_tags.size());

    for (std::size_t i = 0; i < line_elem_nodes_tags.size(); i += 2)
    {
        std::size_t n0 = node_tag_to_index[line_elem_nodes_tags[i]];
        std::size_t n1 = node_tag_to_index[line_elem_nodes_tags[i + 1]];

        mesh.lines.push_back({n0, n1});
    }

    mesh.line_tags.resize(mesh._line_elem_tags.size());

    for (std::size_t i = 0; i < mesh._line_elem_tags.size(); ++i)
    {
        std::size_t tag = mesh._line_elem_tags[i];

        line_elem_tag_to_index[tag] = i;
    }

    std::vector<std::pair<int, int>> physical_groups;
    gmsh::model::getPhysicalGroups(physical_groups);

    for (const auto& [dim, physical_tag] : physical_groups)
    {
        std::string name;
        gmsh::model::getPhysicalName(dim, physical_tag, name);
        mesh.fields_data[name] = FieldInfo {physical_tag, dim};

        std::vector<int> entity_tags;
        gmsh::model::getEntitiesForPhysicalGroup(dim, physical_tag, entity_tags);

        if (dim == 2)
        {
            for (int entity_tag : entity_tags)
            {
                std::vector<int>                      elem_types;
                std::vector<std::vector<std::size_t>> elem_tags, elem_node_tags;

                gmsh::model::mesh::getElements(elem_types, elem_tags, elem_node_tags, dim, entity_tag);

                for (size_t k = 0; k < elem_types.size(); ++k)
                {
                    const auto& tags = elem_tags[k];
                    for (const auto tag : tags)
                    {
                        auto it = tri_elem_tag_to_index.find(tag);
                        if (it != tri_elem_tag_to_index.end())
                        {
                            std::size_t idx         = it->second;
                            mesh.triangle_tags[idx] = physical_tag;
                        }
                    }
                }
            }
        }
        else if (dim == 1)
        {
            for (int entity_tag : entity_tags)
            {
                std::vector<int>                      elem_types;
                std::vector<std::vector<std::size_t>> elem_tags, elem_node_tags;

                gmsh::model::mesh::getElements(elem_types, elem_tags, elem_node_tags, dim, entity_tag);

                for (size_t k = 0; k < elem_types.size(); ++k)
                {
                    const auto& tags = elem_tags[k];
                    for (const auto tag : tags)
                    {
                        auto it = line_elem_tag_to_index.find(tag);
                        if (it != line_elem_tag_to_index.end())
                        {
                            std::size_t idx     = it->second;
                            mesh.line_tags[idx] = physical_tag;
                        }
                    }
                }
            }
        }
    }

    gmsh::finalize();
    return mesh;
};

void Mesh2d::print_info() const
{
    std::cout << "Num nodes: " << coords.size() << '\n';
    std::cout << "Nodes coords:\n";
    for (const auto& e : coords)
    {
        std::cout << '(' << e[0] << ' ' << e[1] << ')' << '\n';
    }

    std::cout << "============\n";
    std::cout << "Num triangles: " << triangles.size() << '\n';
    std::cout << "Triangles:\n";
    for (const auto& t : triangles)
    {
        std::cout << '(' << t[0] << ' ' << t[1] << ' ' << t[2] << ')' << '\n';
    }

    std::cout << "============\n";
    std::cout << "Num lines: " << lines.size() << '\n';
    std::cout << "Lines:\n";
    for (const auto& l : lines)
    {
        std::cout << '(' << l[0] << ' ' << l[1] << ')' << '\n';
    }

    std::cout << "============\n";
    std::cout << "Physical groups:\n";
    for (const auto& [name, field] : fields_data)
    {
        std::cout << name << " (" << field.dim << ' ' << field.tag << ")\n";
    }

    std::cout << "============\n";
    std::cout << "Triangle tags list:\n";
    std::set<int> triangle_tags_set(triangle_tags.begin(), triangle_tags.end());
    for (const auto& t : triangle_tags_set)
    {
        std::cout << t << '\n';
    }

    std::cout << "============\n";
    std::cout << "Line tags list:\n";
    std::set<int> line_tags_set(line_tags.begin(), line_tags.end());
    for (const auto& l : line_tags_set)
    {
        std::cout << l << '\n';
    }

    return;
}