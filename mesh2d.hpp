#pragma once

#include <array>
#include <gmsh.h>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct FieldInfo
{
    int tag;
    int dim;
};

class Mesh2d
{
public:
    std::vector<std::array<double, 2>>         coords;
    std::vector<std::array<int, 3>>            triangles;
    std::vector<std::array<int, 2>>            lines;
    std::vector<int>                           triangle_tags;
    std::vector<int>                           line_tags;
    std::unordered_map<std::string, FieldInfo> fields_data;

    void          print_info() const;
    static Mesh2d read_from_gmsh(const std::string& path);

private:
    std::vector<std::size_t> _triangle_elem_tags;
    std::vector<std::size_t> _line_elem_tags;
};
