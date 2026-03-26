import numpy as np
import meshio
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class Mesh:
    dim: int
    coords: np.ndarray  # (n_nodes, dim)
    cells: Dict[str, np.ndarray]  # e.g. {"triangle": (ne,3), "line": (nb,2)}
    cell_tags: Dict[str, np.ndarray]  # e.g. {"triangle": (ne,), "line": (nb,)}
    field_data: Dict[str, Tuple[int, int]]  # name -> (tag, dim)

    def physical_tag(self, name: str) -> int:
        if name not in self.field_data:
            raise KeyError(f"Physical group '{name}' not found. Available: {list(self.field_data.keys())}")
        return int(self.field_data[name][0])


def read_gmsh_meshio(path: str, dim: int = 2) -> Mesh:
    m = meshio.read(path)

    coords = m.points[:, :dim].astype(float)

    cells = {}
    if "triangle" in m.cells_dict:
        cells["triangle"] = m.cells_dict["triangle"].astype(int)
    if "line" in m.cells_dict:
        cells["line"] = m.cells_dict["line"].astype(int)

    # Physical tags
    cell_tags = {}
    if "gmsh:physical" in m.cell_data_dict:
        phys = m.cell_data_dict["gmsh:physical"]
        for k, arr in phys.items():
            cell_tags[k] = np.array(arr, dtype=int)
    else:
        # if no physicals were defined in gmsh
        for k in cells.keys():
            cell_tags[k] = np.zeros(len(cells[k]), dtype=int)

    return Mesh(
        dim=dim,
        coords=coords,
        cells=cells,
        cell_tags=cell_tags,
        field_data=m.field_data,
    )
