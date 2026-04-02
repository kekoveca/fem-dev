import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class DirichletBC:
    physical_name: str  # например "boundary_one"
    value: Callable[[np.ndarray], float]  # value(xy) -> scalar


def dirichlet_nodes_from_physical(mesh, physical_tag: int) -> np.ndarray:
    if "line" not in mesh.cells:
        raise ValueError("Mesh has no 'line' cells. Can't build boundary nodes.")
    edges = mesh.cells["line"]
    edge_tags = mesh.cell_tags.get("line", None)
    if edge_tags is None:
        raise ValueError("No physical tags for 'line'. Check definition of Physical Groups in gmsh")

    sel_edges = edges[edge_tags == physical_tag]
    nodes = np.unique(sel_edges.ravel())
    return nodes


def apply_dirichlet_elimination(K, F, fixed_nodes, fixed_values):
    """
    Убирает DOF с условиями Дирихле
    """
    fixed_nodes = np.asarray(fixed_nodes, dtype=int)
    fixed_values = np.asarray(fixed_values, dtype=float)

    n = F.shape[0]
    all_nodes = np.arange(n, dtype=int)
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)

    K_reduced = K[np.ix_(free_nodes, free_nodes)]
    K_fd = K[np.ix_(free_nodes, fixed_nodes)]
    F_reduced = F[free_nodes] - K_fd @ fixed_values

    return K_reduced, F_reduced, free_nodes, fixed_nodes, fixed_values


def recover_full_solution(u_free, n_nodes, free_nodes, fixed_nodes, fixed_values):
    """
    Восстанавливает полный вектор решения после исключения DOF.
    """
    u = np.zeros(n_nodes, dtype=float)
    u[free_nodes] = u_free
    u[fixed_nodes] = fixed_values
    return u
