import numpy as np
from dataclasses import dataclass


@dataclass
class DofMap:
    n_nodes: int
    ndofs_per_node: int = 1

    @property
    def n_dofs(self) -> int:
        return self.n_nodes * self.ndofs_per_node

    def element_dofs(self, conn_e: np.ndarray) -> np.ndarray:
        if self.ndofs_per_node == 1:
            return conn_e


def assemble_poisson(mesh, ref_el, kernel, quad_order: int = 1, cell_type: str = "triangle"):
    conn = mesh.cells[cell_type]
    n_elems = conn.shape[0]
    nen = conn.shape[1]

    n_nodes = mesh.coords.shape[0]

    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    for e in range(n_elems):
        nodes = conn[e]
        coords_e = mesh.coords[nodes]

        Ke, Fe = kernel.element_matrix(coords_e, ref_el, quad_order)

        for a in range(nen):
            A = nodes[a]
            F[A] += Fe[a]

            for b in range(nen):
                B = nodes[b]
                K[A, B] += Ke[a, b]

    return K, F
