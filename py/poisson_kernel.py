import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class PoissonKernel:
    f: Callable[[np.ndarray], float]
    k: Callable[[np.ndarray], float]

    def element_matrix(self, elements_points, ref_el, quad_order: int = 2):
        quadrature_points, w = ref_el.quadrature()
        N_ref = ref_el.ref_shape(quadrature_points)
        dN_ref = ref_el.grad_shape_ref(quadrature_points)

        nen = elements_points.shape[0]
        Ke = np.zeros((nen, nen), dtype=float)
        Fe = np.zeros(nen, dtype=float)

        for q_idx in range(quadrature_points.shape[0]):
            J = dN_ref[q_idx] @ elements_points
            detJ = np.linalg.det(J)

            invJ = np.linalg.inv(J)

            xq = N_ref[q_idx] @ elements_points
            fq = float(self.f(xq))
            kq = float(self.k(xq))

            Ke += kq * (dN_ref[q_idx].T @ invJ @ invJ.T @ dN_ref[q_idx]) * detJ * w[q_idx]
            Fe += (N_ref[q_idx] * fq) * detJ * w[q_idx]

        return Ke, Fe
