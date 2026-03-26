import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Tri3:
    dim: int = 2
    nen: int = 3

    def quadrature(self):
        """
        Возвращает (quadrature_points, w):
        quadrature_points: (num_quadrature_points, 2) точки (xi, eta) на эталонном треугольнике
        w:  (num_quadrature_points,)    веса,
        """
        # Эталонный треугольник: xi>=0, eta>=0, xi+eta<=1, площадь = 1/2
        # 1-точечная (точна для линейных)
        quadrature_points = np.array([[1 / 3, 1 / 3]], dtype=float)
        w = np.array([1 / 2], dtype=float)
        return quadrature_points, w

    def ref_shape(self, quadrature_points: np.ndarray) -> np.ndarray:
        xi = quadrature_points[:, 0]
        eta = quadrature_points[:, 1]
        N1 = 1.0 - xi - eta
        N2 = xi
        N3 = eta
        return np.column_stack([N1, N2, N3])

    def grad_shape_ref(self, quadrature_points: np.ndarray) -> np.ndarray:
        """
        dN/dxi, dN/deta для каждого q:
        вернём массив (nq, dim, nen)
        """
        num_quadrature_points = quadrature_points.shape[0]
        dN = np.zeros((num_quadrature_points, 2, 3), dtype=float)

        # N1=1-xi-eta; N2=xi; N3=eta
        dN[:, 0, 0] = -1.0  # dN1/dxi
        dN[:, 1, 0] = -1.0  # dN1/deta
        dN[:, 0, 1] = 1.0  # dN2/dxi
        dN[:, 1, 1] = 0.0  # dN2/deta
        dN[:, 0, 2] = 0.0  # dN3/dxi
        dN[:, 1, 2] = 1.0  # dN3/deta

        return dN
