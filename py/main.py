import numpy as np

from mesh import read_gmsh_meshio
from ref_T3 import Tri3
from poisson_kernel import PoissonKernel
from assemble import assemble_poisson
from bc import DirichletBC, dirichlet_nodes_from_physical, apply_dirichlet_elimination, recover_full_solution

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import gmsh


def plot_solution_2d(mesh, u):
    coords = mesh.coords
    triangles = mesh.cells["triangle"]

    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    plt.figure(figsize=(7, 3))
    plt.tripcolor(triang, u)

    plt.colorbar(label="u")
    plt.triplot(triang, color="k", linewidth=0.5, alpha=0.4)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("FEM solution")
    plt.show()


def create_mesh():
    gmsh.initialize()

    gmsh.model.add("rectangle")

    lc = 0.5
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = []
    cl.append(gmsh.model.geo.addCurveLoop([l1, l2, l3, l4]))

    srf = gmsh.model.geo.addPlaneSurface(cl)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l1], name="boundary_one")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], name="boundary_two")
    gmsh.model.addPhysicalGroup(2, [srf], name="surface")

    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write("rectangle.msh")
    gmsh.finalize()


def main():
    create_mesh()
    mesh = read_gmsh_meshio("rectangle.msh", dim=2)

    ref = Tri3()

    # Пример: k=1, f=1
    kernel = PoissonKernel(
        k=lambda xy: 0.0,
        f=lambda xy: 1.0,
    )

    K, F = assemble_poisson(mesh, ref, kernel, quad_order=2, cell_type="triangle")

    # Дирихле: boundary_one -> u=0, boundary_two -> u=1 (пример)
    bc1 = DirichletBC("boundary_one", value=lambda xy: 0.0)
    bc2 = DirichletBC("boundary_two", value=lambda xy: 0.0)

    fixed_nodes_1 = dirichlet_nodes_from_physical(mesh, mesh.physical_tag(bc1.physical_name))
    fixed_nodes_2 = dirichlet_nodes_from_physical(mesh, mesh.physical_tag(bc2.physical_name))

    fixed = np.unique(np.concatenate([fixed_nodes_1, fixed_nodes_2]))
    # значения на фиксированных узлах (тут константы; если функция от xy — считаем по coords)
    u_fixed = np.zeros_like(fixed, dtype=float)
    for i, node in enumerate(fixed):
        xy = mesh.coords[node]
        # если узел попал в обе группы (углы) — можно задать приоритетом или проверять по границе;
        # здесь просто: если в boundary_two -> 1, иначе 0
        if node in set(fixed_nodes_2):
            u_fixed[i] = bc2.value(xy)
        else:
            u_fixed[i] = bc1.value(xy)

    Kff, Ff, free, fixed, u_fixed = apply_dirichlet_elimination(K, F, fixed, u_fixed)

    u_free = np.linalg.solve(Kff, Ff)
    u = recover_full_solution(u_free, F.shape[0], free, fixed, u_fixed)

    print("Solved. u min/max:", u.min(), u.max())
    plot_solution_2d(mesh, u)


if __name__ == "__main__":
    main()
