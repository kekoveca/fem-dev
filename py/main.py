import numpy as np

from mesh import read_gmsh_meshio
from ref_T3 import Tri3
from poisson_kernel import PoissonKernel
from assemble import assemble_poisson
from bc import DirichletBC, dirichlet_nodes_from_physical, apply_dirichlet_elimination, recover_full_solution
from gmres import GMRES


import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import gmsh

mesh_path = "meshes/test_mesh.msh"


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

    lc = 0.07
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
    gmsh.write(mesh_path)
    gmsh.finalize()


def main():
    test_data_folder = "/home/alex/fem-dev/tests/test_data/"

    create_mesh()

    mesh = read_gmsh_meshio(mesh_path, dim=2)

    ref = Tri3()

    # Пример: k=1, f=1
    kernel = PoissonKernel(
        f=lambda xy: 5.0,
        k=lambda xy: 3.0,
    )

    K, F = assemble_poisson(mesh, ref, kernel, quad_order=2, cell_type="triangle")

    # Дирихле: boundary_one -> u=0, boundary_two -> u=1 (пример)
    bc1 = DirichletBC("boundary_one", value=lambda xy: 3.0)
    bc2 = DirichletBC("boundary_two", value=lambda xy: 5.0)

    fixed_nodes_1 = dirichlet_nodes_from_physical(mesh, mesh.physical_tag(bc1.physical_name))
    fixed_nodes_2 = dirichlet_nodes_from_physical(mesh, mesh.physical_tag(bc2.physical_name))

    fixed = np.unique(np.concatenate([fixed_nodes_1, fixed_nodes_2]))
    # значения на фиксированных узлах (тут константы; если функция от xy — считаем по coords)
    u_fixed = np.zeros_like(fixed, dtype=float)
    for i, node in enumerate(fixed):
        xy = mesh.coords[node]
        # если узел попал в обе группы (углы) — можно задать приоритетом или проверять по границе;
        # здесь просто: если в boundary_two -> 1, иначе 0
        if node in set(fixed_nodes_1):
            u_fixed[i] = bc1.value(xy)
        else:
            u_fixed[i] = bc2.value(xy)

    Kff, Ff, free, fixed, u_fixed = apply_dirichlet_elimination(K, F, fixed, u_fixed)

    u_free, _ = GMRES(Kff, Ff, rtol=1e-8)
    u = recover_full_solution(u_free, F.shape[0], free, fixed, u_fixed)

    np.savetxt(test_data_folder + "reduced_matrix_flattened_test.txt", Kff.flatten())
    np.savetxt(test_data_folder + "reduced_rhs_flattened_test.txt", Ff.flatten())
    np.savetxt(test_data_folder + "full_rhs_flattened_test.txt", F.flatten())
    np.savetxt(test_data_folder + "full_matrix_flattened_test.txt", K.flatten())
    np.savetxt(test_data_folder + "fixed_nodes_test.txt", fixed.flatten())
    np.savetxt(test_data_folder + "fixed_values_test.txt", u_fixed.flatten())
    np.savetxt(test_data_folder + "fixed_values_test.txt", u_fixed.flatten())
    np.savetxt(test_data_folder + "solution_test.txt", u)

    print("Solved. u min/max:", u.min(), u.max())
    # plot_solution_2d(mesh, u)


if __name__ == "__main__":
    main()
