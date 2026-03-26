import gmsh
import sys


def create_mesh(lc):
    gmsh.initialize()

    gmsh.model.add("rectangle")

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        lc = float(sys.argv[1])
        create_mesh(lc)
    else:
        print(f"Usage: python3 create_mesh.py <mesh_size>")
