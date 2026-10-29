#!/usr/bin/env python3
"""Interactive 3D viewer for a molecular electron-density surface colored by electrostatic
potential (both .cube files produced by `rhf_sim --format density,esp`)."""

import argparse

import numpy as np
import vtk
from vtkmodules.util import numpy_support

# CPK-ish colors and covalent radii (bohr) for the elements MiniSCF currently supports.
ELEMENTS = {
    1: ("H", (1.00, 1.00, 1.00), 0.60),
    2: ("He", (0.85, 1.00, 1.00), 0.56),
    3: ("Li", (0.80, 0.50, 1.00), 2.42),
    4: ("Be", (0.76, 1.00, 0.00), 1.81),
    5: ("B", (1.00, 0.71, 0.71), 1.55),
    6: ("C", (0.30, 0.30, 0.30), 1.44),
    7: ("N", (0.19, 0.31, 0.97), 1.36),
    8: ("O", (1.00, 0.05, 0.05), 1.32),
    9: ("F", (0.56, 0.88, 0.31), 1.26),
    10: ("Ne", (0.70, 0.89, 0.96), 1.21),
    11: ("Na", (0.67, 0.36, 0.95), 2.99),
    12: ("Mg", (0.54, 1.00, 0.00), 2.63),
    13: ("Al", (0.75, 0.65, 0.65), 2.43),
    14: ("Si", (0.50, 0.60, 0.60), 2.33),
    15: ("P", (1.00, 0.50, 0.00), 2.22),
    16: ("S", (1.00, 1.00, 0.19), 2.15),
    17: ("Cl", (0.12, 0.94, 0.12), 2.08),
    18: ("Ar", (0.50, 0.82, 0.89), 2.02),
}
DEFAULT_ELEMENT = ("?", (0.80, 0.80, 0.80), 1.50)


def read_cube(path):
    with open(path) as f:
        lines = f.readlines()

    natoms, ox, oy, oz = lines[2].split()
    natoms, origin = int(natoms), tuple(map(float, (ox, oy, oz)))

    nx, dx, _, _ = lines[3].split()
    ny, _, dy, _ = lines[4].split()
    nz, _, _, dz = lines[5].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    dx, dy, dz = float(dx), float(dy), float(dz)

    atoms = []
    for line in lines[6 : 6 + natoms]:
        parts = line.split()
        atoms.append((int(float(parts[0])), *map(float, parts[2:5])))

    values = np.fromstring(" ".join(lines[6 + natoms :]), sep=" ", dtype=np.float64)
    if values.size != nx * ny * nz:
        raise ValueError(
            f"expected {nx * ny * nz} volume values but found {values.size} in '{path}'"
        )

    return {
        "origin": origin,
        "spacing": (dx, dy, dz),
        "dims": (nx, ny, nz),
        "atoms": atoms,
        "data": values.reshape((nx, ny, nz)),
    }


def build_image_data(cube, name):
    nx, ny, nz = cube["dims"]

    # cube files store x slowest / z fastest; vtkImageData wants x fastest / z slowest.
    flat = cube["data"].transpose(2, 1, 0).ravel(order="C")
    scalars = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_DOUBLE)
    scalars.SetName(name)

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*cube["spacing"])
    image.SetOrigin(*cube["origin"])
    image.GetPointData().SetScalars(scalars)
    return image


def build_diverging_lut(vmin, vmax):
    """Blue (most negative, electron-rich) -> white (neutral) -> red (most positive,
    electron-poor), matching the conventional electrostatic-potential color scale."""
    if vmax <= vmin:
        vmax = vmin + 1.0
    vmid = 0.0 if vmin < 0.0 < vmax else (vmin + vmax) / 2.0

    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    ctf.AddRGBPoint(vmin, 0.0, 0.0, 1.0)
    ctf.AddRGBPoint(vmid, 1.0, 1.0, 1.0)
    ctf.AddRGBPoint(vmax, 1.0, 0.0, 0.0)
    return ctf


def warn_if_open(polydata, isovalue):
    """If the density hasn't decayed below `isovalue` by the edge of the computed grid,
    the isosurface gets cut off there instead of closing up on its own, leaving open
    boundary edges (and often a self-intersecting-looking mess) instead of a clean
    envelope."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(polydata)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    n_open = edges.GetOutput().GetNumberOfCells()
    if n_open > 0:
        print(
            f"warning: the density isosurface at isovalue={isovalue} has {n_open} open "
            "boundary edges, likely clipped by the grid boundary; try a larger --isovalue."
        )


def esp_surface_actor(density_image, esp_image, isovalue, lut, vmin, vmax, opacity):
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(density_image)
    mc.ComputeNormalsOn()
    mc.SetValue(0, isovalue)
    mc.Update()
    warn_if_open(mc.GetOutput(), isovalue)

    # Interpolate the ESP volume onto the density isosurface's points.
    probe = vtk.vtkProbeFilter()
    probe.SetInputConnection(mc.GetOutputPort())
    probe.SetSourceData(esp_image)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(probe.GetOutputPort())
    mapper.SetScalarModeToUsePointData()
    mapper.SelectColorArray("esp")
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(vmin, vmax)
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    # Default Specular is 0 and the renderer's default light is a headlight (tied to the
    # camera), so a plain diffuse-only surface barely varies in brightness and looks like
    # a flat single-color blob. A specular highlight under Phong interpolation restores a
    # visible shading gradient regardless of view angle.
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.15)
    prop.SetDiffuse(0.85)
    prop.SetSpecular(0.4)
    prop.SetSpecularPower(30)
    prop.SetOpacity(opacity)
    return actor, probe


def scalar_bar_actor(lut):
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle("electrostatic potential")
    bar.SetOrientationToHorizontal()
    bar.SetNumberOfLabels(3)
    bar.SetLabelFormat("%.3g")
    bar.SetWidth(0.4)
    bar.SetHeight(0.08)
    bar.SetPosition(0.55, 0.05)
    bar.GetTitleTextProperty().SetColor(0.0, 0.0, 0.0)
    bar.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)
    bar.GetTitleTextProperty().ShadowOff()
    bar.GetLabelTextProperty().ShadowOff()
    bar.UnconstrainedFontSizeOn()
    return bar


def atom_actors(atoms, scale):
    actors = []
    for atomic_number, x, y, z in atoms:
        _, color, radius = ELEMENTS.get(atomic_number, DEFAULT_ELEMENT)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius * scale)
        sphere.SetThetaResolution(32)
        sphere.SetPhiResolution(32)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(x, y, z)
        actor.GetProperty().SetColor(*color)
        actors.append(actor)
    return actors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density-cube",
        default="mo_density.cube",
        help="path to the electron-density .cube file, defines the surface geometry "
        "(default: mo_density.cube)",
    )
    parser.add_argument(
        "--esp-cube",
        default="mo_esp.cube",
        help="path to the electrostatic-potential .cube file, colors the surface "
        "(default: mo_esp.cube)",
    )
    parser.add_argument(
        "--isovalue",
        type=float,
        default=0.02,
        help="electron-density isovalue defining the molecular surface, in e/bohr^3 "
        "(default: 0.02)",
    )
    parser.add_argument(
        "--esp-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="fixed min/max for the ESP color scale, in Hartree/e "
        "(default: min/max of the ESP actually found on the surface)",
    )
    parser.add_argument(
        "--show-atoms",
        action="store_true",
        help="also draw atom spheres (hidden under the surface unless --surface-opacity < 1)",
    )
    parser.add_argument(
        "--atom-scale",
        type=float,
        default=0.25,
        help="atom sphere radius as a fraction of covalent radius (default: 0.25)",
    )
    parser.add_argument(
        "--surface-opacity",
        type=float,
        default=1.0,
        help="opacity of the density surface, from 0 (invisible) to 1 (opaque); "
        "lower it together with --show-atoms to see the atoms inside (default: 1.0)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.surface_opacity <= 1.0:
        parser.error("--surface-opacity must be between 0 and 1")

    density_cube = read_cube(args.density_cube)
    esp_cube = read_cube(args.esp_cube)

    density_image = build_image_data(density_cube, "density")
    esp_image = build_image_data(esp_cube, "esp")

    lut_placeholder = build_diverging_lut(-1.0, 1.0)
    actor, probe = esp_surface_actor(
        density_image,
        esp_image,
        args.isovalue,
        lut_placeholder,
        -1.0,
        1.0,
        args.surface_opacity,
    )
    probe.Update()

    if args.esp_range is not None:
        vmin, vmax = args.esp_range
    else:
        esp_on_surface = numpy_support.vtk_to_numpy(
            probe.GetOutput().GetPointData().GetArray("esp")
        )
        if esp_on_surface.size == 0:
            raise ValueError(
                f"the density isosurface at isovalue={args.isovalue} is empty; "
                "try a smaller --isovalue"
            )
        vmin, vmax = float(esp_on_surface.min()), float(esp_on_surface.max())
    print(f"ESP on surface: [{vmin:.6g}, {vmax:.6g}]")

    lut = build_diverging_lut(vmin, vmax)
    actor.GetMapper().SetLookupTable(lut)
    actor.GetMapper().SetScalarRange(vmin, vmax)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(actor)
    renderer.AddActor(scalar_bar_actor(lut))
    if args.show_atoms:
        for atom_actor in atom_actors(density_cube["atoms"], args.atom_scale):
            renderer.AddActor(atom_actor)

    window = vtk.vtkRenderWindow()
    window.SetSize(1200, 900)
    window.SetWindowName(f"MiniSCF - {args.esp_cube} on {args.density_cube}")
    window.AddRenderer(renderer)

    if args.surface_opacity < 1.0:
        # A translucent surface overlaps itself (front and back faces, plus the atoms
        # behind it), and plain alpha blending only looks right if those fragments are
        # drawn back-to-front. Depth peeling sorts them per pixel instead.
        window.SetAlphaBitPlanes(1)
        window.SetMultiSamples(0)
        renderer.SetUseDepthPeeling(1)
        renderer.SetMaximumNumberOfPeels(100)
        renderer.SetOcclusionRatio(0.0)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderer.ResetCamera()
    window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
