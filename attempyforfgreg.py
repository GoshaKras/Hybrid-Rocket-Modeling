"""Estimate 3-D hybrid-rocket port regression from an OBJ fuel-grain mesh.

The OBJ is treated as the *solid fuel grain*.  A voxel model is made from the
mesh, the empty region containing the grain centre is selected as the initial
port, and an Euclidean distance field grows that port into the fuel.  This
works for non-circular, star, and helical ports without assuming a 2-D shape.

Example
-------
    python attempyforfgreg.py goshastar.obj --resolution 180 \
        --max-regression 8 --steps 17 --output regression_results.csv

The program automatically treats a small CAD-sized mesh (longest dimension
under 2 units) as metres, which is how the OBJ files included in this project
are exported. Use ``--unit-scale-mm`` to override this if needed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import time

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from skimage.draw import polygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.backend_bases import ResizeEvent

# Matplotlib 3.11 + Qt can dispatch ResizeEvent objects through widget mouse
# callbacks. Widgets expect ``event.inaxes``; giving resize events that benign
# attribute prevents repeated console tracebacks while resizing the viewer.
if not hasattr(ResizeEvent, "inaxes"):
    ResizeEvent.inaxes = None


# =============================================================================
# USER SETTINGS — change these values, then press Run in your IDE.
# =============================================================================
OBJ_FILE = "circlehellixed.obj"       # OBJ fuel-grain file in this folder
# The named plane is the high-resolution cross-section; its normal is the
# lower-resolution length direction. Example: "X" means a detailed Y-Z slice.
HIGH_RESOLUTION_PLANE = "X"            # "X", "Y", or "Z"
IN_PLANE_RESOLUTION = 500               # 500 x 500 pixels in the selected cross-section
LENGTH_RESOLUTION = 200                 # Number of slices along the selected plane normal
# Kept for old command-line compatibility; the two settings above control IDE runs.
RESOLUTION = IN_PLANE_RESOLUTION
REGRESSION_RATE_MM_PER_S = 1.0         # Normal fuel regression rate
SIMULATION_TIME_S = 50.0               # Total simulated burn time
SLIDER_STEPS = 21                       # Number of regression/time positions in the slider
DISPLAY_LENGTH_UNIT = "mm"             # "mm", "cm", "m", or "in" for plots and viewer labels
OPEN_INTERACTIVE_VIEWER = True          # Open the X/Y/Z interactive viewer after calculating
SMOOTH_VIEWER_RENDERING = False         # Smooth the displayed slice edges; does not alter calculations
SHOW_PLANE_PREVIEW = True               # Show the selected 500x500 plane before the long calculation starts
SAVE_SELECTED_PLANE_ANALYSIS = True      # Generate selected-plane CSV plus curve-fit plots
SELECTED_PLANE_ANALYSIS_PLOT = "selected_plane_regression.png"
SELECTED_PLANE_ANALYSIS_CSV = "selected_plane_regression.csv"
SELECTED_PLANE_CURVE_FITS_CSV = "selected_plane_curve_fits.csv"
SAVE_SELECTED_PLANE_ANALYSIS_PLOT = True  # Write the curve-fit PNG to disk
SHOW_SELECTED_PLANE_ANALYSIS_AFTER_VIEWER = True  # Open curve-fit window after closing the slider viewer
SHOW_CURVE_FIT_EQUATIONS_ON_PLOTS = False  # Show piecewise equations in each graph (CSV always contains them)
SELECTED_PLANE_GRAPH_POINTS = 50     # Number of regression samples used for the selected-plane curves
STOP_SELECTED_PLANE_GRAPHS_WHEN_FUEL_IS_GONE = True  # Stop graph/fit at the first empty selected-plane slice
OUTPUT_FOLDER = "regression_outputs"  # Folder for generated CSV and PNG files
EXPORT_REGRESSED_SNAPSHOT_MM = 50  # Set (for example) 25.0 to export remaining fuel as an OBJ
REGRESSED_SNAPSHOT_FILENAME = "regressed_fuel_snapshot.obj"
# Helical-sweep design inputs. OBJ files do not reliably preserve parametric
# CAD features, so enter pitch and centerline radius from the source drawing.
IS_HELICAL_SWEEP = False              # Set True only after entering verified helix CAD dimensions
HELIX_PITCH_MM = None                  # P: axial advance per full revolution
HELIX_CENTERLINE_RADIUS_MM = None      # r: radius from helix axis to swept-profile centerline
SHOW_HELIX_DIMENSIONS_IN_VIEWER = False # Enable the slider-viewer pitch/curvature toggle
# =============================================================================

DISPLAY_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


def display_unit_scale(unit: str) -> tuple[str, float]:
    """Validate a display unit and return its millimetres-per-unit scale."""
    normalized = unit.lower().strip()
    if normalized not in DISPLAY_UNIT_TO_MM:
        raise ValueError(f"DISPLAY_LENGTH_UNIT must be one of: {', '.join(DISPLAY_UNIT_TO_MM)}")
    return normalized, DISPLAY_UNIT_TO_MM[normalized]


@contextmanager
def activity_indicator(message: str, triangle_count: int):
    """Show an elapsed-time status while an operation has no real progress API."""
    finished = threading.Event()
    started = time.monotonic()

    def report() -> None:
        frames = ("|", "/", "-", "\\")
        frame = 0
        while not finished.wait(0.5):
            elapsed = int(time.monotonic() - started)
            print(f"\r{message} {frames[frame % len(frames)]}  {elapsed}s elapsed", end="", flush=True)
            frame += 1

    worker = threading.Thread(target=report, daemon=True)
    print(f"{message} (this OBJ has {triangle_count:,} triangles)", flush=True)
    worker.start()
    try:
        yield
    finally:
        finished.set()
        worker.join()
        elapsed = time.monotonic() - started
        print(f"\r{message} complete in {elapsed:.1f}s.{' ' * 25}", flush=True)


def load_mesh(path: Path, scale_to_mm: float) -> trimesh.Trimesh:
    """Load an OBJ as one mesh and return coordinates in millimetres."""
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} does not contain a usable triangle mesh")
    mesh = loaded.copy()
    mesh.apply_scale(scale_to_mm)
    return mesh


def infer_unit_scale_to_mm(path: Path, override: float | None) -> float:
    """Choose a safe default for typical CAD OBJ exports.

    OBJ files have no unit metadata.  The supplied models are measured in
    metres (about 0.8 units long); treating them as millimetres caused the
    entire grain to burn during the first 1 mm regression step.
    """
    if override is not None:
        return override
    raw_mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = trimesh.util.concatenate(tuple(raw_mesh.geometry.values()))
    longest_dimension = float(np.max(raw_mesh.extents))
    return 1000.0 if longest_dimension < 2.0 else 1.0


def choose_axis(mesh: trimesh.Trimesh, axis: int | None) -> int:
    """Use the longest mesh dimension as the motor axis unless specified."""
    return int(np.argmax(mesh.extents)) if axis is None else axis


def preview_high_resolution_plane(mesh: trimesh.Trimesh, axis: int, resolution: int) -> int:
    """Let the user inspect and choose the centre plane for fine rasterization."""
    axis_names = ("X", "Y", "Z")
    selected_axis = [axis]
    figure, plot = plt.subplots(figsize=(8.8, 8))
    figure.subplots_adjust(right=0.78, bottom=0.15)
    radio_buttons = RadioButtons(figure.add_axes((0.81, 0.56, 0.14, 0.20)), axis_names, active=axis)
    continue_button = Button(figure.add_axes((0.80, 0.41, 0.15, 0.06)), "Continue")

    def draw_plane() -> None:
        selected = selected_axis[0]
        remaining = [number for number in range(3) if number != selected]
        point = np.zeros(3)
        point[selected] = float(np.mean(mesh.bounds[:, selected]))
        normal = np.zeros(3)
        normal[selected] = 1.0
        section = mesh.section(plane_origin=point, plane_normal=normal)
        plot.clear()
        if section is None:
            plot.text(0.5, 0.5, "No centre section found", ha="center", va="center", transform=plot.transAxes)
        else:
            for loop in section.discrete:
                coordinates = loop[:, remaining]
                plot.plot(coordinates[:, 0], coordinates[:, 1], color="#f97316", linewidth=1.2)
        plot.set_aspect("equal", adjustable="box")
        plot.set_xlabel(f"{axis_names[remaining[0]]} (mm)")
        plot.set_ylabel(f"{axis_names[remaining[1]]} (mm)")
        plot.set_title(f"High-resolution plane: {axis_names[selected]}\n"
                       f"Rasterization: {resolution} by {resolution}")
        plot.grid(True, alpha=0.25)
        figure.canvas.draw_idle()

    def select_plane(label: str) -> None:
        selected_axis[0] = axis_names.index(label)
        draw_plane()

    radio_buttons.on_clicked(select_plane)
    continue_button.on_clicked(lambda _event: plt.close(figure))
    draw_plane()
    print("Choose the high-resolution plane in the preview, then click Continue.")
    plt.show()
    return selected_axis[0]


def make_solid_voxels(mesh: trimesh.Trimesh, high_plane_axis: int,
                      in_plane_resolution: int, length_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize mesh sections into an anisotropic 3-D fuel grid.

    This creates a high-resolution 500x500 (by default) cross-section and a
    smaller number of sections along the chosen axis, avoiding a 500-cubed grid.
    """
    other_axes = [axis for axis in range(3) if axis != high_plane_axis]
    bounds = mesh.bounds
    extents = mesh.extents
    pitches = np.empty(3, dtype=float)
    pitches[high_plane_axis] = extents[high_plane_axis] / length_resolution
    pitches[other_axes] = extents[other_axes] / in_plane_resolution
    shape = [in_plane_resolution + 2] * 3
    shape[high_plane_axis] = length_resolution + 2
    solid = np.zeros(shape, dtype=bool)
    moved = np.moveaxis(solid, high_plane_axis, 0)
    normal = np.zeros(3)
    normal[high_plane_axis] = 1.0
    grid_a = np.arange(in_plane_resolution) + 1
    grid_b = np.arange(in_plane_resolution) + 1
    for slice_index in range(length_resolution):
        position = bounds[0, high_plane_axis] + (slice_index + 0.5) * pitches[high_plane_axis]
        point = np.zeros(3)
        point[high_plane_axis] = position
        section = mesh.section(plane_origin=point, plane_normal=normal)
        if section is None:
            continue
        raster = np.zeros((in_plane_resolution, in_plane_resolution), dtype=bool)
        # XOR handles outer loops and port-hole loops using the even-odd rule.
        for loop in section.discrete:
            if len(loop) < 3:
                continue
            coordinates = loop[:, other_axes]
            columns = (coordinates[:, 0] - bounds[0, other_axes[0]]) / pitches[other_axes[0]]
            rows = (coordinates[:, 1] - bounds[0, other_axes[1]]) / pitches[other_axes[1]]
            rr, cc = polygon(rows, columns, shape=raster.shape)
            layer = np.zeros_like(raster)
            layer[rr, cc] = True
            raster ^= layer
        moved[slice_index + 1, 1:-1, 1:-1] = raster
    if not np.any(solid):
        raise ValueError("No solid fuel was created from OBJ cross-sections")
    return solid, pitches


def central_port_mask(solid: np.ndarray, axis: int) -> np.ndarray:
    """Select the central bore on each axial cross-section.

    A through-port joins the exterior at the grain ends, so a single 3-D empty
    component incorrectly includes both the bore and outside air. Selecting the
    nearest empty 2-D component per slice preserves only the internal port.
    """
    port = np.zeros_like(solid, dtype=bool)
    slices = np.moveaxis(solid, axis, 0)
    port_slices = np.moveaxis(port, axis, 0)
    target = (np.asarray(slices.shape[1:], dtype=float) - 1.0) / 2.0
    structure = ndimage.generate_binary_structure(2, 1)
    selected = 0
    for index, fuel_slice in enumerate(slices):
        labels, count = ndimage.label(~fuel_slice, structure=structure)
        if count == 0:
            continue
        # The outside atmosphere always touches a slice edge. Remove all such
        # labels before choosing the port, including completely empty end slices.
        edge_labels = np.unique(np.concatenate((labels[0, :], labels[-1, :],
                                                labels[:, 0], labels[:, -1])))
        valid = labels > 0
        for edge_label in edge_labels:
            if edge_label != 0:
                valid[labels == edge_label] = False
        candidates = np.argwhere(valid)
        if len(candidates) == 0:
            continue
        nearest = candidates[np.argmin(np.sum((candidates - target) ** 2, axis=1))]
        label = labels[tuple(nearest)]
        port_slices[index] = labels == label
        selected += 1
    if selected == 0:
        raise ValueError("No empty port was found in any axial cross-section")
    return port


def interface_area(port: np.ndarray, fuel: np.ndarray, pitch: np.ndarray) -> float:
    """Fuel area facing the port, excluding the exterior grain surface, in mm²."""
    faces = 0
    for axis in range(3):
        port_before = np.take(port, range(port.shape[axis] - 1), axis=axis)
        port_after = np.take(port, range(1, port.shape[axis]), axis=axis)
        fuel_before = np.take(fuel, range(fuel.shape[axis] - 1), axis=axis)
        fuel_after = np.take(fuel, range(1, fuel.shape[axis]), axis=axis)
        face_area = float(np.prod(np.delete(pitch, axis)))
        faces += face_area * (np.count_nonzero(port_before & fuel_after) + np.count_nonzero(fuel_before & port_after))
    return float(faces)


def slice_metrics(port: np.ndarray, solid: np.ndarray, axis: int, index: int,
                  pitch: np.ndarray) -> tuple[float, float]:
    """Port area and fuel-facing perimeter for a specified voxel slice."""
    port_2d = np.take(port, index, axis=axis)
    solid_2d = np.take(solid, index, axis=axis)
    plane_axes = [value for value in range(3) if value != axis]
    area = float(np.count_nonzero(port_2d) * np.prod(pitch[plane_axes]))
    # Count only faces between the port and remaining fuel.  This deliberately
    # excludes port-to-outside faces after burn-through: the external grain
    # boundary has no fuel on its other side and therefore cannot burn.
    port_rows_before, port_rows_after = port_2d[:-1, :], port_2d[1:, :]
    fuel_rows_before, fuel_rows_after = solid_2d[:-1, :], solid_2d[1:, :]
    row_interfaces = (np.count_nonzero(port_rows_before & fuel_rows_after) +
                      np.count_nonzero(fuel_rows_before & port_rows_after))
    port_columns_before, port_columns_after = port_2d[:, :-1], port_2d[:, 1:]
    fuel_columns_before, fuel_columns_after = solid_2d[:, :-1], solid_2d[:, 1:]
    column_interfaces = (np.count_nonzero(port_columns_before & fuel_columns_after) +
                         np.count_nonzero(fuel_columns_before & port_columns_after))
    perimeter = float(row_interfaces * pitch[plane_axes[1]] +
                      column_interfaces * pitch[plane_axes[0]])
    return area, perimeter


def burning_surface_area(port: np.ndarray, fuel: np.ndarray, pitch: np.ndarray, axis: int) -> float:
    """Integrate the high-resolution port perimeter along the grain length.

    This avoids the artificial surface-area increase caused by counting the
    staircase faces between low-resolution length slices.
    """
    return float(sum(
        slice_metrics(port, fuel, axis, index, pitch)[1] * pitch[axis]
        for index in range(1, port.shape[axis] - 1)
    ))


def mesh_bore_surface_area(mesh: trimesh.Trimesh, axis: int) -> float:
    """Measure the initial bore directly from its triangular mesh surfaces.

    Unlike a perimeter multiplied by axial length, triangle areas retain the
    helical slope.  Bore components are the surfaces that run along the grain
    axis while remaining inside the outer radial envelope; end faces and the
    exterior casing are therefore excluded.
    """
    perpendicular_axes = [number for number in range(3) if number != axis]
    outer_radius = float(np.max(np.linalg.norm(mesh.vertices[:, perpendicular_axes], axis=1)))
    bore_area = 0.0
    for component in mesh.split(only_watertight=False):
        extent_ratio = component.extents[axis] / mesh.extents[axis]
        component_radius = float(np.max(
            np.linalg.norm(component.vertices[:, perpendicular_axes], axis=1)
        ))
        if extent_ratio > 0.95 and component_radius < outer_radius * 0.95:
            bore_area += float(component.area)
    if bore_area <= 0.0:
        raise ValueError("Could not identify an axial interior bore surface in the OBJ")
    return bore_area


def export_regressed_fuel_snapshot(solid: np.ndarray, initial_port: np.ndarray, pitch: np.ndarray,
                                   regression_mm: float, path: Path) -> None:
    """Export the remaining voxelized fuel at a requested regression distance as OBJ."""
    distance_to_port = ndimage.distance_transform_edt(~initial_port, sampling=pitch)
    remaining = solid & ~(solid & (distance_to_port <= regression_mm))
    if not np.any(remaining):
        print(f"Warning: snapshot at {regression_mm:.3f} mm was skipped because no fuel remains. "
              "Choose a smaller EXPORT_REGRESSED_SNAPSHOT_MM value to export a model.")
        return
    # Marching cubes turns the anisotropic voxel model into a triangle mesh.
    snapshot = trimesh.voxel.ops.matrix_to_marching_cubes(remaining, pitch=pitch)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.export(path)
    print(f"Regressed-fuel snapshot at {regression_mm:.3f} mm written to: {path}")


def centre_slice_metrics(port: np.ndarray, solid: np.ndarray, axis: int,
                         pitch: np.ndarray) -> tuple[float, float]:
    """Port area and wetted perimeter at the centre axial slice."""
    return slice_metrics(port, solid, axis, port.shape[axis] // 2, pitch)


def port_bounding_circles(port_2d: np.ndarray, plane_pitch: np.ndarray) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Return largest-inscribed and smallest-enclosing circles for a port slice.

    Coordinates and radii are in mm, with coordinates measured from the lower
    left of the displayed voxel slice.  The enclosing circle uses the convex
    hull of port pixels, which is sufficient because a circle is convex.
    """
    if not np.any(port_2d):
        return None, None
    distance = ndimage.distance_transform_edt(port_2d, sampling=plane_pitch)
    inscribed_index = np.unravel_index(np.argmax(distance), distance.shape)
    inscribed = (
        (inscribed_index[0] + 0.5) * plane_pitch[0],
        (inscribed_index[1] + 0.5) * plane_pitch[1],
        # EDT is measured centre-to-centre; move to the nearest voxel edge so
        # the green circle stays inside the orange port pixels on screen.
        max(0.0, float(distance[inscribed_index] - 0.5 * np.min(plane_pitch))),
    )

    # Pixel centres are enough for this display measurement and avoid turning
    # every filled pixel edge into a separate geometry primitive.
    points = (np.argwhere(port_2d).astype(float) + 0.5) * plane_pitch
    if len(points) == 1:
        return inscribed, (float(points[0, 0]), float(points[0, 1]), 0.5 * float(np.linalg.norm(plane_pitch)))
    try:
        points = points[ConvexHull(points).vertices]
    except QhullError:
        # A degenerate (one-pixel-wide) port is enclosed by its two extremes.
        first, last = points[0], points[-1]
        center = (first + last) / 2.0
        return inscribed, (float(center[0]), float(center[1]),
                           float(np.linalg.norm(last - first) / 2.0 + 0.5 * np.linalg.norm(plane_pitch)))

    # Deterministic randomized incremental minimum-enclosing-circle algorithm.
    points = points[np.random.default_rng(0).permutation(len(points))]
    center = points[0].copy()
    radius = 0.0
    tolerance = 1e-9
    for i, point in enumerate(points):
        if np.linalg.norm(point - center) <= radius + tolerance:
            continue
        center, radius = point.copy(), 0.0
        for j in range(i):
            other = points[j]
            if np.linalg.norm(other - center) <= radius + tolerance:
                continue
            center = (point + other) / 2.0
            radius = float(np.linalg.norm(other - point) / 2.0)
            for k in range(j):
                third = points[k]
                if np.linalg.norm(third - center) <= radius + tolerance:
                    continue
                matrix = 2.0 * np.array([other - point, third - point])
                rhs = np.array([np.dot(other, other) - np.dot(point, point),
                                np.dot(third, third) - np.dot(point, point)])
                try:
                    center = np.linalg.solve(matrix, rhs)
                except np.linalg.LinAlgError:
                    continue
                radius = float(np.linalg.norm(point - center))
    # Expand from pixel centres to their outer corners so the purple circle
    # actually encloses every displayed port voxel.
    enclosing = (float(center[0]), float(center[1]), radius + 0.5 * float(np.linalg.norm(plane_pitch)))
    return inscribed, enclosing


def regression_table(solid: np.ndarray, initial_port: np.ndarray, pitch: np.ndarray, axis: int,
                     distances: np.ndarray, mesh_volume: float, mesh_area: float,
                     initial_bore_area: float | None = None) -> list[dict[str, float]]:
    """Calculate port geometry at each normal-regression distance."""
    # distance_transform_edt measures distance from each fuel voxel to port.
    distance_to_port = ndimage.distance_transform_edt(~initial_port, sampling=pitch)
    initial_fuel_volume = float(np.count_nonzero(solid) * np.prod(pitch))
    rows: list[dict[str, float]] = []
    for regression in distances:
        burned = solid & (distance_to_port <= regression)
        port = initial_port | burned
        remaining = solid & ~burned
        port_area, perimeter = centre_slice_metrics(port, remaining, axis, pitch)
        rows.append({
            "regression_mm": float(regression),
            "center_port_area_mm2": port_area,
            "center_port_perimeter_mm": perimeter,
            "center_port_hydraulic_diameter_mm": float(4.0 * port_area / perimeter) if perimeter > 0 else 0.0,
            "port_volume_mm3": float(np.count_nonzero(port) * np.prod(pitch)),
            "burned_fuel_volume_mm3": float(np.count_nonzero(burned) * np.prod(pitch)),
            "remaining_fuel_volume_mm3": float(np.count_nonzero(remaining) * np.prod(pitch)),
            "initial_fuel_volume_mm3": initial_fuel_volume,
            "burning_surface_area_mm2": burning_surface_area(port, remaining, pitch, axis),
            "mesh_enclosed_volume_mm3": mesh_volume,
            "mesh_total_surface_area_mm2": mesh_area,
        })
    if initial_bore_area is not None:
        measured_initial_area = rows[0]["burning_surface_area_mm2"]
        if measured_initial_area <= 0.0:
            raise ValueError("Cannot normalize bore surface area: the initial port has no surface")
        # Keep later voxel regression states continuous with the exact initial
        # triangle measurement, without using a hand-entered reference value.
        correction = initial_bore_area / measured_initial_area
        for row in rows:
            row["burning_surface_area_mm2"] *= correction
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    import csv
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def picked_plane_regression(solid: np.ndarray, initial_port: np.ndarray, pitch: np.ndarray, axis: int,
                            distances: np.ndarray) -> list[dict[str, float]]:
    """Calculate detailed 2-D regression measurements for the selected plane."""
    distance_to_port = ndimage.distance_transform_edt(~initial_port, sampling=pitch)
    plane_axes = [number for number in range(3) if number != axis]
    rows: list[dict[str, float]] = []
    slice_index = solid.shape[axis] // 2

    def circle_contact_metrics(port_2d: np.ndarray, fuel_2d: np.ndarray) -> tuple[float, float, float]:
        """Return max-inscribed diameter, total contact, and largest contact arc."""
        spacing = np.asarray(pitch[plane_axes], dtype=float)
        distances_2d = ndimage.distance_transform_edt(port_2d, sampling=spacing)
        radius = float(np.max(distances_2d))
        if radius <= 0.0:
            return 0.0, 0.0, 0.0
        center = np.asarray(np.unravel_index(np.argmax(distances_2d), distances_2d.shape), dtype=float)
        sample_count = max(360, int(np.ceil(2.0 * np.pi * radius / np.min(spacing))))
        angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
        # Sample just outside the inscribed circle; fuel there represents the
        # circle's actual contact with the port boundary / star arms.
        rows = np.rint(center[0] + (radius / spacing[0] + 0.75) * np.sin(angles)).astype(int)
        columns = np.rint(center[1] + (radius / spacing[1] + 0.75) * np.cos(angles)).astype(int)
        valid = ((rows >= 0) & (rows < fuel_2d.shape[0]) & (columns >= 0) & (columns < fuel_2d.shape[1]))
        contacts = np.zeros(sample_count, dtype=bool)
        contacts[valid] = fuel_2d[rows[valid], columns[valid]]
        arc_step = 2.0 * np.pi * radius / sample_count
        total_contact = float(np.count_nonzero(contacts) * arc_step)
        if not np.any(contacts):
            return 2.0 * radius, total_contact, 0.0
        # Handle an arc which crosses angle 0 by examining doubled samples.
        doubled = np.concatenate((contacts, contacts))
        changes = np.diff(np.r_[False, doubled, False].astype(int))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        largest_run = min(sample_count, max(ends - starts))
        return 2.0 * radius, total_contact, float(largest_run * arc_step)

    for regression in distances:
        burned = solid & (distance_to_port <= regression)
        port = initial_port | burned
        remaining = solid & ~burned
        port_2d = np.take(port, slice_index, axis=axis)
        solid_2d = np.take(solid, slice_index, axis=axis)
        # Separate true outside air (the empty component touching the image
        # border) from the enclosed initial port. Burn-through occurs only
        # when burning reaches fuel directly adjacent to that outside air.
        empty_labels, _ = ndimage.label(~solid_2d, structure=ndimage.generate_binary_structure(2, 1))
        edge_labels = np.unique(np.concatenate((empty_labels[0, :], empty_labels[-1, :],
                                                  empty_labels[:, 0], empty_labels[:, -1])))
        outside_air = np.isin(empty_labels, edge_labels[edge_labels != 0])
        outer_fuel_perimeter = solid_2d & ndimage.binary_dilation(
            outside_air, structure=ndimage.generate_binary_structure(2, 1)
        )
        burnthrough = bool(np.any(port_2d & outer_fuel_perimeter))
        area, perimeter = slice_metrics(port, remaining, axis, slice_index, pitch)
        max_inscribed_diameter, circle_overlap, largest_arm_contact = circle_contact_metrics(port_2d, np.take(remaining, slice_index, axis=axis))
        port_points = np.argwhere(port_2d)
        if len(port_points):
            center = np.mean(port_points * pitch[plane_axes], axis=0)
            min_enclosing_diameter = float(2.0 * np.max(np.linalg.norm(port_points * pitch[plane_axes] - center, axis=1)))
        else:
            min_enclosing_diameter = 0.0
        rows.append({
            "regression_mm": float(regression),
            "time_s": float(regression / REGRESSION_RATE_MM_PER_S),
            "port_area_mm2": area,
            "port_perimeter_mm": perimeter,
            "hydraulic_diameter_mm": float(4.0 * area / perimeter) if perimeter else 0.0,
            "max_inscribed_diameter_mm": max_inscribed_diameter,
            "min_enclosing_diameter_mm": min_enclosing_diameter,
            "circle_boundary_overlap_mm": circle_overlap,
            "largest_arm_contact_mm": largest_arm_contact,
            "remaining_fuel_area_mm2": float(np.count_nonzero(np.take(remaining, slice_index, axis=axis)) *
                                              np.prod(pitch[plane_axes])),
            "port_volume_mm3": float(np.count_nonzero(port) * np.prod(pitch)),
            "burned_fuel_volume_mm3": float(np.count_nonzero(burned) * np.prod(pitch)),
            "burning_surface_area_mm2": burning_surface_area(port, remaining, pitch, axis),
            "burnthrough": burnthrough,
        })
    return rows


def polynomial_fit(x: np.ndarray, y: np.ndarray, target_r2: float = 0.99) -> tuple[np.poly1d, int, float]:
    """Choose the lowest polynomial degree (2--5) that reaches the R-squared target."""
    maximum_degree = min(5, len(x) - 1)
    best: tuple[np.poly1d, int, float] | None = None
    for degree in range(2, maximum_degree + 1):
        poly = np.poly1d(np.polyfit(x, y, degree))
        fitted = poly(x)
        total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - float(np.sum((y - fitted) ** 2)) / total if total else 1.0
        best = (poly, degree, r_squared)
        if r_squared >= target_r2:
            break
    if best is None:
        return np.poly1d([float(y[0])]), 0, 1.0
    return best


def polynomial_equation(poly: np.poly1d, variable: str = "r") -> str:
    """Format a polynomial in the displayed plot units."""
    degree = poly.order
    terms: list[str] = []
    for index, coefficient in enumerate(poly.c):
        power = degree - index
        if abs(coefficient) < 1e-12:
            continue
        if power == 0:
            term = f"{abs(coefficient):.6g}"
        elif power == 1:
            term = f"{abs(coefficient):.6g}{variable}"
        else:
            term = f"{abs(coefficient):.6g}{variable}^{power}"
        terms.append(("- " if coefficient < 0 else "+ ") + term)
    if not terms:
        return "y = 0"
    expression = " ".join(terms).removeprefix("+ ")
    return f"y = {expression}"


def piecewise_polynomial_fit(x: np.ndarray, y: np.ndarray, burnthrough_index: int | None):
    """Fit separate curves before and after selected-plane burn-through."""
    if burnthrough_index is None or burnthrough_index < 2 or len(x) - burnthrough_index < 3:
        poly, degree, r_squared = polynomial_fit(x, y)
        return [("all regression", x, poly, degree, r_squared)]
    return [("before burn-through", x[:burnthrough_index + 1], *polynomial_fit(x[:burnthrough_index + 1], y[:burnthrough_index + 1])),
            ("after burn-through", x[burnthrough_index:], *polynomial_fit(x[burnthrough_index:], y[burnthrough_index:]))]


def save_picked_plane_analysis(rows: list[dict[str, float]], axis: int, path: Path | None,
                               display_unit: str, display_scale: float, csv_path: Path,
                               curve_fits_path: Path, show: bool = False) -> None:
    """Save selected-plane regression data and fitted engineering curves."""
    fuel_exhaustion_regression = None
    if STOP_SELECTED_PLANE_GRAPHS_WHEN_FUEL_IS_GONE:
        empty_indices = np.flatnonzero(np.asarray(
            [row["remaining_fuel_area_mm2"] <= 0.0 for row in rows], dtype=bool
        ))
        if len(empty_indices):
            # Include the first zero-fuel sample as the final graph point.
            fuel_exhaustion_regression = rows[int(empty_indices[0])]["regression_mm"] / display_scale
            rows = rows[:int(empty_indices[0]) + 1]
    write_csv(rows, csv_path)
    x = np.asarray([row["regression_mm"] for row in rows]) / display_scale
    charts = [
        ("port_area_mm2", f"Port area ({display_unit}²)", display_scale ** 2),
        ("port_perimeter_mm", f"Port perimeter ({display_unit})", display_scale),
        ("hydraulic_diameter_mm", f"Hydraulic diameter ({display_unit})", display_scale),
        ("max_inscribed_diameter_mm", f"Max inscribed diameter ({display_unit})", display_scale),
        ("min_enclosing_diameter_mm", f"Min enclosing diameter ({display_unit})", display_scale),
        ("circle_boundary_overlap_mm", f"Circle boundary contact ({display_unit})", display_scale),
        ("largest_arm_contact_mm", f"Largest contact arc ({display_unit})", display_scale),
        ("port_volume_mm3", f"Whole-grain port volume ({display_unit}³)", display_scale ** 3),
        ("burned_fuel_volume_mm3", f"Burned fuel volume ({display_unit}³)", display_scale ** 3),
        ("burning_surface_area_mm2", f"Whole-grain burning surface ({display_unit}²)", display_scale ** 2),
    ]
    core_figure, core_plots = plt.subplots(4, 2, figsize=(13, 17), constrained_layout=True)
    contact_figure, contact_plots = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # Keep chart order while placing core geometry in the first window and
    # contact-only measurements in the second.
    plots = np.asarray([
        core_plots.flat[0], core_plots.flat[1], core_plots.flat[2], core_plots.flat[3],
        contact_plots.flat[0], contact_plots.flat[1], contact_plots.flat[2],
        core_plots.flat[4], core_plots.flat[5], core_plots.flat[6],
    ])
    core_plots.flat[7].set_visible(False)
    contact_plots.flat[3].set_visible(False)
    fit_rows: list[dict[str, float | str]] = []
    indices = np.flatnonzero(np.asarray([row["burnthrough"] for row in rows], dtype=bool))
    burnthrough_index = int(indices[0]) if len(indices) else None
    burnthrough = x[burnthrough_index] if burnthrough_index is not None else None
    for plot, (key, label, scale) in zip(plots.flat, charts):
        y = np.asarray([row[key] for row in rows]) / scale
        branch_equations: list[str] = []
        for number, (branch, branch_x, branch_poly, branch_degree, branch_r_squared) in enumerate(
                piecewise_polynomial_fit(x, y, burnthrough_index)):
            smooth_branch_x = np.linspace(branch_x.min(), branch_x.max(), 200)
            plot.plot(smooth_branch_x, branch_poly(smooth_branch_x), "--",
                      color=("#f97316", "#16a34a")[number], linewidth=2,
                      label=f"{branch}: degree {branch_degree}, R2={branch_r_squared:.5f}")
            fit_rows.append({"metric": key, "display_unit": label, "branch": branch,
                             "burnthrough_regression": burnthrough if burnthrough is not None else "not reached",
                             "degree": branch_degree, "r_squared": branch_r_squared,
                             "equation": polynomial_equation(branch_poly),
                             **{f"coefficient_r_power_{branch_degree - index}": float(value)
                                for index, value in enumerate(branch_poly.c)}})
            branch_equations.append(f"{branch}: {polynomial_equation(branch_poly)}\nR² = {branch_r_squared:.6f}")
        if burnthrough is not None:
            plot.axvline(burnthrough, color="#dc2626", linestyle="--", linewidth=1.2,
                         label=f"Burn-through: {burnthrough:.3f} {display_unit}")
        # Retained variables keep older plotting code below harmless; the
        # actual visible fits are the orange/green piecewise branches above.
        smooth_x = np.empty(0)
        poly = np.poly1d([0.0])
        degree = 0
        r_squared = 1.0
        plot.plot(x, y, "o", color="#2563eb", label="Measured", markersize=4)
        plot.plot(smooth_x, poly(smooth_x), "-", color="#f97316",
                  label=f"Polynomial fit (degree {degree}, R²={r_squared:.5f})")
        plot.set_xlabel(f"Regression distance ({display_unit})")
        plot.set_ylabel(label)
        plot.grid(True, alpha=0.3)
        handles, labels = plot.get_legend_handles_labels()
        visible = [(handle, text) for handle, text in zip(handles, labels)
                   if not text.startswith("Polynomial fit")]
        plot.legend(*zip(*visible), fontsize=8)
        # Overlay the branch equations so the displayed equations match the
        # orange/green piecewise curves rather than the legacy whole-range fit.
        plot.text(0.02, 0.98, "\n\n".join(branch_equations), transform=plot.transAxes,
                  va="top", fontsize=8,
                  bbox={"boxstyle": "round", "facecolor": "white", "alpha": 1.0})
        plot.text(0.02, 0.98, f"{polynomial_equation(poly)}\nR² = {r_squared:.6f}",
                  transform=plot.transAxes, va="top", fontsize=8,
                  bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
        fit_rows.append({"metric": key, "display_unit": label, "degree": degree,
                         "r_squared": r_squared, "equation": polynomial_equation(poly),
                         **{f"coefficient_r_power_{degree - index}": float(value)
                            for index, value in enumerate(poly.c)}})
        # Hide annotations from the previous whole-range plotting path.
        for annotation in plot.texts:
            annotation.set_visible(False)
        if SHOW_CURVE_FIT_EQUATIONS_ON_PLOTS:
            plot.text(0.02, 0.98, "\n\n".join(branch_equations), transform=plot.transAxes,
                      va="top", fontsize=7,
                      bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    status = (f"burn-through at {burnthrough:.3f} {display_unit}"
              if burnthrough is not None else "burn-through not reached")
    core_figure.suptitle(
        f"Selected {('X', 'Y', 'Z')[axis]} plane: core geometry fits ({status})"
    )
    contact_figure.suptitle(
        f"Selected {('X', 'Y', 'Z')[axis]} plane: circle/contact fits ({status})"
    )
    helix_figure = None
    helix_path = None
    if "pitch_mm" in rows[0] and "radius_of_curvature_mm" in rows[0]:
        helix_figure, helix_plots = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        helix_metrics = (("pitch_mm", "Helix pitch P"), ("radius_of_curvature_mm", "Helix curvature radius R_c"))
        for plot, (key, label) in zip(helix_plots, helix_metrics):
            values = np.asarray([row[key] for row in rows]) / display_scale
            plot.plot(x, values, "o-", color="#7e22ce", markersize=3)
            plot.set_xlabel(f"Regression distance ({display_unit})")
            plot.set_ylabel(f"{label} ({display_unit})")
            plot.grid(True, alpha=0.3)
            plot.text(0.02, 0.95, "Constant in the current uniform-regression model",
                      transform=plot.transAxes, va="top", fontsize=8)
        helix_figure.suptitle("Helix geometry versus regression")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        core_figure.savefig(path, dpi=180, bbox_inches="tight")
        contact_path = path.with_name(f"{path.stem}_contact_metrics{path.suffix}")
        contact_figure.savefig(contact_path, dpi=180, bbox_inches="tight")
        if helix_figure is not None:
            helix_path = path.with_name(f"{path.stem}_helix_geometry{path.suffix}")
            helix_figure.savefig(helix_path, dpi=180, bbox_inches="tight")
    write_csv([row for row in fit_rows if "branch" in row], curve_fits_path)
    print(f"Selected-plane data written to: {csv_path}")
    if path is not None:
        print(f"Selected-plane plots written to: {path}")
        print(f"Selected-plane contact plots written to: {contact_path}")
        if helix_path is not None:
            print(f"Helix-geometry plot written to: {helix_path}")
    print(f"Selected-plane curve-fit equations written to: {curve_fits_path}")
    print(f"Selected-plane {status}")
    if STOP_SELECTED_PLANE_GRAPHS_WHEN_FUEL_IS_GONE:
        if fuel_exhaustion_regression is None:
            print("Selected-plane fuel was not exhausted within the simulated regression range.")
        else:
            print(f"Selected-plane graphs stop at fuel exhaustion: {fuel_exhaustion_regression:.3f} {display_unit}")
    if show:
        print("Close the selected-plane curve-fit window to finish.")
        plt.show()
    plt.close(core_figure)
    plt.close(contact_figure)
    if helix_figure is not None:
        plt.close(helix_figure)


def save_visualization(solid: np.ndarray, initial_port: np.ndarray, pitch: float, axis: int,
                       distances: np.ndarray, rows: list[dict[str, float]], path: Path,
                       regression_rate: float, display_unit: str, display_scale: float,
                       show: bool = False) -> None:
    """Save cross-sections and geometry-history plots to one PNG image."""
    distance_to_port = ndimage.distance_transform_edt(~initial_port, sampling=pitch)
    displayed_indices = np.unique(np.linspace(0, len(distances) - 1, min(6, len(distances))).round().astype(int))
    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, 6)
    cmap = ListedColormap(["#f5f5f5", "#4b5563", "#f97316"])

    for column, row_index in enumerate(displayed_indices):
        regression = distances[row_index]
        burned = solid & (distance_to_port <= regression)
        remaining = solid & ~burned
        port = initial_port | burned
        # 0 = open space, 1 = remaining fuel, 2 = port / burned space.
        image = np.zeros(solid.shape, dtype=np.uint8)
        image[remaining] = 1
        image[port] = 2
        centre = image.shape[axis] // 2
        section = np.take(image, centre, axis=axis)
        plot = figure.add_subplot(grid[0, column])
        plot.imshow(section.T, origin="lower", cmap=cmap,
                    interpolation="bicubic" if SMOOTH_VIEWER_RENDERING else "nearest")
        plot.set_title(f"{regression / display_scale:.2f} {display_unit}")
        plot.set_xticks([])
        plot.set_yticks([])
        if column == 0:
            plot.set_ylabel("Center cross-section\norange = port")

    charts = [
        ("center_port_area_mm2", f"Center port area ({display_unit}²)", display_scale ** 2),
        ("center_port_perimeter_mm", f"Center perimeter ({display_unit})", display_scale),
        ("port_volume_mm3", f"Port volume ({display_unit}³)", display_scale ** 3),
        ("burning_surface_area_mm2", f"Burning surface area ({display_unit}²)", display_scale ** 2),
    ]
    values_x = np.asarray([row["regression_mm"] for row in rows]) / display_scale
    for chart_index, (key, label, value_scale) in enumerate(charts):
        plot = figure.add_subplot(grid[1 + chart_index // 2, (chart_index % 2) * 3:(chart_index % 2 + 1) * 3])
        values_y = np.asarray([row[key] for row in rows]) / value_scale
        plot.plot(values_x, values_y, "o-", color="#2563eb", markersize=3)
        plot.set_xlabel(f"Regression distance ({display_unit})")
        plot.set_ylabel(label)
        plot.grid(True, alpha=0.3)

    figure.suptitle("3-D fuel-port regression: center slices and geometry history", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Visualization written to: {path}")
    if show:
        plt.show()
    plt.close(figure)


def show_interactive_viewer(solid: np.ndarray, initial_port: np.ndarray, pitch: float,
                            distances: np.ndarray, rows: list[dict[str, float]],
                            initial_axis: int, regression_rate: float,
                            display_unit: str, display_scale: float,
                            helix_dimensions: dict[str, float] | None = None) -> None:
    """Open a slider-controlled cross-section viewer for the regression model."""
    distance_to_port = ndimage.distance_transform_edt(~initial_port, sampling=pitch)
    states: list[tuple[np.ndarray, np.ndarray]] = []
    print("Preparing interactive regression states...")
    for regression in distances:
        burned = solid & (distance_to_port <= regression)
        states.append((initial_port | burned, solid & ~burned))

    axis_names = ("X", "Y", "Z")
    cmap = ListedColormap(["#f5f5f5", "#4b5563", "#f97316"])
    figure = plt.figure(figsize=(14, 8.5))
    image_axis = figure.add_axes((0.10, 0.25, 0.52, 0.58))
    info = figure.text(0.67, 0.74, "", va="top", fontsize=11,
                       bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    figure.text(0.10, 0.975,
                "Interactive fuel-grain regression\nOrange = port / burned fuel; gray = remaining fuel; cyan = original port outline",
                fontsize=13, va="top")

    regression_slider_axis = figure.add_axes((0.10, 0.135, 0.47, 0.03))
    regression_slider = Slider(regression_slider_axis, "Time / regression", 0, len(distances) - 1,
                               valinit=0, valstep=1)
    slice_slider_axis = figure.add_axes((0.10, 0.075, 0.47, 0.03))
    slice_slider = Slider(slice_slider_axis, "Slice position", 0, solid.shape[initial_axis] - 1,
                          valinit=solid.shape[initial_axis] // 2, valstep=1)
    radio_axis = figure.add_axes((0.83, 0.055, 0.08, 0.13))
    radio_buttons = RadioButtons(radio_axis, axis_names, active=initial_axis)
    overlay_axis = figure.add_axes((0.65, 0.205, 0.25, 0.07))
    original_overlay = CheckButtons(overlay_axis, ["Show 0-regression outline"], [True])
    circle_overlay = CheckButtons(figure.add_axes((0.65, 0.285, 0.25, 0.07)),
                                  ["Show inscribed / enclosing circles"], [False])
    helix_overlay = None
    if helix_dimensions is not None and SHOW_HELIX_DIMENSIONS_IN_VIEWER:
        helix_overlay = CheckButtons(figure.add_axes((0.65, 0.365, 0.25, 0.07)),
                                     ["Show helix dimensions"], [True])
    state_box_axis = figure.add_axes((0.65, 0.125, 0.11, 0.04))
    state_box = TextBox(state_box_axis, "State", initial="0")
    slice_box_axis = figure.add_axes((0.65, 0.065, 0.11, 0.04))
    slice_box = TextBox(slice_box_axis, "Slice", initial=str(solid.shape[initial_axis] // 2))

    selected_axis = [initial_axis]
    changing_axis = [False]
    show_original_outline = [True]
    show_bounding_circles = [False]
    show_helix_dimensions = [helix_overlay is not None]

    def redraw(_value=None) -> None:
        state_index = int(regression_slider.val)
        slice_index = int(slice_slider.val)
        selected_plane = selected_axis[0]
        regression_slider.label.set_text(
            f"Time / regression: {distances[state_index] / regression_rate:.3f} s / "
            f"{distances[state_index] / display_scale:.3f} {display_unit}"
        )
        port, remaining = states[state_index]
        image = np.zeros(solid.shape, dtype=np.uint8)
        image[remaining] = 1
        image[port] = 2
        section = np.take(image, slice_index, axis=selected_plane)
        port_area, perimeter = slice_metrics(port, remaining, selected_plane, slice_index, pitch)
        hydraulic_diameter_text = (
            f"{4.0 * port_area / perimeter / display_scale:,.3f} {display_unit}"
            if perimeter > 0.0 else "N/A (no enclosed port in this slice)"
        )
        image_axis.clear()
        remaining_axes = [number for number in range(3) if number != selected_plane]
        image_axis.imshow(section.T, origin="lower", cmap=cmap,
                          interpolation="bicubic" if SMOOTH_VIEWER_RENDERING else "nearest",
                          extent=(0, section.shape[0] * pitch[remaining_axes[0]] / display_scale,
                                  0, section.shape[1] * pitch[remaining_axes[1]] / display_scale))
        if show_original_outline[0]:
            original_port_slice = np.take(initial_port, slice_index, axis=selected_plane)
            image_axis.contour(original_port_slice.T.astype(float), levels=[0.5], colors="#06b6d4",
                               linewidths=1.5,
                               extent=(0, section.shape[0] * pitch[remaining_axes[0]] / display_scale,
                                       0, section.shape[1] * pitch[remaining_axes[1]] / display_scale))
        inscribed_circle = enclosing_circle = None
        if show_bounding_circles[0]:
            inscribed_circle, enclosing_circle = port_bounding_circles(
                np.take(port, slice_index, axis=selected_plane), pitch[remaining_axes]
            )
            for circle, color in ((inscribed_circle, "#22c55e"), (enclosing_circle, "#a855f7")):
                if circle is not None:
                    center_x, center_y, radius = circle
                    image_axis.add_patch(plt.Circle(
                        (center_x / display_scale, center_y / display_scale), radius / display_scale,
                        fill=False, color=color, linewidth=2.0, linestyle="--",
                    ))
        if show_helix_dimensions[0] and helix_dimensions is not None:
            width = section.shape[0] * pitch[remaining_axes[0]] / display_scale
            height = section.shape[1] * pitch[remaining_axes[1]] / display_scale
            pitch_length = helix_dimensions["pitch_mm"] / display_scale
            curvature_radius = helix_dimensions["radius_of_curvature_mm"] / display_scale
            # A transverse view cannot show pitch; it reports the calculated
            # helix radius of curvature instead.
            if selected_plane == initial_axis:
                image_axis.text(width * 0.5, height * 0.08,
                                f"Helix curvature radius R_c = {curvature_radius:.2f} {display_unit}",
                                color="#7e22ce", ha="center", fontsize=9,
                                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
            # A longitudinal view supports pitch P along the motor-axis direction.
            elif initial_axis in remaining_axes:
                direction = remaining_axes.index(initial_axis)
                span = width if direction == 0 else height
                start, end = max(0.0, span * 0.5 - pitch_length / 2), min(span, span * 0.5 + pitch_length / 2)
                level = height * 0.08 if direction == 0 else width * 0.08
                if direction == 0:
                    image_axis.annotate("", xy=(end, level), xytext=(start, level),
                                        arrowprops={"arrowstyle": "<->", "color": "#dc2626", "linewidth": 1.5})
                    image_axis.text((start + end) / 2, level + height * 0.035, f"P = {pitch_length:.2f} {display_unit}",
                                    color="#b91c1c", ha="center", fontsize=9)
                else:
                    image_axis.annotate("", xy=(level, end), xytext=(level, start),
                                        arrowprops={"arrowstyle": "<->", "color": "#dc2626", "linewidth": 1.5})
                    image_axis.text(level + width * 0.035, (start + end) / 2, f"P = {pitch_length:.2f} {display_unit}",
                                    color="#b91c1c", va="center", fontsize=9)
        remaining_names = [axis_names[number] for number in remaining_axes]
        image_axis.set_xlabel(f"{remaining_names[0]} direction ({display_unit})")
        image_axis.set_ylabel(f"{remaining_names[1]} direction ({display_unit})")
        image_axis.set_title(f"{axis_names[selected_plane]} slice {slice_index}  |  "
                             f"time = {distances[state_index] / regression_rate:.3f} s  |  "
                             f"regression = {distances[state_index] / display_scale:.3f} {display_unit}",
                             pad=10)
        row = rows[state_index]
        circle_text = ""
        if show_bounding_circles[0]:
            inscribed_diameter = 2.0 * inscribed_circle[2] / display_scale if inscribed_circle else 0.0
            enclosing_diameter = 2.0 * enclosing_circle[2] / display_scale if enclosing_circle else 0.0
            circle_text = (f"\n\nLargest enclosed circle diameter: {inscribed_diameter:,.3f} {display_unit}\n"
                           f"Smallest enclosing circle diameter: {enclosing_diameter:,.3f} {display_unit}\n"
                           f"Diameter difference: {enclosing_diameter - inscribed_diameter:,.3f} {display_unit}")
        info.set_text(
            circle_text +
            f"Selected {axis_names[selected_plane]} slice\n"
            f"Port area: {port_area / display_scale ** 2:,.2f} {display_unit}²\n"
            f"Port perimeter: {perimeter / display_scale:,.2f} {display_unit}\n\n"
            f"Hydraulic diameter (4A/P): {hydraulic_diameter_text}\n\n"
            f"Whole grain at this state\n"
            f"Port volume: {row['port_volume_mm3'] / display_scale ** 3:,.2f} {display_unit}³\n"
            f"Burned fuel: {row['burned_fuel_volume_mm3'] / display_scale ** 3:,.2f} {display_unit}³\n"
            f"Remaining fuel: {row['remaining_fuel_volume_mm3'] / display_scale ** 3:,.2f} {display_unit}³\n"
            f"Burning surface: {row['burning_surface_area_mm2'] / display_scale ** 2:,.2f} {display_unit}²"
        )
        figure.canvas.draw_idle()

    def change_axis(label: str) -> None:
        selected_axis[0] = axis_names.index(label)
        new_maximum = solid.shape[selected_axis[0]] - 1
        changing_axis[0] = True
        slice_slider.valmax = new_maximum
        slice_slider.ax.set_xlim(slice_slider.valmin, new_maximum)
        slice_slider.set_val(min(new_maximum, solid.shape[selected_axis[0]] // 2))
        changing_axis[0] = False
        redraw()

    def set_slider_from_text(text: str, slider: Slider) -> None:
        """Accept an integer typed beside a slider and clamp it to its range."""
        try:
            value = int(float(text.strip()))
        except ValueError:
            return
        slider.set_val(int(np.clip(value, slider.valmin, slider.valmax)))

    regression_slider.on_changed(redraw)
    slice_slider.on_changed(lambda value: None if changing_axis[0] else redraw(value))
    radio_buttons.on_clicked(change_axis)
    original_overlay.on_clicked(lambda _label: (show_original_outline.__setitem__(0, not show_original_outline[0]), redraw()))
    circle_overlay.on_clicked(lambda _label: (show_bounding_circles.__setitem__(0, not show_bounding_circles[0]), redraw()))
    if helix_overlay is not None:
        helix_overlay.on_clicked(lambda _label: (show_helix_dimensions.__setitem__(0, not show_helix_dimensions[0]), redraw()))
    state_box.on_submit(lambda text: set_slider_from_text(text, regression_slider))
    slice_box.on_submit(lambda text: set_slider_from_text(text, slice_slider))
    redraw()
    print("Interactive viewer is ready. Close its window to finish the script.")
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Voxel-based 3-D port-regression analysis for OBJ fuel grains.")
    parser.add_argument(
        "obj_file", type=Path, nargs="?", default=Path(OBJ_FILE),
        help=f"Closed OBJ mesh of the solid fuel grain (default: {OBJ_FILE})",
    )
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help=f"Voxels across the largest grain dimension (default: {RESOLUTION})")
    parser.add_argument("--max-regression", type=float, default=None, help="Override final regression distance, mm")
    parser.add_argument("--steps", type=int, default=SLIDER_STEPS,
                        help=f"Number of regression/time states including zero (default: {SLIDER_STEPS})")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), help="Motor axis: 0=X, 1=Y, 2=Z; default is longest dimension")
    parser.add_argument("--unit-scale-mm", type=float, default=None, help="Override automatic OBJ scale (metres: 1000; inches: 25.4; millimetres: 1)")
    output_folder = Path(OUTPUT_FOLDER)
    parser.add_argument("--output", type=Path, default=output_folder / "port_regression.csv", help="CSV output path")
    parser.add_argument("--plot", type=Path, default=output_folder / "port_regression_visualization.png", help="PNG visualization output path")
    parser.add_argument("--plane-plot", type=Path, default=output_folder / SELECTED_PLANE_ANALYSIS_PLOT,
                        help="PNG of selected-plane measurements and curve fits")
    parser.add_argument("--plane-output", type=Path, default=output_folder / SELECTED_PLANE_ANALYSIS_CSV,
                        help="CSV of selected-plane regression measurements")
    parser.add_argument("--curve-fits-output", type=Path, default=output_folder / SELECTED_PLANE_CURVE_FITS_CSV,
                        help="CSV of selected-plane curve-fit equations and coefficients")
    parser.add_argument("--no-plane-analysis", action="store_true",
                        help="Skip selected-plane CSV and curve-fit plots")
    parser.add_argument("--show", action="store_true", help="Open the saved summary plot as well as the interactive viewer")
    parser.add_argument("--no-interactive", action="store_true", help="Do not open the interactive regression viewer")
    args = parser.parse_args()
    if not args.obj_file.is_file():
        parser.error(f"OBJ file not found: {args.obj_file}")
    max_regression_mm = (REGRESSION_RATE_MM_PER_S * SIMULATION_TIME_S
                         if args.max_regression is None else args.max_regression)
    if (args.resolution < 20 or args.steps < 2 or SELECTED_PLANE_GRAPH_POINTS < 3 or max_regression_mm < 0 or
            (args.unit_scale_mm is not None and args.unit_scale_mm <= 0)):
        parser.error("resolution must be >=20, steps >=2, SELECTED_PLANE_GRAPH_POINTS >=3, and scales/distances non-negative")
    if IS_HELICAL_SWEEP:
        dimensions = (HELIX_CENTERLINE_RADIUS_MM, HELIX_PITCH_MM)
        if any(value is None or value <= 0 for value in dimensions):
            parser.error("Enter positive HELIX_CENTERLINE_RADIUS_MM and HELIX_PITCH_MM when IS_HELICAL_SWEEP is True")

    scale_to_mm = infer_unit_scale_to_mm(args.obj_file, args.unit_scale_mm)
    print("Loading OBJ mesh...")
    mesh = load_mesh(args.obj_file, scale_to_mm)
    plane_name = HIGH_RESOLUTION_PLANE.strip().upper()
    if plane_name not in ("X", "Y", "Z"):
        parser.error("HIGH_RESOLUTION_PLANE must be X, Y, or Z")
    axis = ("X", "Y", "Z").index(plane_name) if args.axis is None else args.axis
    in_plane_resolution = IN_PLANE_RESOLUTION if args.resolution == RESOLUTION else args.resolution
    estimated_grid = (in_plane_resolution + 2) ** 2 * (LENGTH_RESOLUTION + 2)
    print(f"High-resolution plane: {plane_name} ({in_plane_resolution} x {in_plane_resolution}); "
          f"length slices: {LENGTH_RESOLUTION}; total grid: about {estimated_grid:,} cells.")
    if SHOW_PLANE_PREVIEW:
        axis = preview_high_resolution_plane(mesh, axis, in_plane_resolution)
        plane_name = ("X", "Y", "Z")[axis]
        print(f"Selected high-resolution plane: {plane_name}")
    with activity_indicator("Rasterizing high-resolution cross-sections...", len(mesh.faces)):
        solid, pitch = make_solid_voxels(mesh, axis, in_plane_resolution, LENGTH_RESOLUTION)
    print("Identifying the centre port and calculating regression...")
    port = central_port_mask(solid, axis)
    if EXPORT_REGRESSED_SNAPSHOT_MM is not None:
        if EXPORT_REGRESSED_SNAPSHOT_MM < 0:
            parser.error("EXPORT_REGRESSED_SNAPSHOT_MM must be non-negative or None")
        export_regressed_fuel_snapshot(
            solid, port, pitch, EXPORT_REGRESSED_SNAPSHOT_MM,
            Path(OUTPUT_FOLDER) / REGRESSED_SNAPSHOT_FILENAME,
        )
    display_unit, display_scale = display_unit_scale(DISPLAY_LENGTH_UNIT)
    distances = np.linspace(0.0, max_regression_mm, args.steps)
    if not mesh.is_watertight:
        # A signed volume calculated from an open mesh is not physically valid.
        # The voxel fuel volume is still reported and is the appropriate value.
        mesh_volume = float("nan")
        print("Warning: OBJ is not watertight; mesh_enclosed_volume_mm3 is unavailable. "
              "Use initial_fuel_volume_mm3 (the voxel estimate) instead.")
    else:
        mesh_volume = abs(float(mesh.volume))
    bore_area = mesh_bore_surface_area(mesh, axis)
    print(f"Initial bore surface measured from OBJ triangles: {bore_area:.6f} mm^2")
    rows = regression_table(solid, port, pitch, axis, distances, mesh_volume, float(mesh.area),
                            initial_bore_area=bore_area)
    helix_dimensions = None
    if IS_HELICAL_SWEEP:
        # For a helix r(theta) = (r cos theta, r sin theta, P theta / 2pi),
        # its osculating radius is r * (1 + (P / (2 pi r))^2).
        radius_of_curvature = HELIX_CENTERLINE_RADIUS_MM * (
            1.0 + (HELIX_PITCH_MM / (2.0 * np.pi * HELIX_CENTERLINE_RADIUS_MM)) ** 2
        )
        helix_dimensions = {
            "pitch_mm": HELIX_PITCH_MM,
            "centerline_radius_mm": HELIX_CENTERLINE_RADIUS_MM,
            "radius_of_curvature_mm": radius_of_curvature,
        }
        for row in rows:
            row.update({"is_helical_sweep": True, **helix_dimensions})
        print(f"Helical sweep: P={HELIX_PITCH_MM:.3f} mm; centerline r={HELIX_CENTERLINE_RADIUS_MM:.3f} mm; "
              f"radius of curvature R_c={radius_of_curvature:.3f} mm")
    write_csv(rows, args.output)
    plane_rows: list[dict[str, float]] | None = None
    if SAVE_SELECTED_PLANE_ANALYSIS and not args.no_plane_analysis:
        graph_distances = np.linspace(0.0, max_regression_mm, SELECTED_PLANE_GRAPH_POINTS)
        plane_rows = picked_plane_regression(solid, port, pitch, axis, graph_distances)
        if helix_dimensions is not None:
            for row in plane_rows:
                row.update({"is_helical_sweep": True, **helix_dimensions})
    save_visualization(solid, port, pitch, axis, distances, rows, args.plot,
                       REGRESSION_RATE_MM_PER_S, display_unit, display_scale, args.show)

    print(f"Mesh: {args.obj_file}")
    print(f"OBJ scale: 1 unit = {scale_to_mm:g} mm")
    print(f"Regression rate: {REGRESSION_RATE_MM_PER_S:g} mm/s; simulated time: {SIMULATION_TIME_S:g} s")
    print(f"Axis: {axis}; voxel pitches (X/Y/Z): {pitch}; grid: {tuple(solid.shape)}")
    print(f"Results written to: {args.output}")
    for name, value in rows[0].items():
        print(f"Initial {name}: {value:.6g}")
    if OPEN_INTERACTIVE_VIEWER and not args.no_interactive:
        show_interactive_viewer(solid, port, pitch, distances, rows, axis,
                                REGRESSION_RATE_MM_PER_S, display_unit, display_scale, helix_dimensions)
    if plane_rows is not None:
        save_picked_plane_analysis(
            plane_rows, axis,
            args.plane_plot if SAVE_SELECTED_PLANE_ANALYSIS_PLOT else None,
            display_unit, display_scale, args.plane_output, args.curve_fits_output,
            show=SHOW_SELECTED_PLANE_ANALYSIS_AFTER_VIEWER,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
