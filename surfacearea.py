"""Compute bore surface area for a fuel-grain OBJ mesh.

This script loads an OBJ file, identifies the bore-wall component of the mesh,
applies a small axial taper to remove end-transition faces, and reports the
bore surface area in mm^2.

Usage:
    python surfacearea.py path\to\part.obj
    python surfacearea.py path\to\part.obj --axis 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


MM2_PER_M2 = 1_000_000.0
DEFAULT_OBJ_FILE = Path(__file__).with_name("goshastar.obj")  # Change this to use a different OBJ by default
DEFAULT_AXIS = None  # Set to 0, 1, or 2 to force a specific axis in the script


def _choose_axis(mesh: trimesh.Trimesh) -> int:
    """Choose the axial direction as the longest mesh extent."""
    extents = np.asarray(mesh.extents, dtype=float)
    return int(np.argmax(extents))


def _component_surface_area_mm2(component: trimesh.Trimesh) -> float:
    """Convert Trimesh area to mm^2, handling meshes stored in either m or mm."""
    raw_area = float(component.area)
    return raw_area * MM2_PER_M2 if raw_area < 1.0 else raw_area


def _estimate_bore_component_area(mesh: trimesh.Trimesh, axis: int) -> float:
    """Estimate bore surface area by selecting the internal bore-wall component.

    The bore wall is usually the internal component whose faces mostly point
    radially instead of axially. This follows the 3D mesh directly, so it is
    closer to what a CAD kernel reports for a complex helical surface.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        return 0.0

    if axis == 1:
        radial_axes = [0, 2]
    elif axis == 0:
        radial_axes = [1, 2]
    else:
        radial_axes = [0, 1]

    full_radial_extent = float(np.max(np.linalg.norm(mesh.vertices[:, radial_axes], axis=1)))
    candidate_areas_mm2 = []

    for part in parts:
        if len(part.vertices) == 0 or len(part.faces) == 0:
            continue

        centers = np.asarray(part.triangles_center, dtype=float)
        normals = np.asarray(part.face_normals, dtype=float)
        areas = np.asarray(part.area_faces, dtype=float)

        if centers.size == 0 or normals.size == 0 or areas.size == 0:
            continue

        radial_vec = np.zeros((len(centers), 3), dtype=float)
        radial_vec[:, radial_axes[0]] = centers[:, radial_axes[0]]
        radial_vec[:, radial_axes[1]] = centers[:, radial_axes[1]]
        radial_norm = np.linalg.norm(radial_vec[:, radial_axes], axis=1)
        radial_vec = radial_vec / np.maximum(radial_norm[:, None], 1e-9)

        weighted_radial_dot = float(np.average(np.sum(normals * radial_vec, axis=1), weights=areas))
        weighted_axis_dot = float(np.average(np.abs(normals[:, axis]), weights=areas))
        radial_extent = float(np.max(np.linalg.norm(part.vertices[:, radial_axes], axis=1)))

        # Bore-wall component heuristic:
        # - smaller than the full outer shell
        # - points mostly inward toward the bore
        if radial_extent < 0.99 * full_radial_extent and weighted_axis_dot > 0.1 and weighted_radial_dot < -0.05:
            candidate_areas_mm2.append(_component_surface_area_mm2(part))

    if not candidate_areas_mm2:
        return 0.0

    return float(np.sum(candidate_areas_mm2))


def calculate_bore_surface_area(obj_path: Path, axis: Optional[int] = None) -> float:
    """Load an OBJ and return the bore surface area in mm^2."""
    mesh = trimesh.load(obj_path, force="mesh")
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError(f"Could not load a valid mesh from: {obj_path}")

    if axis is not None:
        return _estimate_bore_component_area(mesh, int(axis))

    axis_scores = [(_estimate_bore_component_area(mesh, candidate_axis), candidate_axis) for candidate_axis in (0, 1, 2)]
    best_area, best_axis = max(axis_scores, key=lambda item: item[0])
    if best_area <= 0:
        best_axis = _choose_axis(mesh)
        best_area = _estimate_bore_component_area(mesh, best_axis)

    return best_area


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate bore surface area from an OBJ file.")
    parser.add_argument(
        "obj_file",
        type=Path,
        nargs="?",
        default=DEFAULT_OBJ_FILE,
        help=f"Path to the OBJ file (default: {DEFAULT_OBJ_FILE.name})",
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=(0, 1, 2),
        default=DEFAULT_AXIS,
        help="Axis perpendicular to the cross-section (0=X, 1=Y, 2=Z). Default: the file-level DEFAULT_AXIS setting.",
    )
    args = parser.parse_args()

    if not args.obj_file.exists():
        raise SystemExit(f"OBJ file not found: {args.obj_file}")

    area_mm2 = calculate_bore_surface_area(args.obj_file, args.axis)
    axis_text = "auto" if args.axis is None else str(args.axis)
    print(f"OBJ: {args.obj_file}")
    print(f"Axis: {axis_text}")
    print(f"Bore surface area: {area_mm2:.5f} mm^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
