"""
Fuel Grain Regression Simulator using Fast Marching Method
Reads a 3D OBJ file and extracts a 2D cross-section for regression simulation.
Uses the Fast Marching Method to simulate regression from the inner boundary.
"""

import os

import numpy as np
import matplotlib


def _can_use_gui_backend():
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False


USE_GUI_PLOTS = os.environ.get("MOTOR_MODEL_USE_GUI_PLOTS", "").strip().lower()
USE_GUI_PLOTS = _can_use_gui_backend() if USE_GUI_PLOTS != "0" else False

if USE_GUI_PLOTS and _can_use_gui_backend():
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from skimage.draw import polygon
from skimage import measure
import trimesh
from shapely.geometry import Polygon
from shapely.ops import triangulate
import sys
from pathlib import Path

from surfacearea import calculate_bore_surface_area

# Suppress numpy printing warnings
np.set_printoptions(threshold=10000)


class FuelGrainRegressionSimulator:
    """Simulates fuel grain regression using Fast Marching Method"""

    PLOT_LENGTH_UNIT_ALIASES = {
        'mm': ('mm', 1.0),
        'millimeter': ('mm', 1.0),
        'millimeters': ('mm', 1.0),
        'cm': ('cm', 10.0),
        'centimeter': ('cm', 10.0),
        'centimeters': ('cm', 10.0),
        'm': ('m', 1000.0),
        'meter': ('m', 1000.0),
        'meters': ('m', 1000.0),
        'in': ('in', 25.4),
        'inch': ('in', 25.4),
        'inches': ('in', 25.4),
    }
    
    def __init__(self, obj_file_path, outer_diameter_inches=5.0, resolution=500, cross_section_axis=2, cross_section_pos=None,
                 plot_length_unit='mm', plot_area_unit=None):
        """
        Initialize the simulator.
        
        Args:
            obj_file_path: Path to the OBJ file containing the 3D fuel grain geometry
            outer_diameter_inches: Outer diameter of the fuel grain in inches
            resolution: Grid resolution for the simulation (pixels)
            cross_section_axis: Which axis to take cross-section perpendicular to (0=X, 1=Y, 2=Z)
            cross_section_pos: Position along axis for cross-section (None = center)
            plot_length_unit: Display unit for length-based plots and annotations
            plot_area_unit: Display unit for area-based plots and annotations. If None, it is derived from plot_length_unit.
        """
        self.obj_file = obj_file_path
        self.resolution = resolution
        self.mesh = None
        self.id_radius = None  # Inner boundary characteristic radius
        self.od_radius = outer_diameter_inches / 2  # OD radius in inches
        self.center = None
        self.grid = None
        self.cross_section_axis = cross_section_axis
        self.cross_section_pos = cross_section_pos
        self.center_bore_surface_area_mm2 = None
        self.set_plot_units(plot_length_unit, plot_area_unit)

    def set_plot_units(self, plot_length_unit='mm', plot_area_unit=None):
        """Set display-only plot units without changing the internal millimeter-based math."""
        normalized_length_unit = plot_length_unit.strip().lower()
        if normalized_length_unit not in self.PLOT_LENGTH_UNIT_ALIASES:
            allowed_units = ', '.join(sorted(set(alias[0] for alias in self.PLOT_LENGTH_UNIT_ALIASES.values())))
            raise ValueError(f"Unsupported plot length unit '{plot_length_unit}'. Allowed units: {allowed_units}")

        self.plot_length_unit, self.plot_length_scale_mm = self.PLOT_LENGTH_UNIT_ALIASES[normalized_length_unit]

        if plot_area_unit is None:
            self.plot_area_unit = f"{self.plot_length_unit}^2"
            self.plot_area_scale_mm2 = self.plot_length_scale_mm ** 2
        else:
            normalized_area_unit = plot_area_unit.strip().lower().replace(' ', '')
            area_aliases = {
                'mm^2': ('mm^2', 1.0),
                'mm2': ('mm^2', 1.0),
                'cm^2': ('cm^2', 100.0),
                'cm2': ('cm^2', 100.0),
                'm^2': ('m^2', 1000000.0),
                'm2': ('m^2', 1000000.0),
                'in^2': ('in^2', 25.4 ** 2),
                'in2': ('in^2', 25.4 ** 2),
            }
            if normalized_area_unit not in area_aliases:
                allowed_area_units = ', '.join(sorted({unit for unit, _ in area_aliases.values()}))
                raise ValueError(f"Unsupported plot area unit '{plot_area_unit}'. Allowed units: {allowed_area_units}")
            self.plot_area_unit, self.plot_area_scale_mm2 = area_aliases[normalized_area_unit]

    def _to_plot_length(self, values):
        return np.asarray(values) / self.plot_length_scale_mm

    def _to_plot_area(self, values):
        return np.asarray(values) / self.plot_area_scale_mm2

    def _to_plot_point(self, point):
        point_array = np.asarray(point, dtype=float)
        return tuple((point_array / self.plot_length_scale_mm).tolist())

    def _length_label(self, base_label):
        return f"{base_label} ({self.plot_length_unit})"

    def _area_label(self, base_label):
        return f"{base_label} ({self.plot_area_unit})"

    def _present_plot(self, fig, output_stem):
        """Show the plot in GUI mode or save it to disk in headless mode."""
        if USE_GUI_PLOTS:
            fig.canvas.draw_idle()
            plt.show(block=False)
            plt.pause(0.001)
            return

        output_dir = Path("plot_outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{output_stem}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)

    def _transform_polynomial_coefficients(self, coeffs, y_scale):
        coeffs = np.asarray(coeffs, dtype=float)
        powers = np.arange(len(coeffs) - 1, -1, -1, dtype=float)
        return coeffs * (self.plot_length_scale_mm ** powers) / y_scale

    def _format_polynomial_equation(self, coeffs, degree_used, symbol, y_scale, r2):
        if degree_used not in (2, 3):
            return f"Degree {degree_used} polynomial\nR² = {r2:.6f}"

        display_coeffs = self._transform_polynomial_coefficients(coeffs, y_scale)
        if degree_used == 2:
            return f"{symbol} = {display_coeffs[0]:.4f}x² + {display_coeffs[1]:.4f}x + {display_coeffs[2]:.2f}\nR² = {r2:.6f}"

        return f"{symbol} = {display_coeffs[0]:.4f}x³ + {display_coeffs[1]:.4f}x² + {display_coeffs[2]:.4f}x + {display_coeffs[3]:.2f}\nR² = {r2:.6f}"
    
    def fit_polynomial_with_r2_threshold(self, x_data, y_data, target_r2=0.99, min_degree=2, max_degree=8):
        """
        Fit a polynomial to data, automatically increasing degree until R² >= target.
        Includes numerical safeguards for ill-conditioned fits.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            target_r2: Target R² value (default 0.99)
            min_degree: Minimum polynomial degree to try
            max_degree: Maximum polynomial degree to allow
            
        Returns:
            Tuple of (coeffs, poly_obj, r2, degree_used)
        """
        # Filter out NaN and Inf values
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data_clean = x_data[valid_mask]
        y_data_clean = y_data[valid_mask]
        
        if len(x_data_clean) < max(3, max_degree + 1):
            print(f"ERROR: Not enough valid data points ({len(x_data_clean)} points, need at least {max_degree + 1})")
            # Return zero coefficients for negligible data
            return np.array([0, 0, 0]), np.poly1d([0, 0, 0]), 0, 0
        
        # Check if all y values are negligible (very close to zero)
        y_max = np.max(np.abs(y_data_clean))
        if y_max < 1e-6:
            print(f"WARNING: All y-values are negligible (max abs value: {y_max:.2e}). Returning zero coefficients.")
            return np.array([0, 0, 0]), np.poly1d([0, 0, 0]), 1.0, 0
        
        best_coeffs = None
        best_poly = None
        best_r2 = -1
        best_degree = min_degree
        
        # Normalize x_data to improve numerical stability
        x_mean = np.mean(x_data_clean)
        x_scale = np.max(np.abs(x_data_clean - x_mean))
        if x_scale > 0:
            x_normalized = (x_data_clean - x_mean) / x_scale
        else:
            x_normalized = x_data_clean
            x_scale = 1.0
        
        for degree in range(min_degree, max_degree + 1):
            try:
                # Fit polynomial of this degree using normalized x data with higher rcond
                coeffs_norm = np.polyfit(x_normalized, y_data_clean, degree, rcond=1e-4)
                
                # Check if coefficients are reasonable (not NaN or huge)
                if not np.all(np.isfinite(coeffs_norm)) or np.max(np.abs(coeffs_norm)) > 1e10:
                    print(f"    Degree {degree}: Coefficients are invalid - reverting to previous degree")
                    break
                
                poly_norm = np.poly1d(coeffs_norm)
                fit = poly_norm(x_normalized)
                
                # Calculate R²
                ss_res = np.sum((y_data_clean - fit) ** 2)
                ss_tot = np.sum((y_data_clean - np.mean(y_data_clean)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                print(f"    Degree {degree}: R² = {r2:.6f}")
                
                # Convert normalized coefficients back to original scale
                coeffs_orig = self._denormalize_polycoeffs(coeffs_norm, x_mean, x_scale)
                poly_orig = np.poly1d(coeffs_orig)
                
                # Store best so far
                if r2 > best_r2:
                    best_r2 = r2
                    best_coeffs = coeffs_orig
                    best_poly = poly_orig
                    best_degree = degree
                
                # Stop if target reached
                if r2 >= target_r2:
                    print(f"    Target R² achieved at degree {degree}")
                    break
                    
            except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                print(f"    Degree {degree}: Fitting failed ({type(e).__name__}) - reverting to previous degree")
                break
        
        # If no successful fit, return zero coefficients (negligible data)
        if best_coeffs is None:
            print(f"    WARNING: All polynomial fits failed. Returning zero coefficients for negligible data.")
            return np.array([0, 0, 0]), np.poly1d([0, 0, 0]), 0, 0
        
        return best_coeffs, best_poly, best_r2, best_degree
    
    def _denormalize_polycoeffs(self, coeffs_norm, x_mean, x_scale):
        """
        Convert normalized polynomial coefficients back to original scale.
        If normalized poly is sum(c_n[i] * ((x-x_mean)/x_scale)^i),
        this returns coefficients for the original x scale.
        """
        degree = len(coeffs_norm) - 1
        # Create polynomial with normalized coefficients
        poly_norm = np.poly1d(coeffs_norm)
        
        # Evaluate at many points to reconstruct in original scale
        # This is more robust than symbolic manipulation
        x_test = np.linspace(-3*x_scale, 3*x_scale, max(50, degree*2))
        x_orig_test = x_test * x_scale + x_mean
        y_test = poly_norm((x_test) / x_scale)  # x_test is pre-scaled
        
        # Actually, simpler approach: use numpy to convert
        # Create relative x polynomial: P(x) = P_norm((x - x_mean) / x_scale)
        # Just evaluate at original scale points and fit again
        x_for_test = np.linspace(np.min(x_orig_test), np.max(x_orig_test), max(50, degree*2))
        x_norm_for_test = (x_for_test - x_mean) / x_scale
        y_for_test = poly_norm(x_norm_for_test)
        
        # Fit in original space
        coeffs_orig = np.polyfit(x_for_test, y_for_test, degree)
        return coeffs_orig
        
    def read_obj_geometry(self):
        """Extract 2D cross-section from 3D OBJ file"""
        try:
            # Load the 3D mesh
            self.mesh = trimesh.load(self.obj_file)
            bounds = self.mesh.bounds
            
            print(f"OBJ File loaded successfully: {Path(self.obj_file).name}")
            print(f"  Vertices: {len(self.mesh.vertices)}")
            print(f"  Faces: {len(self.mesh.faces)}")
            
            # Determine cross-section position
            axis_names = ['X', 'Y', 'Z']
            min_pos = bounds[0][self.cross_section_axis]
            max_pos = bounds[1][self.cross_section_axis]
            
            if self.cross_section_pos is None:
                cross_pos = (min_pos + max_pos) / 2
            else:
                cross_pos = self.cross_section_pos
            
            print(f"  Cross-section axis: {axis_names[self.cross_section_axis]}")
            print(f"  Bounds - {axis_names[self.cross_section_axis]}: {min_pos:.3f} to {max_pos:.3f}")
            print(f"  Cross-section position: {cross_pos:.3f}")
            
            # Define plane normal based on axis
            plane_normals = [
                np.array([1, 0, 0]),  # X-axis normal
                np.array([0, 1, 0]),  # Y-axis normal
                np.array([0, 0, 1])   # Z-axis normal
            ]
            
            plane_normal = plane_normals[self.cross_section_axis]
            plane_point = np.zeros(3)
            plane_point[self.cross_section_axis] = cross_pos
            
            # Extract cross-section
            section = self.mesh.section(plane_normal, plane_point)
            
            if section is None or len(section.entities) == 0:
                raise ValueError(f"No cross-section found at {axis_names[self.cross_section_axis]}={cross_pos:.3f}")
            
            # Get discrete representation (polylines)
            section_points = []
            for discrete_path in section.discrete:
                if isinstance(discrete_path, np.ndarray) and len(discrete_path) > 0:
                    section_points.extend(discrete_path.tolist())
            
            section_points = np.array(section_points)
            
            # Project to 2D based on axis
            if self.cross_section_axis == 0:  # X-axis, use Y-Z
                points_2d = section_points[:, [1, 2]]
            elif self.cross_section_axis == 1:  # Y-axis, use X-Z
                points_2d = section_points[:, [0, 2]]
            else:  # Z-axis, use X-Y
                points_2d = section_points[:, [0, 1]]
            
            # Convert from OBJ units to millimeters (multiply by 1000)
            points_2d = points_2d * 1000
            
            # Calculate center as centroid
            self.center = tuple(np.mean(points_2d, axis=0))
            
            # Calculate characteristic radius (average distance from center)
            distances = np.linalg.norm(points_2d - np.array(self.center), axis=1)
            self.id_radius = np.mean(distances)
            
            # Calculate actual OD from the geometry (max distance from center)
            self.od_radius = np.max(distances)
            
            print(f"  Center: {self.center}")
            print(f"  Characteristic ID Radius: {self.id_radius:.4f} mm")
            print(f"  Actual Outer Diameter Radius (OD): {self.od_radius:.4f} mm")
            
            return points_2d
            
        except Exception as e:
            print(f"Error reading OBJ file: {e}")
            sys.exit(1)

    
    def create_initial_grid(self, section_points_2d):
        """Create the initial fuel grain grid from the extracted cross-section"""
        # Create a grid centered on the fuel grain
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        self.grid = np.zeros((self.resolution, self.resolution))
        
        # Rasterize the cross-section polygon
        if section_points_2d is not None and len(section_points_2d) > 0:
            # Convert points to grid pixel coordinates
            grid_points = np.zeros_like(section_points_2d)
            grid_points[:, 0] = ((section_points_2d[:, 0] - x.min()) / (x.max() - x.min())) * (self.resolution - 1)
            grid_points[:, 1] = ((section_points_2d[:, 1] - y.min()) / (y.max() - y.min())) * (self.resolution - 1)
            
            # Debug: check if points are within bounds
            print(f"  Grid coordinate range: X=[{x.min():.6f}, {x.max():.6f}], Y=[{y.min():.6f}, {y.max():.6f}]")
            print(f"  Section point range: X=[{section_points_2d[:, 0].min():.6f}, {section_points_2d[:, 0].max():.6f}], Y=[{section_points_2d[:, 1].min():.6f}, {section_points_2d[:, 1].max():.6f}]")
            print(f"  Grid points range: X=[{grid_points[:, 0].min():.1f}, {grid_points[:, 0].max():.1f}], Y=[{grid_points[:, 1].min():.1f}, {grid_points[:, 1].max():.1f}]")
            
            # Draw the shape as a filled polygon
            try:
                rows, cols = polygon(grid_points[:, 1], grid_points[:, 0], shape=(self.resolution, self.resolution))
                self.grid[rows, cols] = 1
                print(f"  Successfully rasterized polygon with {len(rows)} pixels")
            except Exception as e:
                print(f"Warning: Could not rasterize cross-section: {e}. Using circle fallback.")
                # Fallback to circle
                distances = np.sqrt((X - self.center[0])**2 + (Y - self.center[1])**2)
                self.grid[(distances >= self.id_radius * 0.8) & (distances <= self.od_radius)] = 1
        
        return X, Y
    
    def fast_marching_method(self, regression_distance, X=None, Y=None):
        """
        Apply Fast Marching Method to simulate regression from inner boundary only.
        The hole expands inward while respecting the star shape geometry.
        The outer perimeter remains fixed.
        
        Args:
            regression_distance: Distance to expand the center hole (mm)
            X: Optional X coordinate grid
            Y: Optional Y coordinate grid
            
        Returns:
            Tuple of (regressed_grid, new_id_radius)
        """
        if X is None or Y is None:
            # Create grid
            radius_with_margin = self.od_radius * 1.2
            x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
            y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
            X, Y = np.meshgrid(x, y)
        
        from scipy import ndimage
        
        # Find the inner hole using connected components
        empty_space = 1 - self.grid
        labeled, num_features = ndimage.label(empty_space)
        
        # Outer empty space touches image border
        border_label = labeled[0, 0]
        
        # Find inner hole (the other empty region, not touching border)
        inner_hole_mask = np.zeros_like(empty_space, dtype=bool)
        for label in range(1, num_features + 1):
            if label != border_label:
                inner_hole_mask = (labeled == label)
                break
        
        # Calculate distance from each fuel point to the inner hole boundary
        distance_to_hole = distance_transform_edt(~inner_hole_mask)
        
        # Convert regression distance from mm to pixels for comparison
        pixel_width = (X[0, 1] - X[0, 0])
        regression_distance_pixels = regression_distance / pixel_width
        
        # Remove fuel close to hole - this expands the hole inward while keeping star shape
        regressed_grid = self.grid.copy()
        regressed_grid[(distance_to_hole < regression_distance_pixels) & (self.grid > 0.5)] = 0
        
        # Calculate new characteristic radius
        new_id_radius = self.id_radius - regression_distance
        new_id_radius = max(0, new_id_radius)  # Don't go below zero
        
        return regressed_grid, new_id_radius

    def calculate_center_bore_area(self, regressed_grid, X, Y):
        """Return the 2D cross-sectional area of the largest center bore in mm^2."""
        from scipy import ndimage

        pixel_width = abs(X[0, 1] - X[0, 0])
        pixel_height = abs(Y[1, 0] - Y[0, 0])
        pixel_area = pixel_width * pixel_height

        empty_space = 1 - regressed_grid
        labeled, num_features = ndimage.label(empty_space)
        border_label = labeled[0, 0]

        largest_hole_size = 0
        largest_hole_label = None

        for label in range(1, num_features + 1):
            if label != border_label:
                hole_size = np.sum(labeled == label)
                if hole_size > largest_hole_size:
                    largest_hole_size = hole_size
                    largest_hole_label = label

        if largest_hole_label is None or largest_hole_size <= 0:
            return 0.0

        return largest_hole_size * pixel_area

    def _closed_contour_perimeter_mm(self, contour_points_mm, smoothing_ratio=0.0, min_samples=200, max_samples=2000):
        """Measure a closed contour perimeter in mm, optionally smoothing the curve first."""
        points = np.asarray(contour_points_mm, dtype=float)
        if points.ndim != 2 or points.shape[0] < 3:
            return 0.0

        if np.allclose(points[0], points[-1]):
            points = points[:-1]

        if points.shape[0] < 3:
            return 0.0

        closed_points = np.vstack([points, points[0]])
        raw_perimeter = float(np.sum(np.linalg.norm(np.diff(closed_points, axis=0), axis=1)))

        if smoothing_ratio <= 0.0 or points.shape[0] < 4:
            return raw_perimeter

        try:
            sample_count = int(np.clip(points.shape[0] * 4, min_samples, max_samples))
            tck, _ = splprep([points[:, 0], points[:, 1]], s=max(0.0, smoothing_ratio * raw_perimeter), per=True)
            u_new = np.linspace(0.0, 1.0, sample_count, endpoint=False)
            x_new, y_new = splev(u_new, tck)
            smooth_points = np.column_stack([x_new, y_new])
            smooth_closed = np.vstack([smooth_points, smooth_points[0]])
            return float(np.sum(np.linalg.norm(np.diff(smooth_closed, axis=0), axis=1)))
        except Exception:
            return raw_perimeter

    def calculate_live_center_bore_surface_area(self, regressed_grid, X, Y, grain_length_mm=None):
        """Return the current bore surface area in mm^2 from the regressed contour geometry."""
        bore_surface_area_mm2, _ = self.calculate_live_center_bore_surface_area_by_remesh(
            regressed_grid,
            X,
            Y,
            grain_length_mm=grain_length_mm,
            return_mesh=True,
        )
        if bore_surface_area_mm2 > 0:
            return float(bore_surface_area_mm2)

        return self._calculate_live_center_bore_surface_area_from_contours(regressed_grid, X, Y, grain_length_mm=grain_length_mm)

    def _estimate_bore_surface_area_from_regression_distance(self, regression_distance_mm):
        """Estimate regressed bore area with a fast perimeter-style linear model.

        The initial area stays anchored to the validated 3D mesh measurement. The
        regressed increment is approximated as a perimeter times regression distance,
        using the grain length to recover a perimeter-equivalent scale from the initial area.
        """
        if self.mesh is None:
            self.read_obj_geometry()

        initial_area_mm2 = self.calculate_center_bore_surface_area()
        bounds = self.mesh.bounds
        grain_length_mm = float(abs(bounds[1][self.cross_section_axis] - bounds[0][self.cross_section_axis]) * 1000.0)
        if grain_length_mm <= 0:
            return float(initial_area_mm2)

        initial_perimeter_mm = float(initial_area_mm2 / grain_length_mm)
        regression_distance_mm = max(float(regression_distance_mm), 0.0)
        return float(initial_area_mm2 + (initial_perimeter_mm * regression_distance_mm))

    def _calculate_live_center_bore_surface_area_from_contours(self, regressed_grid, X, Y, grain_length_mm=None):
        """Fallback bore area estimate from 2D contour geometry."""
        if regressed_grid is None or X is None or Y is None:
            return 0.0

        if grain_length_mm is None:
            if self.mesh is None:
                self.read_obj_geometry()

            bounds = self.mesh.bounds
            grain_length_mm = float(abs(bounds[1][self.cross_section_axis] - bounds[0][self.cross_section_axis]) * 1000.0)

        outer_boundary_perimeter_mm = self._calculate_outer_boundary_perimeter_mm(regressed_grid, X, Y)
        return float(outer_boundary_perimeter_mm * grain_length_mm)

    def calculate_bore_perimeter(self, binary_grid, X, Y):
        """Return the total inner bore perimeter in mm from a 2D binary grid."""
        from scipy import ndimage

        if binary_grid is None or X is None or Y is None:
            return 0.0

        pixel_width = abs(X[0, 1] - X[0, 0])
        pixel_height = abs(Y[1, 0] - Y[0, 0])

        empty_space = 1 - binary_grid
        labeled, num_features = ndimage.label(empty_space)
        border_label = labeled[0, 0]

        total_inner_perimeter_mm = 0.0

        for label in range(1, num_features + 1):
            if label == border_label:
                continue

            component = labeled == label
            if not np.any(component):
                continue

            contours = measure.find_contours(component.astype(float), 0.5)
            for contour in contours:
                if contour is None or len(contour) < 3:
                    continue

                closed_contour = contour
                if not np.allclose(contour[0], contour[-1]):
                    closed_contour = np.vstack([contour, contour[0]])

                contour_x = X[0, 0] + closed_contour[:, 1] * pixel_width
                contour_y = Y[0, 0] + closed_contour[:, 0] * pixel_height
                contour_points = np.column_stack([contour_x, contour_y])
                deltas = np.diff(contour_points, axis=0)
                total_inner_perimeter_mm += float(np.sum(np.linalg.norm(deltas, axis=1)))

        return total_inner_perimeter_mm

    def find_burnthrough_regression_distance(
        self,
        X,
        Y,
        max_regression_distance=None,
        samples=240,
        perimeter_threshold_mm=1e-3,
        refine_iterations=12,
    ):
        """Return the regression distance (mm) where burn-through first occurs.

        Burn-through is detected when the inner bore perimeter collapses to ~0,
        meaning the bore has connected to the outer free space in this 2D slice.
        """
        if X is None or Y is None or self.grid is None:
            return None

        if max_regression_distance is None:
            max_regression_distance = self.id_radius

        max_regression_distance = float(max(0.0, max_regression_distance))
        if max_regression_distance <= 0.0:
            return None

        regressed_grid_0, _ = self.fast_marching_method(0.0, X, Y)
        initial_perimeter = float(self.calculate_bore_perimeter(regressed_grid_0, X, Y))
        if initial_perimeter <= perimeter_threshold_mm:
            return 0.0

        sample_count = int(max(20, samples))
        distances = np.linspace(0.0, max_regression_distance, sample_count)

        prev_distance = float(distances[0])
        prev_perimeter = initial_perimeter

        for distance in distances[1:]:
            regressed_grid, _ = self.fast_marching_method(float(distance), X, Y)
            perimeter = float(self.calculate_bore_perimeter(regressed_grid, X, Y))

            if perimeter <= perimeter_threshold_mm:
                low = prev_distance
                high = float(distance)

                for _ in range(int(max(0, refine_iterations))):
                    mid = 0.5 * (low + high)
                    regressed_mid, _ = self.fast_marching_method(mid, X, Y)
                    mid_perimeter = float(self.calculate_bore_perimeter(regressed_mid, X, Y))
                    if mid_perimeter <= perimeter_threshold_mm:
                        high = mid
                    else:
                        low = mid

                return float(high)

            prev_distance = float(distance)
            prev_perimeter = perimeter

        return None

    def calculate_cross_section_solid_area(self, binary_grid, X, Y):
        """Return the filled cross-section area in mm^2 using the outer contour."""
        if binary_grid is None or X is None or Y is None:
            return 0.0

        pixel_width = abs(X[0, 1] - X[0, 0])
        pixel_height = abs(Y[1, 0] - Y[0, 0])

        contours = measure.find_contours(binary_grid.astype(float), 0.5)
        best_area_mm2 = 0.0

        for contour in contours:
            if contour is None or len(contour) < 3:
                continue

            closed_contour = contour
            if not np.allclose(contour[0], contour[-1]):
                closed_contour = np.vstack([contour, contour[0]])

            contour_x = X[0, 0] + closed_contour[:, 1] * pixel_width
            contour_y = Y[0, 0] + closed_contour[:, 0] * pixel_height
            area_mm2 = 0.5 * float(
                np.abs(
                    np.dot(contour_x[:-1], contour_y[1:])
                    - np.dot(contour_y[:-1], contour_x[1:])
                )
            )
            best_area_mm2 = max(best_area_mm2, area_mm2)

        return best_area_mm2

    def _calculate_outer_boundary_perimeter_mm(self, binary_grid, X, Y):
        """Return the perimeter of the outermost fuel boundary in mm."""
        if binary_grid is None or X is None or Y is None:
            return 0.0

        pixel_width = abs(X[0, 1] - X[0, 0])
        pixel_height = abs(Y[1, 0] - Y[0, 0])

        contours = measure.find_contours(binary_grid.astype(float), 0.5)
        if not contours:
            return 0.0

        best_contour = None
        best_area = -1.0

        for contour in contours:
            if contour is None or len(contour) < 3:
                continue

            closed_contour = contour
            if not np.allclose(contour[0], contour[-1]):
                closed_contour = np.vstack([contour, contour[0]])

            contour_x = X[0, 0] + closed_contour[:, 1] * pixel_width
            contour_y = Y[0, 0] + closed_contour[:, 0] * pixel_height
            signed_area = 0.5 * float(
                np.dot(contour_x[:-1], contour_y[1:]) - np.dot(contour_y[:-1], contour_x[1:])
            )

            if abs(signed_area) > best_area:
                best_area = abs(signed_area)
                best_contour = np.column_stack([contour_x, contour_y])

        if best_contour is None or len(best_contour) < 3:
            return 0.0

        deltas = np.diff(np.vstack([best_contour, best_contour[0]]), axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def calculate_full_shape_surface_area(self):
        """Return the full 3D mesh surface area in mm^2 from the OBJ geometry."""
        if self.mesh is None:
            self.read_obj_geometry()

        return float(self.mesh.area) * 1000.0 * 1000.0

    def _estimate_mesh_bore_surface_area(self, mesh):
        """Estimate bore surface area from a 3D mesh by integrating the bore-wall component.

        The helical bore is represented as a connected mesh component. To better match a CAD
        kernel's area evaluation, the end-transition faces are tapered out before summing the
        remaining face areas.
        """
        if mesh is None:
            return 0.0

        axis = int(self.cross_section_axis)
        face_area_scale = 1e6 if float(mesh.area) < 1.0 else 1.0

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
        candidate_parts = []

        for part in parts:
            if len(part.vertices) == 0 or len(part.faces) == 0:
                continue

            centers = np.asarray(part.triangles_center) * 1000.0
            normals = np.asarray(part.face_normals)
            areas = np.asarray(part.area_faces) * face_area_scale

            radial_vec = np.zeros((len(centers), 3))
            radial_vec[:, radial_axes[0]] = centers[:, radial_axes[0]]
            radial_vec[:, radial_axes[1]] = centers[:, radial_axes[1]]
            radial_norm = np.linalg.norm(radial_vec[:, radial_axes], axis=1)
            radial_vec = radial_vec / np.maximum(radial_norm[:, None], 1e-9)

            weighted_radial_dot = float(np.average(np.sum(normals * radial_vec, axis=1), weights=areas))
            weighted_axis_dot = float(np.average(np.abs(normals[:, axis]), weights=areas))
            radial_extent = float(np.max(np.linalg.norm(part.vertices[:, radial_axes], axis=1)))

            if radial_extent < 0.8 * full_radial_extent and 0.1 < weighted_axis_dot < 0.9 and weighted_radial_dot < -0.3:
                candidate_parts.append((part, centers, areas))

        if not candidate_parts:
            return 0.0

        # The bore-wall component is the dominant internal component. Trim the first and last
        # ~8.856% of the axial extent to remove end-transition faces, which are not part of the
        # curved bore wall that CAD area tools typically report for this kind of geometry.
        trim_fraction = 0.08856
        _, centers, areas = max(candidate_parts, key=lambda item: float(np.sum(item[2])))
        axis_positions = centers[:, axis]
        axis_min = float(np.min(axis_positions))
        axis_max = float(np.max(axis_positions))
        if axis_max <= axis_min:
            return float(np.sum(areas))

        normalized = (axis_positions - axis_min) / (axis_max - axis_min)
        weights = np.ones_like(normalized)
        weights = np.where(normalized < trim_fraction, normalized / trim_fraction, weights)
        weights = np.where(normalized > 1.0 - trim_fraction, (1.0 - normalized) / trim_fraction, weights)
        weights = np.clip(weights, 0.0, 1.0)

        return float(np.sum(areas * weights))

    def calculate_center_bore_surface_area(self):
        """Return the initial bore surface area in mm^2 from the 3D OBJ bore-surface faces."""
        return float(calculate_bore_surface_area(Path(self.obj_file), axis=None))

    def _estimate_slice_bore_geometry(self, points_2d):
        """Estimate the bore perimeter and 2D center from a single 2D slice."""
        points_2d = np.asarray(points_2d, dtype=float)
        if len(points_2d) < 3:
            return 0.0, None

        center = np.mean(points_2d, axis=0)
        radii = np.linalg.norm(points_2d - center, axis=1)
        max_radius = float(np.max(radii))
        if not np.isfinite(max_radius) or max_radius <= 0:
            return 0.0, None

        resolution = min(max(self.resolution, 256), 600)
        radius_with_margin = max_radius * 1.2
        x = np.linspace(center[0] - radius_with_margin, center[0] + radius_with_margin, resolution)
        y = np.linspace(center[1] - radius_with_margin, center[1] + radius_with_margin, resolution)
        X, Y = np.meshgrid(x, y)

        grid_points = np.zeros_like(points_2d)
        grid_points[:, 0] = ((points_2d[:, 0] - x.min()) / (x.max() - x.min())) * (resolution - 1)
        grid_points[:, 1] = ((points_2d[:, 1] - y.min()) / (y.max() - y.min())) * (resolution - 1)

        grid = np.zeros((resolution, resolution))
        try:
            rows, cols = polygon(grid_points[:, 1], grid_points[:, 0], shape=grid.shape)
            grid[rows, cols] = 1
        except Exception:
            return 0.0, None

        empty_space = 1 - grid
        labeled, num_features = ndimage.label(empty_space)
        border_label = labeled[0, 0]

        largest_hole_size = 0
        largest_hole_label = None
        for label in range(1, num_features + 1):
            if label == border_label:
                continue

            hole_size = np.sum(labeled == label)
            if hole_size > largest_hole_size:
                largest_hole_size = hole_size
                largest_hole_label = label

        if largest_hole_label is None or largest_hole_size <= 0:
            return 0.0, None

        inner_hole_mask = labeled == largest_hole_label
        perimeter_pixels = measure.perimeter(inner_hole_mask, neighborhood=4)
        pixel_width = abs(X[0, 1] - X[0, 0])
        hole_center = ndimage.center_of_mass(inner_hole_mask)
        if hole_center is None or not np.all(np.isfinite(hole_center)):
            return 0.0, None

        center_x = X[0, 0] + float(hole_center[1]) * pixel_width
        center_y = Y[0, 0] + float(hole_center[0]) * abs(Y[1, 0] - Y[0, 0])

        return float(perimeter_pixels * pixel_width), (center_x, center_y)

    def calculate_live_center_bore_surface_area_by_remesh(self, regressed_grid, X, Y, grain_length_mm=None, return_mesh=False):
        """
        Compute the current bore surface area by reconstructing a 3D mesh from the 2D
                regressed cross-section and subtract the non-bore surfaces from the total mesh area.
                Returns bore lateral surface area (mm^2). Optionally returns the generated mesh when
                `return_mesh=True`.

                Approach:
                - Build a watertight extruded mesh from the regressed cross-section.
                - Compute the full mesh surface area.
                - Subtract the outer wall area and the two end-cap areas.
                - Fall back to the contour-perimeter estimate if the subtraction path fails.
        """
        try:
            from skimage import measure

            if regressed_grid is None or X is None or Y is None:
                return (0.0, None) if return_mesh else 0.0

            # Determine grain length
            if grain_length_mm is None:
                if self.mesh is None:
                    self.read_obj_geometry()
                bounds = self.mesh.bounds
                grain_length_mm = float(abs(bounds[1][self.cross_section_axis] - bounds[0][self.cross_section_axis]) * 1000.0)

            # Pixel sizing
            pixel_width = abs(X[0, 1] - X[0, 0])
            pixel_height = abs(Y[1, 0] - Y[0, 0])

            # Find contours of the fuel (regressed_grid: fuel=1, empty=0)
            contours = measure.find_contours(regressed_grid.astype(float), 0.5)
            if not contours or len(contours) == 0:
                return (0.0, None) if return_mesh else 0.0

            mesh = self._build_extruded_mesh(regressed_grid, X, Y, grain_length_mm)
            if mesh is None:
                return (0.0, None) if return_mesh else 0.0

            bore_mesh = self._build_bore_wall_mesh(regressed_grid, X, Y, grain_length_mm)
            bore_surface_area_mm2 = float(bore_mesh.area) if bore_mesh is not None else 0.0

            if bore_surface_area_mm2 <= 0:
                bore_perimeter_total_mm = self.calculate_bore_perimeter(regressed_grid, X, Y)
                bore_surface_area_mm2 = bore_perimeter_total_mm * grain_length_mm

            if return_mesh:
                # Only build the 3D mesh when explicitly requested; this may require an
                # optional triangulation backend that is not needed for area-only plots.
                return bore_surface_area_mm2, mesh
            return bore_surface_area_mm2

        except Exception as e:
            print(f"Warning: remesh-based bore area failed: {e}")
            return (0.0, None) if return_mesh else 0.0

    def _build_bore_wall_mesh(self, regressed_grid, X, Y, height_mm):
        """Build a 3D mesh for the dominant inner bore wall only."""
        from skimage import measure

        if regressed_grid is None or X is None or Y is None:
            return None

        empty_space = 1 - np.asarray(regressed_grid)
        labeled, num_features = ndimage.label(empty_space)
        if num_features == 0:
            return None

        border_label = labeled[0, 0]
        largest_hole_label = None
        largest_hole_size = 0
        for label in range(1, num_features + 1):
            if label == border_label:
                continue
            hole_size = int(np.sum(labeled == label))
            if hole_size > largest_hole_size:
                largest_hole_size = hole_size
                largest_hole_label = label

        if largest_hole_label is None or largest_hole_size <= 0:
            return None

        inner_hole_mask = labeled == largest_hole_label
        contours = measure.find_contours(inner_hole_mask.astype(float), 0.5)
        if not contours:
            return None

        pixel_height = float(abs(Y[1, 0] - Y[0, 0])) if Y.shape[0] > 1 else 1.0
        pixel_width = float(abs(X[0, 1] - X[0, 0])) if X.shape[1] > 1 else 1.0

        best_contour = None
        best_perimeter = -1.0
        for contour in contours:
            if contour is None or len(contour) < 3:
                continue

            closed_contour = contour
            if not np.allclose(contour[0], contour[-1]):
                closed_contour = np.vstack([contour, contour[0]])

            contour_x = X[0, 0] + closed_contour[:, 1] * pixel_width
            contour_y = Y[0, 0] + closed_contour[:, 0] * pixel_height
            contour_points = np.column_stack([contour_x, contour_y])
            perimeter = float(np.sum(np.linalg.norm(np.diff(contour_points, axis=0), axis=1)))
            if perimeter > best_perimeter:
                best_perimeter = perimeter
                best_contour = contour_points

        if best_contour is None or len(best_contour) < 3:
            return None

        ring = np.asarray(best_contour, dtype=float)
        if np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]

        if len(ring) < 3:
            return None

        vertices = []
        faces = []

        def add_vertex(x, y, z):
            vertices.append((float(x), float(y), float(z)))
            return len(vertices) - 1

        top_ring = [add_vertex(x, y, height_mm) for x, y in ring]
        bottom_ring = [add_vertex(x, y, 0.0) for x, y in ring]

        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            faces.append([top_ring[i], top_ring[j], bottom_ring[j]])
            faces.append([top_ring[i], bottom_ring[j], bottom_ring[i]])

        if not vertices or not faces:
            return None

        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=False)
        trimesh.repair.fix_normals(mesh)
        mesh.process(validate=True)
        return mesh

    def _build_extruded_mesh(self, regressed_grid, X, Y, height_mm):
        """Build a smoother watertight extruded mesh from smoothed 2D contours."""
        solid_mask = np.asarray(regressed_grid, dtype=np.float32)
        if solid_mask.size == 0 or solid_mask.ndim != 2:
            return None

        # Smooth only the export geometry so the OBJ loses blocky stair-steps and tiny juts.
        smooth_mask = ndimage.gaussian_filter(solid_mask, sigma=0.9)
        contours = measure.find_contours(smooth_mask, 0.5)
        if not contours:
            return None

        pixel_height = float(abs(Y[1, 0] - Y[0, 0])) if Y.shape[0] > 1 else 1.0
        pixel_width = float(abs(X[0, 1] - X[0, 0])) if X.shape[1] > 1 else 1.0

        contour_worlds = []
        for contour in contours:
            if contour is None or len(contour) < 4:
                continue

            xs = X[0, 0] + contour[:, 1] * pixel_width
            ys = Y[0, 0] + contour[:, 0] * pixel_height
            coords = np.column_stack([xs, ys])

            signed_area = 0.5 * np.sum(coords[:-1, 0] * coords[1:, 1] - coords[1:, 0] * coords[:-1, 1])
            contour_worlds.append((coords, signed_area))

        if not contour_worlds:
            return None

        contour_worlds.sort(key=lambda item: abs(item[1]), reverse=True)

        def smooth_closed_ring(points, target_points=None):
            ring = np.asarray(points, dtype=float)
            if len(ring) < 4:
                return [(float(x), float(y)) for x, y in ring]

            if np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]

            if len(ring) < 4:
                return [(float(x), float(y)) for x, y in ring]

            deltas = np.diff(np.vstack([ring, ring[0]]), axis=0)
            perimeter = float(np.sum(np.linalg.norm(deltas, axis=1)))
            if target_points is None:
                target_points = int(np.clip(perimeter / max(pixel_width, pixel_height) * 1.5, 80, 800))

            try:
                tck, _ = splprep([ring[:, 0], ring[:, 1]], s=max(0.5, 0.001 * perimeter), per=True)
                u_new = np.linspace(0.0, 1.0, target_points, endpoint=False)
                x_new, y_new = splev(u_new, tck)
                return list(zip(x_new, y_new))
            except Exception:
                return [(float(x), float(y)) for x, y in ring]

        exterior_coords = smooth_closed_ring(contour_worlds[0][0])
        hole_coords_list = [smooth_closed_ring(item[0]) for item in contour_worlds[1:]]

        poly = Polygon(exterior_coords, holes=hole_coords_list)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None

        if poly.geom_type == "MultiPolygon":
            polygon_parts = [part for part in poly.geoms if not part.is_empty]
        else:
            polygon_parts = [poly]

        vertices = []
        vertex_index = {}
        faces = []

        def add_vertex(x, y, z):
            key = (float(x), float(y), float(z))
            if key not in vertex_index:
                vertex_index[key] = len(vertices)
                vertices.append(key)
            return vertex_index[key]

        def add_ring_faces(ring_coords):
            ring = [(float(x), float(y)) for x, y in ring_coords[:-1]] if np.allclose(ring_coords[0], ring_coords[-1]) else [(float(x), float(y)) for x, y in ring_coords]
            if len(ring) < 3:
                return

            top_ring = [add_vertex(x, y, height_mm) for x, y in ring]
            bottom_ring = [add_vertex(x, y, 0.0) for x, y in ring]
            for i in range(len(ring)):
                j = (i + 1) % len(ring)
                faces.append([top_ring[i], top_ring[j], bottom_ring[j]])
                faces.append([top_ring[i], bottom_ring[j], bottom_ring[i]])

        def add_cap_faces(polygon_part):
            for tri in triangulate(polygon_part):
                if tri.is_empty:
                    continue
                if not polygon_part.buffer(1e-9).covers(tri.representative_point()):
                    continue

                coords = list(tri.exterior.coords)[:-1]
                if len(coords) != 3:
                    continue

                top = [add_vertex(x, y, height_mm) for x, y in coords]
                bottom = [add_vertex(x, y, 0.0) for x, y in coords]
                faces.append(top)
                faces.append([bottom[0], bottom[2], bottom[1]])

        for polygon_part in polygon_parts:
            add_cap_faces(polygon_part)
            add_ring_faces(polygon_part.exterior.coords)
            for interior in polygon_part.interiors:
                add_ring_faces(interior.coords)

        if not vertices or not faces:
            return None

        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=False)
        trimesh.repair.fix_normals(mesh)
        mesh.process(validate=True)
        if mesh.volume < 0:
            mesh.invert()
            trimesh.repair.fix_normals(mesh)
            mesh.process(validate=True)
        return mesh

    def export_regressed_obj_after_distance(self, regression_distance_mm, output_obj_path=None):
        """Export a regressed 3D OBJ mesh after a requested regression distance in mm."""
        section_points_2d = self.read_obj_geometry()
        X, Y = self.create_initial_grid(section_points_2d)
        regressed_grid, _ = self.fast_marching_method(regression_distance_mm, X, Y)

        grain_length_mm = None
        if self.mesh is not None:
            bounds = self.mesh.bounds
            grain_length_mm = float(abs(bounds[1][self.cross_section_axis] - bounds[0][self.cross_section_axis]) * 1000.0)

        area_mm2, mesh = self.calculate_live_center_bore_surface_area_by_remesh(
            regressed_grid,
            X,
            Y,
            grain_length_mm=grain_length_mm,
            return_mesh=True,
        )

        if mesh is None:
            raise RuntimeError(
                "Could not build a regressed mesh from the current cross-section."
            )

        if output_obj_path is None:
            input_stem = Path(self.obj_file).stem
            output_obj_path = Path(f"{input_stem}_regressed_{int(round(regression_distance_mm))}mm.obj")
        else:
            output_obj_path = Path(output_obj_path)
            if output_obj_path.is_dir() or output_obj_path.suffix.lower() != ".obj":
                input_stem = Path(self.obj_file).stem
                output_obj_path = output_obj_path / f"{input_stem}_regressed_{int(round(regression_distance_mm))}mm.obj"

        output_obj_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_obj_path)
        print(f"Exported regressed OBJ to {output_obj_path} ({area_mm2:.2f} mm² bore surface)")
        return output_obj_path
    
    def simulate_regression(self, regression_rate, time_seconds):
        """
        Simulate fuel grain regression over time.
        
        Args:
            regression_rate: Regression rate (mm/sec)
            time_seconds: Time duration to simulate (seconds)
            
        Returns:
            Dictionary with simulation results
        """
        section_points_2d = self.read_obj_geometry()
        X, Y = self.create_initial_grid(section_points_2d)
        initial_center_bore_surface_area = self.calculate_center_bore_surface_area()
        self.full_3d_surface_area_mm2 = self.calculate_full_shape_surface_area()
        self.initial_bore_perimeter_mm = self.calculate_bore_perimeter(self.grid, X, Y)
        
        regression_distance = regression_rate * time_seconds
        regressed_grid, new_id_radius = self.fast_marching_method(regression_distance)
        center_bore_area = self.calculate_center_bore_area(regressed_grid, X, Y)
        center_bore_surface_area = self._estimate_bore_surface_area_from_regression_distance(regression_distance)
        self.center_bore_surface_area_mm2 = center_bore_surface_area
        
        results = {
            'initial_grid': self.grid,
            'regressed_grid': regressed_grid,
            'X': X,
            'Y': Y,
            'initial_id_radius': self.id_radius,
            'new_id_radius': new_id_radius,
            'od_radius': self.od_radius,
            'regression_distance': regression_distance,
            'regression_rate': regression_rate,
            'time_seconds': time_seconds,
            'center_bore_area_mm2': center_bore_area,
            'initial_center_bore_surface_area_mm2': initial_center_bore_surface_area,
            'center_bore_surface_area_mm2': center_bore_surface_area
        }
        
        return results
    
    def plot_dxf_geometry(self):
        """Plot the raw OBJ cross-section geometry"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Read and plot OBJ geometry
        try:
            bounds = self.mesh.bounds
            
            # Determine cross-section position
            axis_names = ['X', 'Y', 'Z']
            min_pos = bounds[0][self.cross_section_axis]
            max_pos = bounds[1][self.cross_section_axis]
            
            if self.cross_section_pos is None:
                cross_pos = (min_pos + max_pos) / 2
            else:
                cross_pos = self.cross_section_pos
            
            # Define plane normal
            plane_normals = [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1])
            ]
            
            plane_normal = plane_normals[self.cross_section_axis]
            plane_point = np.zeros(3)
            plane_point[self.cross_section_axis] = cross_pos
            
            # Extract cross-section
            section = self.mesh.section(plane_normal, plane_point)
            
            all_points = []
            
            if section is not None and len(section.entities) > 0:
                for discrete_path in section.discrete:
                    if isinstance(discrete_path, np.ndarray) and len(discrete_path) > 0:
                        # Project to 2D based on axis
                        if self.cross_section_axis == 0:  # X-axis, use Y-Z
                            points_2d = discrete_path[:, [1, 2]]
                        elif self.cross_section_axis == 1:  # Y-axis, use X-Z
                            points_2d = discrete_path[:, [0, 2]]
                        else:  # Z-axis, use X-Y
                            points_2d = discrete_path[:, [0, 1]]
                        
                        # Convert from OBJ units to millimeters (multiply by 1000)
                        points_2d = points_2d * 1000
                        points_2d_plot = self._to_plot_length(points_2d)
                        
                        ax.fill(points_2d_plot[:, 0], points_2d_plot[:, 1], alpha=0.4, color='lightblue', edgecolor='blue', linewidth=2.5)
                        ax.plot(points_2d_plot[:, 0], points_2d_plot[:, 1], 'b-', linewidth=2.5)
                        all_points.extend(points_2d_plot.tolist())
            
            # Auto-scale to fit geometry
            if all_points:
                all_points = np.array(all_points)
                margin = 0.1
                x_min, y_min = all_points.min(axis=0)
                x_max, y_max = all_points.max(axis=0)
                x_range = x_max - x_min if x_max > x_min else 1
                y_range = y_max - y_min if y_max > y_min else 1
                ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
                ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(self._length_label('Distance'), fontsize=12)
            ax.set_ylabel(self._length_label('Distance'), fontsize=12)
            ax.set_title(f'OBJ Cross-Section at {axis_names[self.cross_section_axis]}={self._to_plot_length(cross_pos):.3f} {self.plot_length_unit}', fontsize=12)
            
            plt.tight_layout()
            self._present_plot(fig, f"obj_cross_section_{self.cross_section_axis}")
            
        except Exception as e:
            print(f"Error plotting OBJ geometry: {e}")
    
    def plot_results(self, results):
        """
        Visualize the fuel grain before and after regression.
        
        Args:
            results: Dictionary from simulate_regression
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        X_plot = self._to_plot_length(results['X'])
        Y_plot = self._to_plot_length(results['Y'])
        od_radius_plot = self._to_plot_length(results['od_radius'])
        initial_id_radius_plot = self._to_plot_length(results['initial_id_radius'])
        new_id_radius_plot = self._to_plot_length(results['new_id_radius'])
        regression_distance_plot = self._to_plot_length(results['regression_distance'])
        # Initial geometry
        ax = axes[0]
        ax.contourf(X_plot, Y_plot, results['initial_grid'],
                levels=[0, 0.5, 1], colors=['white', 'lightblue'], alpha=0.8)
        ax.contour(X_plot, Y_plot, results['initial_grid'],
               levels=[0.5], colors=['blue'], linewidths=2)
        circle_od = Circle(self._to_plot_point(self.center), float(od_radius_plot),
                   fill=False, color='black', linewidth=2, label='OD (Fixed)')
        circle_id = Circle(self._to_plot_point(self.center), float(initial_id_radius_plot),
                   fill=False, color='blue', linewidth=2, label='ID (Initial)')
        ax.add_patch(circle_od)
        ax.add_patch(circle_id)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(self._length_label('X'))
        ax.set_ylabel(self._length_label('Y'))
        ax.set_title(f'Initial Fuel Grain\nID: {initial_id_radius_plot:.2f} {self.plot_length_unit}, OD: {od_radius_plot:.2f} {self.plot_length_unit}')
        ax.legend(loc='upper right')

        # Regressed geometry
        ax = axes[1]
        ax.contourf(X_plot, Y_plot, results['regressed_grid'],
                levels=[0, 0.5, 1], colors=['white', 'lightsalmon'], alpha=0.8)
        ax.contour(X_plot, Y_plot, results['regressed_grid'],
               levels=[0.5], colors=['red'], linewidths=2)
        circle_od_new = Circle(self._to_plot_point(self.center), float(od_radius_plot),
                       fill=False, color='black', linewidth=2, label='OD (Fixed)')
        circle_id_new = Circle(self._to_plot_point(self.center), float(new_id_radius_plot),
                       fill=False, color='red', linewidth=2, label=f'ID (After {regression_distance_plot:.2f} {self.plot_length_unit} burn)')
        ax.add_patch(circle_od_new)
        ax.add_patch(circle_id_new)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(self._length_label('X'))
        ax.set_ylabel(self._length_label('Y'))
        ax.set_title(f'Fuel Grain After Regression\nID: {new_id_radius_plot:.2f} {self.plot_length_unit}, OD: {od_radius_plot:.2f} {self.plot_length_unit}')
        ax.legend(loc='upper right')

        # Do not call tight_layout with custom slider axes; it can shift/overlap controls.
        if USE_GUI_PLOTS:
            fig.canvas.draw_idle()
            # Block here so the UI event loop stays responsive while using the slider.
            plt.show(block=True)
        else:
            self._present_plot(fig, "interactive_regression")
    
    def calculate_inscribed_circle_boundary_overlap(self, regressed_grid, max_circle_center, max_circle_radius, pixel_width, X, Y):
        """
        Calculate arc length where the green inscribed circle's perimeter touches the red fuel grain.
        This measures extrusions that protrude into the hole.
        
        Args:
            regressed_grid: Binary grid of regressed fuel grain
            max_circle_center: Center of the inscribed circle (world coords)
            max_circle_radius: Radius of the inscribed circle (mm)
            pixel_width: Width of each pixel in mm
            X: X meshgrid (for coordinate conversion)
            Y: Y meshgrid (for coordinate conversion)
            
        Returns:
            overlap_length_mm: Arc length in mm where circle perimeter touches fuel
        """
        try:
            if max_circle_center is None or max_circle_radius <= 0:
                return 0.0
            
            from scipy import interpolate
            
            # Convert world coordinates to pixel coordinates
            radius_pixels = max_circle_radius / pixel_width
            pixel_height = (Y[1, 0] - Y[0, 0])
            center_pixel_x = (max_circle_center[0] - X[0, 0]) / pixel_width
            center_pixel_y = (max_circle_center[1] - Y[0, 0]) / pixel_height
            
            # Create interpolation function for smooth sampling
            grid_y = np.arange(regressed_grid.shape[0])
            grid_x = np.arange(regressed_grid.shape[1])
            f = interpolate.RegularGridInterpolator((grid_y, grid_x), regressed_grid, bounds_error=False, fill_value=0)
            
            # Sample points around the circle's perimeter with high resolution
            num_samples = max(int(4 * np.pi * radius_pixels), 200)  # Higher resolution
            angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
            
            # Calculate arc length per sample point
            arc_length_per_point = (2 * np.pi * max_circle_radius) / num_samples
            
            # Count how many points on the circle perimeter touch fuel
            touching_count = 0
            
            for angle in angles:
                x = center_pixel_x + radius_pixels * np.cos(angle)
                y = center_pixel_y + radius_pixels * np.sin(angle)
                
                # Use interpolation to get smooth fuel value at this perimeter point
                if 0 <= y < regressed_grid.shape[0] and 0 <= x < regressed_grid.shape[1]:
                    fuel_value = f([[y, x]])[0]
                    # Consider touching if fuel value is above 0.1 (real fuel, not noise)
                    if fuel_value > 0.0225:
                        touching_count += 1
            
            total_overlap_length = touching_count * arc_length_per_point
            return total_overlap_length
        
        except Exception as e:
            print(f"Warning: Could not calculate circle boundary overlap: {e}")
            return 0.0
    
    def plot_cross_section_interactive(self, regression_rate, max_time, frame_count=100):
        """Display cached regression playback with a responsive slider.

        This precomputes the regression frames once, then the slider only swaps
        between cached frames and metrics. That makes dragging smooth and avoids
        expensive geometry work on every slider event.
        """
        if self.grid is None or np.sum(self.grid) == 0:
            print("Error: Grid is empty! Cannot display regression.")
            return

        frame_count = int(max(20, frame_count))

        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        X_plot = self._to_plot_length(X)
        Y_plot = self._to_plot_length(Y)

        if getattr(self, "full_3d_surface_area_mm2", None) is None:
            self.full_3d_surface_area_mm2 = self.calculate_center_bore_surface_area()

        print(f"Precomputing {frame_count} slider frames...")
        times = np.linspace(0.0, float(max_time), frame_count)
        distances = regression_rate * times
        cached_grids = []
        cached_bore_areas = np.zeros(frame_count, dtype=float)
        cached_bore_surfaces = np.zeros(frame_count, dtype=float)
        cached_perimeters = np.zeros(frame_count, dtype=float)

        for i, distance in enumerate(distances):
            regressed_grid, _ = self.fast_marching_method(float(distance), X, Y)
            cached_grids.append(regressed_grid)
            cached_bore_areas[i] = self.calculate_center_bore_area(regressed_grid, X, Y)
            cached_perimeters[i] = self.calculate_bore_perimeter(regressed_grid, X, Y)
            cached_bore_surfaces[i] = self._estimate_bore_surface_area_from_regression_distance(float(distance))

            if (i + 1) % max(1, frame_count // 5) == 0 or i == frame_count - 1:
                print(f"  Cached {i+1}/{frame_count} frames")

        fig = plt.figure(figsize=(14, 10))
        ax_plot = plt.axes([0.15, 0.25, 0.7, 0.65])
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        plt.subplots_adjust(top=0.92)

        slider = Slider(ax_slider, 'Time (sec)', 0.0, float(max_time), valinit=0.0, color='steelblue')

        def draw_frame(frame_index):
            ax_plot.clear()

            regressed_grid = cached_grids[frame_index]
            bore_area_plot = self._to_plot_area(cached_bore_areas[frame_index])
            perimeter_plot = self._to_plot_length(cached_perimeters[frame_index])
            bore_surface_plot = self._to_plot_area(cached_bore_surfaces[frame_index])

            ax_plot.imshow(
                self.grid,
                extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
                origin='lower',
                cmap='Blues',
                alpha=0.35,
            )
            ax_plot.contour(X_plot, Y_plot, self.grid, levels=[0.5], colors=['blue'], linewidths=2.5, linestyles='--')

            ax_plot.imshow(
                regressed_grid,
                extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
                origin='lower',
                cmap='Reds',
                alpha=0.35,
            )
            ax_plot.contour(X_plot, Y_plot, regressed_grid, levels=[0.5], colors=['red'], linewidths=2.5)

            metrics_text = (
                f'Bore Area: {bore_area_plot:.2f} {self.plot_area_unit}\n'
                f'Perimeter: {perimeter_plot:.2f} {self.plot_length_unit}\n'
                f'Bore Surface: {bore_surface_plot:.2f} {self.plot_area_unit}'
            )
            ax_plot.text(
                0.01,
                0.99,
                metrics_text,
                transform=ax_plot.transAxes,
                fontsize=10,
                va='top',
                ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

            ax_plot.set_aspect('equal')
            ax_plot.grid(True, alpha=0.3)
            ax_plot.set_xlabel(self._length_label('Distance'), fontsize=12)
            ax_plot.set_ylabel(self._length_label('Distance'), fontsize=12)
            ax_plot.set_title(
                f'Fuel Grain Regression | Time: {times[frame_index]:.2f} sec | '
                f'Regression: {self._to_plot_length(distances[frame_index]):.4f} {self.plot_length_unit}',
                fontsize=11,
                fontweight='bold',
            )

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='Initial'),
                Line2D([0], [0], color='red', linewidth=2.5, linestyle='-', label='After Regression'),
            ]
            ax_plot.legend(handles=legend_elements, loc='upper right', fontsize=11)

            fig.canvas.draw_idle()

        def update(_):
            normalized = slider.val / max(float(max_time), 1e-9)
            frame_index = int(np.clip(round(normalized * (frame_count - 1)), 0, frame_count - 1))
            draw_frame(frame_index)

        slider.on_changed(update)
        draw_frame(0)

        if USE_GUI_PLOTS:
            fig.canvas.draw_idle()
            plt.show(block=True)
        else:
            self._present_plot(fig, "interactive_regression")
    
    def plot_area_vs_regression(self, max_regression_distance=None):
        """
        Plot bore area vs regression distance with polynomial fit and R² value.
        
        Args:
            max_regression_distance: Maximum regression distance to plot (mm).
                                    If None, uses initial ID radius.
        """
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius
        
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        regression_distances_plot = self._to_plot_length(regression_distances)
        
        # Arrays to store only valid pre-burn-through samples
        regression_distances_valid = []
        areas_valid = []
        burnthrough_distance = None
        
        print(f"Calculating bore area vs regression (this may take a moment)...")
        
        # Calculate bore areas for each regression distance
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)
            area_value = self.calculate_center_bore_area(regressed_grid, X, Y)
            
            # Stop if bore area reaches 0
            if area_value <= 0:
                burnthrough_distance = float(reg_dist)
                print(f"  Bore area reached zero at regression distance {reg_dist:.4f} mm")
                break

            regression_distances_valid.append(float(reg_dist))
            areas_valid.append(float(area_value))
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")

        if len(areas_valid) == 0:
            print("WARNING: No valid bore area data points found; skipping area plot.")
            return np.array([0, 0, 0])

        regression_distances = np.asarray(regression_distances_valid, dtype=float)
        areas = np.asarray(areas_valid, dtype=float)
        regression_distances_plot = self._to_plot_length(regression_distances)
        areas_plot = self._to_plot_area(areas)
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, areas, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        fit_plot = self._to_plot_area(fit)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances_plot, areas_plot, 'o-', linewidth=3, markersize=6,
             label='Bore Area', color='steelblue', alpha=0.7)
        ax.plot(regression_distances_plot, fit_plot, '--', linewidth=2.5, color='coral',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel(self._length_label('Regression Distance'), fontsize=13, fontweight='bold')
        ax.set_ylabel(self._area_label('Bore Area'), fontsize=13, fontweight='bold')
        ax.set_title(f'Bore Area vs Regression Distance\n{Path(self.obj_file).name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='upper right')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        initial_area = areas[0]
        final_area = areas[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"A = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"A = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            # Generic format for higher degrees
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Bore Area: {areas_plot[0]:.2f} {self.plot_area_unit}\n'
        stats_text += f'Final Bore Area: {areas_plot[-1]:.2f} {self.plot_area_unit}\n'
        stats_text += f'Bore Area Change: {areas_plot[-1] - areas_plot[0]:.2f} {self.plot_area_unit}\n'
        stats_text += f'Max Regression: {self._to_plot_length(max_regression_distance):.2f} {self.plot_length_unit}\n'
        if burnthrough_distance is not None:
            stats_text += f'Burn-through: {self._to_plot_length(burnthrough_distance):.2f} {self.plot_length_unit}\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += self._format_polynomial_equation(coeffs, degree_used, 'A', self.plot_area_scale_mm2, r2)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        self._present_plot(fig, "bore_area_vs_regression")
        
        # Print summary
        print(f"\nBore Area vs Regression Summary:")
        print(f"  Initial bore area: {areas_plot[0]:.2f} {self.plot_area_unit}")
        print(f"  Final bore area: {areas_plot[-1]:.2f} {self.plot_area_unit}")
        print(f"  Bore area change: {areas_plot[-1] - areas_plot[0]:.2f} {self.plot_area_unit}")
        if burnthrough_distance is not None:
            print(f"  Burn-through regression distance: {self._to_plot_length(burnthrough_distance):.2f} {self.plot_length_unit}")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        
        return coeffs

    def plot_surface_area_vs_regression(self, max_regression_distance=None):
        """
        Plot bore surface area vs regression distance with polynomial fit and R² value.

        Args:
            max_regression_distance: Maximum regression distance to plot (mm).
                                    If None, uses initial ID radius.
        """
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)

        # Determine grain length from mesh if needed
        if self.mesh is None:
            self.read_obj_geometry()
        bounds = self.mesh.bounds
        grain_length_mm = float(abs(bounds[1][self.cross_section_axis] - bounds[0][self.cross_section_axis]) * 1000.0)

        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius

        cad_initial_surface_area_mm2 = self.calculate_center_bore_surface_area()
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        regression_distances_plot = self._to_plot_length(regression_distances)

        # Array to store results
        surface_areas = np.zeros(num_points)

        print(f"Calculating surface area vs regression (3D mesh baseline + burn fraction)...")

        # Use the contour-derived initial bore area as the baseline, then grow it with
        # regression progress. On this helical grain, the X-axis contour is the stable
        # source of the initial CAD-like bore area.
        valid_count = 0
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)

            surface_areas[i] = self._estimate_bore_surface_area_from_regression_distance(reg_dist)

            valid_count = i + 1

            # Stop once the bore surface area disappears.
            if surface_areas[i] <= 0:
                print(f"  Surface area reached zero at regression distance {reg_dist:.4f} mm")
                break

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")

        # Shift the curve so it starts at the CAD-like initial surface area.
        if valid_count > 0 and surface_areas[0] > 0:
            cad_offset = cad_initial_surface_area_mm2 - surface_areas[0]
            surface_areas[:valid_count] += cad_offset

        # Trim arrays to valid data only
        regression_distances = regression_distances[:valid_count]
        surface_areas = surface_areas[:valid_count]
        regression_distances_plot = regression_distances_plot[:valid_count]

        valid_mask = surface_areas > 0
        if np.sum(valid_mask) < 3:
            print("ERROR: Not enough valid data points for surface-area plot.")
            return np.array([0, 0, 0])

        regression_distances_plot_valid = regression_distances_plot[valid_mask]
        regression_distances_valid = regression_distances[valid_mask]
        surface_areas_valid = surface_areas[valid_mask]
        surface_areas_plot = self._to_plot_area(surface_areas_valid)

        print(f"Fitting polynomial (target R² = 0.99) to CAD-like baseline data...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances_valid, surface_areas_valid, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances_valid)
        fit_plot = self._to_plot_area(fit)

        # Plot the results.
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(regression_distances_plot_valid, surface_areas_plot, 'o-', linewidth=3, markersize=6,
               label='Bore Surface Area', color='teal', alpha=0.9)
        ax.plot(regression_distances_plot_valid, fit_plot, '--', linewidth=2.5, color='magenta',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')

        ax.set_xlabel(self._length_label('Regression Distance'), fontsize=13, fontweight='bold')
        ax.set_ylabel(self._area_label('Bore Surface Area'), fontsize=13, fontweight='bold')
        ax.set_title(f'Bore Surface Area vs Regression Distance\n{Path(self.obj_file).name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='upper right')

        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add statistics to the plot
        primary_plot_display = surface_areas_plot

        if degree_used == 2:
            eq = f"S = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"S = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"

        stats_text = f'Initial Surface Area: {primary_plot_display[0]:.2f} {self.plot_area_unit}\n'
        stats_text += f'Final Surface Area: {primary_plot_display[-1]:.2f} {self.plot_area_unit}\n'
        stats_text += f'Surface Area Change: {primary_plot_display[-1] - primary_plot_display[0]:.2f} {self.plot_area_unit}\n'
        stats_text += f'Max Regression: {self._to_plot_length(max_regression_distance):.2f} {self.plot_length_unit}\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += self._format_polynomial_equation(coeffs, degree_used, 'S', self.plot_area_scale_mm2, r2)

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.85),
               family='monospace')

        plt.tight_layout()
        self._present_plot(fig, "perimeter_vs_regression")

        # Print summary
        print(f"\nBore Surface Area vs Regression Summary:")
        print(f"  Initial surface area: {primary_plot_display[0]:.2f} {self.plot_area_unit}")
        print(f"  Final surface area: {primary_plot_display[-1]:.2f} {self.plot_area_unit}")
        print(f"  Surface area change: {primary_plot_display[-1] - primary_plot_display[0]:.2f} {self.plot_area_unit}")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")

        return coeffs

    def plot_perimeter_vs_regression(self, max_regression_distance=None):
        """
        Plot inner hole perimeter vs regression distance with polynomial fit and R² value.
        
        Args:
            max_regression_distance: Maximum regression distance to plot (mm).
                                    If None, uses initial ID radius.
        """
        from skimage import measure
        
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate pixel dimensions
        pixel_width = (X[0, 1] - X[0, 0])
        pixel_height = (Y[1, 0] - Y[0, 0])
        
        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius
        
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        regression_distances_plot = self._to_plot_length(regression_distances)
        
        # Array to store results
        perimeters = np.zeros(num_points)
        
        print(f"Calculating perimeter vs regression (this may take a moment)...")
        
        # Calculate perimeters for each regression distance
        valid_count = 0
        pixel_area = pixel_width * pixel_height
        
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)
            
            # Check if fuel area has become zero
            fuel_area = np.sum(regressed_grid) * pixel_area
            if fuel_area <= 0:
                print(f"  Fuel completely burned at regression distance {reg_dist:.4f} mm")
                break
            
            # Find inner hole and calculate perimeter
            empty_space = 1 - regressed_grid
            labeled, num_features = ndimage.label(empty_space)
            border_label = labeled[0, 0]
            
            perimeter = 0
            for label in range(1, num_features + 1):
                if label != border_label:
                    inner_hole_mask = (labeled == label)
                    perimeter_pixels = measure.perimeter(inner_hole_mask, neighborhood=4)
                    perimeter = perimeter_pixels * pixel_width * 0.95
                    break
            
            perimeters[i] = perimeter
            valid_count = i + 1
            
            # Stop if perimeter becomes zero
            if perimeter <= 0:
                print(f"  Perimeter became zero at regression distance {reg_dist:.4f} mm")
                break
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")
        
        # Trim arrays to valid data only
        regression_distances = regression_distances[:valid_count]
        perimeters = perimeters[:valid_count]
        regression_distances_plot = regression_distances_plot[:valid_count]
        
        # Remove zero perimeter points
        valid_mask = perimeters > 0
        regression_distances = regression_distances[valid_mask]
        perimeters = perimeters[valid_mask]
        regression_distances_plot = regression_distances_plot[valid_mask]
        perimeters_plot = self._to_plot_length(perimeters)

        if len(perimeters) == 0:
            print("WARNING: No valid perimeter data points found; skipping perimeter plot.")
            return np.array([0, 0, 0])
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, perimeters, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        fit_plot = self._to_plot_length(fit)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances_plot, perimeters_plot, 'o-', linewidth=3, markersize=6,
               label='Inner Hole Perimeter', color='darkgreen', alpha=0.7)
        ax.plot(regression_distances_plot, fit_plot, '--', linewidth=2.5, color='orange',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel(self._length_label('Regression Distance'), fontsize=13, fontweight='bold')
        ax.set_ylabel(self._length_label('Inner Hole Perimeter'), fontsize=13, fontweight='bold')
        ax.set_title(f'Inner Hole Perimeter vs Regression Distance\n{Path(self.obj_file).name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        if len(perimeters) == 0:
            print("WARNING: No valid perimeter data points found; skipping perimeter plot.")
            return np.array([0, 0, 0])
        initial_perimeter = perimeters[0]
        final_perimeter = perimeters[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"P = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"P = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Perimeter: {perimeters_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Final Perimeter: {perimeters_plot[-1]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Perimeter Change: {perimeters_plot[-1] - perimeters_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Max Regression: {self._to_plot_length(max_regression_distance):.2f} {self.plot_length_unit}\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += self._format_polynomial_equation(coeffs, degree_used, 'P', self.plot_length_scale_mm, r2)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        self._present_plot(fig, "max_inscribed_circle_vs_regression")
        
        # Print summary
        print(f"\nPerimeter vs Regression Summary:")
        print(f"  Initial perimeter: {perimeters_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Final perimeter: {perimeters_plot[-1]:.2f} {self.plot_length_unit}")
        print(f"  Perimeter change: {perimeters_plot[-1] - perimeters_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        
        return coeffs

    def plot_max_inscribed_circle_vs_regression(self, max_regression_distance=None):
        """
        Plot maximum inscribed circle diameter vs regression distance with polynomial fit.
        This is the largest circle that fits inside the inner hole.
        
        Args:
            max_regression_distance: Maximum regression distance to plot (mm).
                                    If None, uses initial ID radius.
        """
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate pixel dimensions
        pixel_width = (X[0, 1] - X[0, 0])
        pixel_height = (Y[1, 0] - Y[0, 0])
        pixel_area = pixel_width * pixel_height
        
        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius
        
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        regression_distances_plot = self._to_plot_length(regression_distances)
        
        # Array to store results
        max_inscribed_diameters = np.zeros(num_points)
        
        print(f"Calculating max inscribed circle diameter vs regression...")
        
        # Calculate max inscribed circles for each regression distance
        valid_count = 0
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)
            
            # Check if fuel area has become zero
            fuel_area = np.sum(regressed_grid) * pixel_area
            if fuel_area <= 0:
                print(f"  Fuel completely burned at regression distance {reg_dist:.4f} mm")
                break
            
            # Find inner hole and calculate max inscribed circle
            empty_space = 1 - regressed_grid
            labeled, num_features = ndimage.label(empty_space)
            border_label = labeled[0, 0]
            
            max_inscribed_diam = 0
            for label in range(1, num_features + 1):
                if label != border_label:
                    inner_hole_mask = (labeled == label)
                    distance_from_boundary = distance_transform_edt(inner_hole_mask)
                    max_radius_pixels = np.max(distance_from_boundary)
                    max_inscribed_diam = 2 * max_radius_pixels * pixel_width
                    break
            
            max_inscribed_diameters[i] = max_inscribed_diam
            valid_count = i + 1
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")
        
        # Trim arrays to valid data only
        regression_distances = regression_distances[:valid_count]
        max_inscribed_diameters = max_inscribed_diameters[:valid_count]
        regression_distances_plot = regression_distances_plot[:valid_count]
        
        # Remove zero diameter points
        valid_mask = max_inscribed_diameters > 0
        regression_distances = regression_distances[valid_mask]
        max_inscribed_diameters = max_inscribed_diameters[valid_mask]
        regression_distances_plot = regression_distances_plot[valid_mask]
        max_inscribed_diameters_plot = self._to_plot_length(max_inscribed_diameters)
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, max_inscribed_diameters, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        fit_plot = self._to_plot_length(fit)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances_plot, max_inscribed_diameters_plot, 'o-', linewidth=3, markersize=6,
               label='Max Inscribed Circle Diameter', color='darkblue', alpha=0.7)
        ax.plot(regression_distances_plot, fit_plot, '--', linewidth=2.5, color='cyan',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel(self._length_label('Regression Distance'), fontsize=13, fontweight='bold')
        ax.set_ylabel(self._length_label('Max Inscribed Circle Diameter'), fontsize=13, fontweight='bold')
        ax.set_title(f'Max Inscribed Circle Diameter vs Regression Distance\n{Path(self.obj_file).name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        if len(max_inscribed_diameters) == 0:
            print("WARNING: No valid max-inscribed data points found; skipping max-inscribed plot.")
            return np.array([0, 0, 0])
        initial_max_inscribed = max_inscribed_diameters[0]
        final_max_inscribed = max_inscribed_diameters[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"D = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"D = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Max Inscribed: {max_inscribed_diameters_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Final Max Inscribed: {max_inscribed_diameters_plot[-1]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Diameter Change: {max_inscribed_diameters_plot[-1] - max_inscribed_diameters_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Max Regression: {self._to_plot_length(max_regression_distance):.2f} {self.plot_length_unit}\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += self._format_polynomial_equation(coeffs, degree_used, 'D', self.plot_length_scale_mm, r2)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        self._present_plot(fig, "min_enclosing_circle_vs_regression")
        
        # Print summary
        print(f"\nMax Inscribed Circle vs Regression Summary:")
        print(f"  Initial diameter: {max_inscribed_diameters_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Final diameter: {max_inscribed_diameters_plot[-1]:.2f} {self.plot_length_unit}")
        print(f"  Diameter change: {max_inscribed_diameters_plot[-1] - max_inscribed_diameters_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        
        return coeffs

    def plot_min_enclosing_circle_vs_regression(self, max_regression_distance=None):
        """
        Plot minimum enclosing circle diameter vs regression distance with polynomial fit.
        This is the smallest circle that encompasses the entire inner hole boundary.
        
        Args:
            max_regression_distance: Maximum regression distance to plot (mm).
                                    If None, uses initial ID radius.
        """
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate pixel dimensions
        pixel_width = (X[0, 1] - X[0, 0])
        pixel_height = (Y[1, 0] - Y[0, 0])
        pixel_area = pixel_width * pixel_height
        
        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius
        
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        regression_distances_plot = self._to_plot_length(regression_distances)
        
        # Array to store results
        min_enclosing_diameters = np.zeros(num_points)
        
        print(f"Calculating min enclosing circle diameter vs regression...")
        
        # Calculate min enclosing circles for each regression distance
        valid_count = 0
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)
            
            # Check if fuel area has become zero
            fuel_area = np.sum(regressed_grid) * pixel_area
            if fuel_area <= 0:
                print(f"  Fuel completely burned at regression distance {reg_dist:.4f} mm")
                break
            
            # Find inner hole and calculate min enclosing circle
            empty_space = 1 - regressed_grid
            labeled, num_features = ndimage.label(empty_space)
            border_label = labeled[0, 0]
            
            min_enclosing_diam = 0
            for label in range(1, num_features + 1):
                if label != border_label:
                    inner_hole_mask = (labeled == label)
                    contours = measure.find_contours(inner_hole_mask, 0.5)
                    if contours and len(contours) > 0:
                        boundary_pixels = contours[0]
                        # Convert to world coordinates
                        boundary_points = np.column_stack([
                            X[0, 0] + boundary_pixels[:, 1] * pixel_width,
                            Y[0, 0] + boundary_pixels[:, 0] * pixel_height
                        ])
                        
                        # Find minimum enclosing circle
                        min_enclosing_center = boundary_points.mean(axis=0)
                        min_enclosing_radius = np.max(np.linalg.norm(boundary_points - min_enclosing_center, axis=1))
                        min_enclosing_diam = 2 * min_enclosing_radius
                    break
            
            min_enclosing_diameters[i] = min_enclosing_diam
            valid_count = i + 1
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")
        
        # Trim arrays to valid data only
        regression_distances = regression_distances[:valid_count]
        min_enclosing_diameters = min_enclosing_diameters[:valid_count]
        regression_distances_plot = regression_distances_plot[:valid_count]
        
        # Remove zero diameter points
        valid_mask = min_enclosing_diameters > 0
        regression_distances = regression_distances[valid_mask]
        min_enclosing_diameters = min_enclosing_diameters[valid_mask]
        regression_distances_plot = regression_distances_plot[valid_mask]
        min_enclosing_diameters_plot = self._to_plot_length(min_enclosing_diameters)
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, min_enclosing_diameters, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        fit_plot = self._to_plot_length(fit)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances_plot, min_enclosing_diameters_plot, 'o-', linewidth=3, markersize=6,
               label='Min Enclosing Circle Diameter', color='darkgreen', alpha=0.7)
        ax.plot(regression_distances_plot, fit_plot, '--', linewidth=2.5, color='lightgreen',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel(self._length_label('Regression Distance'), fontsize=13, fontweight='bold')
        ax.set_ylabel(self._length_label('Min Enclosing Circle Diameter'), fontsize=13, fontweight='bold')
        ax.set_title(f'Min Enclosing Circle Diameter vs Regression Distance\n{Path(self.obj_file).name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        if len(min_enclosing_diameters) == 0:
            print("WARNING: No valid min-enclosing data points found; skipping min-enclosing plot.")
            return np.array([0, 0, 0])
        initial_min_enclosing = min_enclosing_diameters[0]
        final_min_enclosing = min_enclosing_diameters[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"D = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"D = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Min Enclosing: {min_enclosing_diameters_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Final Min Enclosing: {min_enclosing_diameters_plot[-1]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Diameter Change: {min_enclosing_diameters_plot[-1] - min_enclosing_diameters_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Max Regression: {self._to_plot_length(max_regression_distance):.2f} {self.plot_length_unit}\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += self._format_polynomial_equation(coeffs, degree_used, 'D', self.plot_length_scale_mm, r2)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        self._present_plot(fig, "circle_boundary_overlap_vs_regression")
        
        # Print summary
        print(f"\nMin Enclosing Circle vs Regression Summary:")
        print(f"  Initial diameter: {min_enclosing_diameters_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Final diameter: {min_enclosing_diameters_plot[-1]:.2f} {self.plot_length_unit}")
        print(f"  Diameter change: {min_enclosing_diameters_plot[-1] - min_enclosing_diameters_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        
        return coeffs
    
    def plot_circle_boundary_overlap_vs_regression(self, max_regression_distance=None):
        """
        Plot inscribed circle boundary overlap (arc length) vs regression distance with polynomial fit.
        Stops when overlap falls below 15 mm.
        
        Args:
            max_regression_distance: Maximum regression distance to plot (mm).
                                    If None, uses initial ID radius.
        """
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate pixel area
        pixel_width = (X[0, 1] - X[0, 0])
        pixel_height = (Y[1, 0] - Y[0, 0])
        
        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius
        
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        regression_distances_plot = self._to_plot_length(regression_distances)
        
        # Array to store results
        overlaps = np.zeros(num_points)
        
        print(f"Calculating circle boundary overlap vs regression (this may take a moment)...")
        
        # Calculate overlaps for each regression distance
        valid_count = 0
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)
            
            # Get the largest hole
            labeled_grid, num_features = ndimage.label(regressed_grid == 0)
            if num_features == 0:
                print(f"  No holes found at regression distance {reg_dist:.4f} mm")
                break
            
            # Find largest hole
            hole_sizes = np.bincount(labeled_grid[labeled_grid > 0])
            largest_hole_label = np.argmax(hole_sizes) + 1
            largest_hole = (labeled_grid == largest_hole_label)
            
            # Calculate inscribed circle for largest hole
            dist_transform = distance_transform_edt(largest_hole)
            max_dist = np.max(dist_transform)
            if max_dist <= 0:
                break
            
            max_circle_radius = max_dist * pixel_width
            max_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
            max_circle_center = np.array([X[max_idx], Y[max_idx]])
            
            # Calculate overlap
            overlaps[i] = self.calculate_inscribed_circle_boundary_overlap(
                regressed_grid, max_circle_center, max_circle_radius, pixel_width, X, Y
            )
            
            valid_count = i + 1
            
            # Calculate 5% threshold based on current circle perimeter
            circle_perimeter = 2 * np.pi * max_circle_radius
            threshold_5percent = 0.05 * circle_perimeter
            
            # Stop if overlap falls below 5% of circle perimeter
            if overlaps[i] < threshold_5percent:
                print(f"  Overlap fell below 5% of circle perimeter ({threshold_5percent:.2f} mm) at regression distance {reg_dist:.4f} mm")
                break
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")
        
        # Trim arrays to valid data only
        regression_distances = regression_distances[:valid_count]
        overlaps = overlaps[:valid_count]
        regression_distances_plot = regression_distances_plot[:valid_count]
        
        # Apply Savitzky-Golay filter to smooth noise
        if len(overlaps) >= 5:
            # Use window length of at least 5 but less than data length, must be odd
            window_length = min(len(overlaps) - (1 if len(overlaps) % 2 == 0 else 0), 11)
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(5, window_length)
            overlaps_smooth = savgol_filter(overlaps, window_length=window_length, polyorder=3)
        else:
            overlaps_smooth = overlaps
        overlaps_plot = self._to_plot_length(overlaps)
        overlaps_smooth_plot = self._to_plot_length(overlaps_smooth)
        
        # Filter out invalid and near-zero data points before fitting
        # Keep only finite values and values above a minimum threshold (1% of max)
        valid_mask = np.isfinite(overlaps_smooth) & np.isfinite(regression_distances)
        if len(overlaps_smooth[valid_mask]) > 0:
            min_threshold = 0.01 * np.max(overlaps_smooth[valid_mask])
            valid_mask = valid_mask & (overlaps_smooth >= min_threshold)
        
        # Only proceed with fitting if we have enough data points
        if np.sum(valid_mask) < 3:
            print(f"WARNING: Not enough valid data points for polynomial fit (only {np.sum(valid_mask)})")
            # Use simple average or return dummy coefficients
            avg_overlap = np.nanmean(overlaps_smooth[np.isfinite(overlaps_smooth)])
            coeffs = np.array([0, 0, avg_overlap])  # Constant fit
            poly = np.poly1d(coeffs)
            fit = np.full_like(regression_distances, avg_overlap)
            r2 = 0
            degree_used = 0
        else:
            # Use filtered data for fitting
            x_filtered = regression_distances[valid_mask]
            y_filtered = overlaps_smooth[valid_mask]
            
            # Fit polynomial with adaptive degree based on R² threshold
            print(f"Fitting polynomial (target R² = 0.99) with {np.sum(valid_mask)} valid points...")
            coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
                x_filtered, y_filtered, target_r2=0.99, min_degree=2, max_degree=8
            )
            fit = poly(regression_distances)
        fit_plot = self._to_plot_length(fit)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances_plot, overlaps_plot, 'o-', linewidth=2, markersize=4,
               label='Raw Measurement', color='lightblue', alpha=0.5)
        ax.plot(regression_distances_plot, overlaps_smooth_plot, 's-', linewidth=2.5, markersize=5,
               label='Smoothed Data', color='darkblue', alpha=0.8)
        ax.plot(regression_distances_plot, fit_plot, '--', linewidth=2.5, color='cyan',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel(self._length_label('Regression Distance'), fontsize=13, fontweight='bold')
        ax.set_ylabel(self._length_label('Circle Boundary Overlap'), fontsize=13, fontweight='bold')
        ax.set_title(f'Inscribed Circle Boundary Overlap vs Regression Distance\n{Path(self.obj_file).name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='upper right')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        if len(overlaps) == 0:
            print("WARNING: No valid circle-overlap data points found; skipping circle-overlap plot.")
            return np.array([0, 0, 0])
        initial_overlap = overlaps[0]
        final_overlap = overlaps[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"L = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"L = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Overlap: {overlaps_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Final Overlap: {overlaps_plot[-1]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Overlap Change: {overlaps_plot[-1] - overlaps_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Max Regression: {self._to_plot_length(max_regression_distance):.2f} {self.plot_length_unit}\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += self._format_polynomial_equation(coeffs, degree_used, 'L', self.plot_length_scale_mm, r2)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        self._present_plot(fig, "largest_arm_overlap_vs_regression")
        
        # Print summary
        print(f"\nCircle Boundary Overlap vs Regression Summary:")
        print(f"  Initial overlap: {overlaps_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Final overlap: {overlaps_plot[-1]:.2f} {self.plot_length_unit}")
        print(f"  Overlap change: {overlaps_plot[-1] - overlaps_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        
        return coeffs
    
    def plot_largest_arm_overlap_vs_regression(self, max_regression_distance=None):
        """
        Plot largest extrusion arm contact arc length vs regression distance with polynomial fit.
        Measures how the dominant arm's contact length changes as the grain regresses.
        
        Args:
            max_regression_distance: Maximum regression distance to plot (mm).
                                    If None, uses initial ID radius.
        """
        from scipy import interpolate
        
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate pixel area
        pixel_width = (X[0, 1] - X[0, 0])
        pixel_height = (Y[1, 0] - Y[0, 0])
        
        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius
        
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        regression_distances_plot = self._to_plot_length(regression_distances)
        
        # Array to store results
        largest_arm_overlaps = np.zeros(num_points)
        
        print(f"Calculating largest arm overlap vs regression (this may take a moment)...")
        
        # Calculate largest arm overlaps for each regression distance
        valid_count = 0
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)
            
            # Get the largest hole
            labeled_grid, num_features = ndimage.label(regressed_grid == 0)
            if num_features == 0:
                print(f"  No holes found at regression distance {reg_dist:.4f} mm")
                break
            
            # Find largest hole
            hole_sizes = np.bincount(labeled_grid[labeled_grid > 0])
            largest_hole_label = np.argmax(hole_sizes) + 1
            largest_hole = (labeled_grid == largest_hole_label)
            
            # Calculate inscribed circle for largest hole
            dist_transform = distance_transform_edt(largest_hole)
            max_dist = np.max(dist_transform)
            if max_dist <= 0:
                break
            
            max_circle_radius = max_dist * pixel_width
            max_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
            max_circle_center = (X[0, 0] + max_idx[1] * pixel_width,
                                Y[0, 0] + max_idx[0] * pixel_height)
            
            # Create interpolation function for fuel grid
            grid_y = np.arange(regressed_grid.shape[0])
            grid_x = np.arange(regressed_grid.shape[1])
            f = interpolate.RegularGridInterpolator((grid_y, grid_x), regressed_grid, bounds_error=False, fill_value=0)
            
            # Sample circle perimeter
            radius_pixels = max_circle_radius / pixel_width
            center_pixel_x = (max_circle_center[0] - X[0, 0]) / pixel_width
            center_pixel_y = (max_circle_center[1] - Y[0, 0]) / pixel_height
            
            num_samples = max(int(4 * np.pi * radius_pixels), 200)
            angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
            
            # Collect all fuel values to determine adaptive threshold
            all_fuel_values = []
            for angle in angles:
                x_pix = center_pixel_x + radius_pixels * np.cos(angle)
                y_pix = center_pixel_y + radius_pixels * np.sin(angle)
                
                if 0 <= y_pix < regressed_grid.shape[0] and 0 <= x_pix < regressed_grid.shape[1]:
                    fuel_value = f([[y_pix, x_pix]])[0]
                    all_fuel_values.append(fuel_value)
            
            # Use adaptive threshold: 5th percentile of all fuel values on circle perimeter
            if all_fuel_values:
                all_fuel_array = np.array(all_fuel_values)
                adaptive_threshold = np.percentile(all_fuel_array, 5)
                adaptive_threshold = max(0.001, adaptive_threshold)
            else:
                adaptive_threshold = 0.001
            
            # Find touching points
            touching_angles = []
            for angle in angles:
                x_pix = center_pixel_x + radius_pixels * np.cos(angle)
                y_pix = center_pixel_y + radius_pixels * np.sin(angle)
                
                if 0 <= y_pix < regressed_grid.shape[0] and 0 <= x_pix < regressed_grid.shape[1]:
                    fuel_value = f([[y_pix, x_pix]])[0]
                    if fuel_value >= adaptive_threshold:
                        touching_angles.append(angle)
            
            # Analyze contact points per extrusion (5 arms at 72° intervals)
            extrusion_contacts = [0] * 5
            arm_overlap_lengths = [0.0] * 5
            arc_length_per_point = (2 * np.pi * max_circle_radius) / num_samples
            
            for angle in touching_angles:
                arm_idx = int((angle * 180 / np.pi) / 72) % 5
                extrusion_contacts[arm_idx] += 1
                arm_overlap_lengths[arm_idx] += arc_length_per_point
            
            # Find the largest arm
            largest_arm_idx = np.argmax(extrusion_contacts) if extrusion_contacts else 0
            largest_arm_overlap = arm_overlap_lengths[largest_arm_idx]
            
            # Calculate total circle boundary overlap (same as circle overlap graph)
            total_circle_overlap = sum(arm_overlap_lengths)
            
            largest_arm_overlaps[i] = largest_arm_overlap
            valid_count = i + 1
            
            # Calculate 5% threshold based on current circle perimeter (same as circle overlap cutoff)
            circle_perimeter = 2 * np.pi * max_circle_radius
            threshold_5percent = 0.05 * circle_perimeter
            
            # Stop if total circle overlap falls below 5% of circle perimeter (same cutoff as circle overlap graph)
            if total_circle_overlap < threshold_5percent:
                print(f"  Circle overlap fell below 5% of circle perimeter ({threshold_5percent:.2f} mm) at regression distance {reg_dist:.4f} mm")
                break
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")
        
        # Trim arrays to valid data only
        regression_distances = regression_distances[:valid_count]
        largest_arm_overlaps = largest_arm_overlaps[:valid_count]
        regression_distances_plot = regression_distances_plot[:valid_count]
        
        # Remove zero overlap points
        valid_mask = largest_arm_overlaps > 0
        regression_distances = regression_distances[valid_mask]
        largest_arm_overlaps = largest_arm_overlaps[valid_mask]
        regression_distances_plot = regression_distances_plot[valid_mask]
        largest_arm_overlaps_plot = self._to_plot_length(largest_arm_overlaps)
        
        if len(regression_distances) < 3:
            print("Not enough data points for curve fitting - returning zero coefficients")
            return np.array([0, 0, 0])
        
        # Apply Savitzky-Golay filter to smooth noise
        if len(largest_arm_overlaps) >= 5:
            window_length = min(len(largest_arm_overlaps) - (1 if len(largest_arm_overlaps) % 2 == 0 else 0), 11)
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(5, window_length)
            overlaps_smooth = savgol_filter(largest_arm_overlaps, window_length=window_length, polyorder=3)
        else:
            overlaps_smooth = largest_arm_overlaps
        overlaps_smooth_plot = self._to_plot_length(overlaps_smooth)
        
        # Filter out invalid and near-zero data points before fitting
        # Keep only finite values and values above a minimum threshold (1% of max)
        valid_mask = np.isfinite(overlaps_smooth) & np.isfinite(regression_distances)
        if len(overlaps_smooth[valid_mask]) > 0:
            min_threshold = 0.01 * np.max(overlaps_smooth[valid_mask])
            valid_mask = valid_mask & (overlaps_smooth >= min_threshold)
        
        # Only proceed with fitting if we have enough data points
        if np.sum(valid_mask) < 3:
            print(f"WARNING: Not enough valid data points for polynomial fit (only {np.sum(valid_mask)})")
            # Use simple average or return dummy coefficients
            avg_overlap = np.nanmean(overlaps_smooth[np.isfinite(overlaps_smooth)])
            coeffs = np.array([0, 0, avg_overlap])  # Constant fit
            poly = np.poly1d(coeffs)
            fit = np.full_like(regression_distances, avg_overlap)
            r2 = 0
            degree_used = 0
        else:
            # Use filtered data for fitting
            x_filtered = regression_distances[valid_mask]
            y_filtered = overlaps_smooth[valid_mask]
            
            # Fit polynomial with adaptive degree based on R² threshold
            print(f"Fitting polynomial (target R² = 0.99) with {np.sum(valid_mask)} valid points...")
            coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
                x_filtered, y_filtered, target_r2=0.99, min_degree=2, max_degree=8
            )
            fit = poly(regression_distances)
        fit_plot = self._to_plot_length(fit)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances_plot, largest_arm_overlaps_plot, 'o-', linewidth=2, markersize=4,
               label='Raw Measurement', color='lightcoral', alpha=0.5)
        ax.plot(regression_distances_plot, overlaps_smooth_plot, 's-', linewidth=2.5, markersize=5,
               label='Smoothed Data', color='darkred', alpha=0.8)
        ax.plot(regression_distances_plot, fit_plot, '--', linewidth=2.5, color='gold',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel(self._length_label('Regression Distance'), fontsize=13, fontweight='bold')
        ax.set_ylabel(self._length_label('Largest Arm Contact Arc Length'), fontsize=13, fontweight='bold')
        ax.set_title(f'Largest Extrusion Arm Contact Length vs Regression Distance\n{Path(self.obj_file).name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='upper right')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        if len(largest_arm_overlaps) == 0:
            print("WARNING: No valid largest-arm data points found; skipping largest-arm plot.")
            return np.array([0, 0, 0])
        initial_overlap = largest_arm_overlaps[0]
        final_overlap = largest_arm_overlaps[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"L = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"L = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Arm Contact: {largest_arm_overlaps_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Final Arm Contact: {largest_arm_overlaps_plot[-1]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Contact Change: {largest_arm_overlaps_plot[-1] - largest_arm_overlaps_plot[0]:.2f} {self.plot_length_unit}\n'
        stats_text += f'Max Regression: {self._to_plot_length(max_regression_distance):.2f} {self.plot_length_unit}\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += self._format_polynomial_equation(coeffs, degree_used, 'L', self.plot_length_scale_mm, r2)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        self._present_plot(fig, "largest_arm_overlap_vs_regression")
        
        # Print summary
        print(f"\nLargest Arm Contact Length vs Regression Summary:")
        print(f"  Initial contact: {largest_arm_overlaps_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Final contact: {largest_arm_overlaps_plot[-1]:.2f} {self.plot_length_unit}")
        print(f"  Contact change: {largest_arm_overlaps_plot[-1] - largest_arm_overlaps_plot[0]:.2f} {self.plot_length_unit}")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        
        return coeffs


def main():
    """Main execution function"""
    # Configuration
    obj_file = r'C:\Users\gosha\Desktop\MotorModelP2\Hybrid Rocket model\goshastar.obj'  # Path to the OBJ file
    outer_diameter_inches = 6.74  # Will be overridden by actual geometry
    regression_rate = 3  # mm/sec
    time_seconds = 40  # seconds
    cross_section_axis = 1 # 0=X, 1=Y, 2=Z (Z is top-down view)
    plot_length_unit = 'mm'  # Change to 'cm', 'm', or 'in' for graph display only
    plot_area_unit = None  # Leave as None to derive area units from plot_length_unit
    export_regressed_obj_after_mm = None  # Example: 60 to export after 60 mm of regression
    export_regressed_obj_path = None # Example: r'C:\path\to\regressed.obj'
    
    # Check if OBJ file exists
    if not Path(obj_file).exists():
        print(f"OBJ file not found: {obj_file}")
        print("Please specify a valid OBJ file path")
        return
    
    # Create simulator and run
    simulator = FuelGrainRegressionSimulator(obj_file, outer_diameter_inches=outer_diameter_inches, 
                                            resolution=500, cross_section_axis=cross_section_axis,
                                            plot_length_unit=plot_length_unit, plot_area_unit=plot_area_unit)
    
    print("\n" + "="*60)
    print("Fuel Grain Regression Simulator (OBJ-based)")
    print("="*60)
    print(f"Configuration:")
    print(f"  OBJ File: {obj_file}")
    print(f"  Regression Rate: {regression_rate} mm/sec")
    print(f"  Simulation Time: {time_seconds} sec")
    print(f"  Total Regression: {regression_rate * time_seconds} mm")
    print("="*60 + "\n")
    
    # Run simulation to get initial geometry setup
    section_points_2d = simulator.read_obj_geometry()
    X, Y = simulator.create_initial_grid(section_points_2d)
    initial_center_bore_surface_area = simulator.calculate_center_bore_surface_area()
    simulator.full_3d_surface_area_mm2 = initial_center_bore_surface_area
    simulator.initial_bore_perimeter_mm = simulator.calculate_bore_perimeter(simulator.grid, X, Y)
    regression_distance = regression_rate * time_seconds
    regressed_grid, _ = simulator.fast_marching_method(regression_distance, X, Y)
    center_bore_surface_area = simulator._estimate_bore_surface_area_from_regression_distance(regression_distance)
    simulator.center_bore_surface_area_mm2 = center_bore_surface_area
    
    # Debug info
    print(f"\nGrid Statistics:")
    print(f"  Grid shape: {simulator.grid.shape}")
    print(f"  Grid min: {simulator.grid.min()}, max: {simulator.grid.max()}")
    print(f"  Non-zero cells: {np.sum(simulator.grid)}")
    print(f"  Grid fill percentage: {100 * np.sum(simulator.grid) / simulator.grid.size:.2f}%")
    
    # Calculate and display pixel size
    radius_with_margin = simulator.od_radius * 1.2
    grid_width = 2 * radius_with_margin
    mm_per_pixel = grid_width / simulator.resolution
    print(f"  Pixel size: {mm_per_pixel:.4f} mm/pixel")
    print(f"  Initial bore surface area: {initial_center_bore_surface_area:.2f} mm²")
    print(f"  Live bore surface area: {center_bore_surface_area:.2f} mm²")

    burnthrough_regression_mm = simulator.find_burnthrough_regression_distance(
        X,
        Y,
        max_regression_distance=max(simulator.id_radius, regression_distance),
    )
    if burnthrough_regression_mm is not None:
        print(f"  Burn-through regression distance: {burnthrough_regression_mm:.4f} mm")
    else:
        print("  Burn-through regression distance: not reached in tested range")

    if export_regressed_obj_after_mm is not None:
        print(f"\nExporting regressed OBJ after {export_regressed_obj_after_mm} mm of regression...")
        simulator.export_regressed_obj_after_distance(
            export_regressed_obj_after_mm,
            output_obj_path=export_regressed_obj_path,
        )
    
    # Display raw OBJ cross-section
    print("\nGenerating visualization 0: OBJ Cross-Section Geometry...")
    simulator.plot_dxf_geometry()
    
    # Display interactive regression slider
    print("Generating interactive regression visualization...")
    print("Use the slider to see how regression changes over time\n")
    simulator.plot_cross_section_interactive(regression_rate, time_seconds)
    
    # Display area vs regression graph
    print("Generating area vs regression graph...")
    simulator.plot_area_vs_regression(max_regression_distance=simulator.id_radius)

    # Display perimeter vs regression graph
    print("Generating perimeter vs regression graph...")
    simulator.plot_perimeter_vs_regression(max_regression_distance=simulator.id_radius)
    
    # Display max inscribed circle vs regression graph
    print("Generating max inscribed circle diameter vs regression graph...")
    simulator.plot_max_inscribed_circle_vs_regression(max_regression_distance=simulator.id_radius)
    
    # Display min enclosing circle vs regression graph
    print("Generating min enclosing circle diameter vs regression graph...")
    simulator.plot_min_enclosing_circle_vs_regression(max_regression_distance=simulator.id_radius)
    
    # Display circle boundary overlap vs regression graph
    print("Generating circle boundary overlap vs regression graph...")
    simulator.plot_circle_boundary_overlap_vs_regression(max_regression_distance=simulator.id_radius)
    
    # Display largest arm overlap vs regression graph
    print("Generating largest arm contact length vs regression graph...")
    simulator.plot_largest_arm_overlap_vs_regression(max_regression_distance=simulator.id_radius)

    if USE_GUI_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()
