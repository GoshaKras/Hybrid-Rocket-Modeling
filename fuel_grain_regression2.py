"""
Fuel Grain Regression Simulator using Fast Marching Method
Reads a 3D OBJ file and extracts a 2D cross-section for regression simulation.
Uses the Fast Marching Method to simulate regression from the inner boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.draw import polygon
from skimage import measure
import trimesh
import sys
from pathlib import Path

# Suppress numpy printing warnings
np.set_printoptions(threshold=10000)


class FuelGrainRegressionSimulator:
    """Simulates fuel grain regression using Fast Marching Method"""
    def __init__(self, obj_file_path, outer_diameter_inches=5.0, resolution=500, cross_section_axis=2, cross_section_pos=None):
        """
        Initialize the simulator.
        
        Args:
            obj_file_path: Path to the OBJ file containing the 3D fuel grain geometry
            outer_diameter_inches: Outer diameter of the fuel grain in inches
            resolution: Grid resolution for the simulation (pixels)
            cross_section_axis: Which axis to take cross-section perpendicular to (0=X, 1=Y, 2=Z)
            cross_section_pos: Position along axis for cross-section (None = center)
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
        
        regression_distance = regression_rate * time_seconds
        regressed_grid, new_id_radius = self.fast_marching_method(regression_distance)
        
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
            'time_seconds': time_seconds
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
                        
                        ax.fill(points_2d[:, 0], points_2d[:, 1], alpha=0.4, color='lightblue', edgecolor='blue', linewidth=2.5)
                        ax.plot(points_2d[:, 0], points_2d[:, 1], 'b-', linewidth=2.5)
                        all_points.extend(points_2d.tolist())
            
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
            ax.set_xlabel('Distance (mm)', fontsize=12)
            ax.set_ylabel('Distance (mm)', fontsize=12)
            ax.set_title(f'OBJ Cross-Section at {axis_names[self.cross_section_axis]}={cross_pos:.3f}', fontsize=12)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting OBJ geometry: {e}")
    
    def plot_results(self, results):
        """
        Visualize the fuel grain before and after regression.
        
        Args:
            results: Dictionary from simulate_regression
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initial geometry
        ax = axes[0]
        im1 = ax.contourf(results['X'], results['Y'], results['initial_grid'], 
                          levels=[0, 0.5, 1], colors=['white', 'lightblue'], alpha=0.8)
        ax.contour(results['X'], results['Y'], results['initial_grid'], 
                  levels=[0.5], colors=['blue'], linewidths=2)
        
        # Draw circles on initial
        circle_od = plt.Circle(self.center, results['od_radius'], 
                              fill=False, color='black', linewidth=2, label='OD (Fixed)')
        circle_id = plt.Circle(self.center, results['initial_id_radius'], 
                              fill=False, color='blue', linewidth=2, label='ID (Initial)')
        ax.add_patch(circle_od)
        ax.add_patch(circle_id)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(f'Initial Fuel Grain\nID: {results["initial_id_radius"]:.2f} mm, OD: {results["od_radius"]:.2f} mm')
        ax.legend(loc='upper right')
        
        # Regressed geometry
        ax = axes[1]
        im2 = ax.contourf(results['X'], results['Y'], results['regressed_grid'], 
                          levels=[0, 0.5, 1], colors=['white', 'lightsalmon'], alpha=0.8)
        ax.contour(results['X'], results['Y'], results['regressed_grid'], 
                  levels=[0.5], colors=['red'], linewidths=2)
        
        # Draw circles on regressed
        circle_od_new = plt.Circle(self.center, results['od_radius'], 
                                  fill=False, color='black', linewidth=2, label='OD (Fixed)')
        circle_id_new = plt.Circle(self.center, results['new_id_radius'], 
                                  fill=False, color='red', linewidth=2, label=f'ID (After {results["regression_distance"]:.2f} mm burn)')
        ax.add_patch(circle_od_new)
        ax.add_patch(circle_id_new)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(f'Fuel Grain After Regression\nID: {results["new_id_radius"]:.2f} mm, OD: {results["od_radius"]:.2f} mm')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
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
    
    def plot_cross_section_interactive(self, regression_rate, max_time):
        """
        Display regression over time with an interactive slider
        
        Args:
            regression_rate: Regression rate (mm/sec)
            max_time: Maximum time to show (seconds)
        """
        # Check if grid is populated
        if self.grid is None or np.sum(self.grid) == 0:
            print("Error: Grid is empty! Cannot display regression.")
            return
        
        # Set up the figure
        fig = plt.figure(figsize=(14, 10))
        ax_plot = plt.axes([0.15, 0.25, 0.7, 0.65])
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        
        # Add spacing for title
        plt.subplots_adjust(top=0.92)
        
        # Create slider for time
        slider = Slider(ax_slider, 'Time (sec)', 0, max_time, valinit=0, color='steelblue')
        
        # Create meshgrid once
        radius_with_margin = self.od_radius * 1.2
        x = np.linspace(self.center[0] - radius_with_margin, self.center[0] + radius_with_margin, self.resolution)
        y = np.linspace(self.center[1] - radius_with_margin, self.center[1] + radius_with_margin, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        def update(val):
            """Update visualization"""
            ax_plot.clear()
            
            time_val = slider.val
            regression_distance = regression_rate * time_val
            
            # Get regressed grid
            regressed_grid, new_id_radius = self.fast_marching_method(regression_distance, X, Y)
            
            # Calculate area and perimeter
            pixel_width = (X[0, 1] - X[0, 0])
            pixel_height = (Y[1, 0] - Y[0, 0])
            pixel_area = abs(pixel_width * pixel_height)
            
            # Area of fuel grain (in mm²)
            fuel_area = np.sum(regressed_grid) * pixel_area
            
            # ID Perimeter - calculate actual boundary of the inner star-shaped hole
            from skimage import measure
            from scipy import ndimage
            
            # Find the inner hole boundary
            empty_space = 1 - regressed_grid
            labeled, num_features = ndimage.label(empty_space)
            border_label = labeled[0, 0]
            
            # Find the largest inner hole (excluding the outer border)
            inner_perimeter = 0
            max_inscribed_diameter = 0
            max_circle_center = None
            max_circle_radius = 0
            min_enclosing_diameter = 0
            min_enclosing_center = None
            min_enclosing_radius = 0
            fuel_perimeter = 0
            overlap_length_mm = 0
            
            largest_hole_size = 0
            largest_hole_label = None
            
            for label in range(1, num_features + 1):
                if label != border_label:
                    hole_size = np.sum(labeled == label)
                    if hole_size > largest_hole_size:
                        largest_hole_size = hole_size
                        largest_hole_label = label
            
            # Process only the largest hole
            if largest_hole_label is not None:
                inner_hole_mask = (labeled == largest_hole_label)
                
                if np.sum(inner_hole_mask) > 0:
                    # Calculate inner hole perimeter using neighborhood=4 with 0.95 correction
                    perimeter_pixels = measure.perimeter(inner_hole_mask, neighborhood=4)
                    inner_perimeter = perimeter_pixels * pixel_width * 0.95
                    
                    # Calculate fuel grain perimeter (outer boundary)
                    # Use neighborhood=4 with 0.95 correction factor for 5% overestimation
                    fuel_perimeter_pixels = measure.perimeter(regressed_grid, neighborhood=4)
                    fuel_perimeter = fuel_perimeter_pixels * pixel_width * 0.95
                    
                    # Find the largest circle that fits inside the hole
                    # Use distance transform - max value is the radius of largest inscribed circle
                    distance_from_boundary = distance_transform_edt(inner_hole_mask)
                    max_radius_pixels = np.max(distance_from_boundary)
                    max_inscribed_diameter = 2 * max_radius_pixels * pixel_width
                    max_circle_radius = max_radius_pixels * pixel_width
                    
                    print(f"Largest hole: perimeter={inner_perimeter:.2f} mm, inscribed_radius={max_circle_radius:.2f} mm")
                    
                    # Find center of largest inscribed circle (point with max distance)
                    center_pixel = np.unravel_index(np.argmax(distance_from_boundary), distance_from_boundary.shape)
                    # Convert pixel coordinates to world coordinates
                    max_circle_center = (X[0, 0] + center_pixel[1] * pixel_width,
                                        Y[0, 0] + center_pixel[0] * pixel_height)
                    
                    # Find the smallest circle that encompasses the entire hole boundary (minimum enclosing circle)
                    contours = measure.find_contours(inner_hole_mask, 0.5)
                    if contours and len(contours) > 0:
                        boundary_pixels = contours[0]
                        # Convert to world coordinates
                        boundary_points = np.column_stack([
                            X[0, 0] + boundary_pixels[:, 1] * pixel_width,
                            Y[0, 0] + boundary_pixels[:, 0] * pixel_height
                        ])
                        
                        # Find minimum enclosing circle using centroid + max distance
                        min_enclosing_center = boundary_points.mean(axis=0)
                        min_enclosing_radius = np.max(np.linalg.norm(boundary_points - min_enclosing_center, axis=1))
                        min_enclosing_diameter = 2 * min_enclosing_radius
                    else:
                        # Fallback if no contours found
                        min_enclosing_center = max_circle_center
                        min_enclosing_radius = max_circle_radius
                        min_enclosing_diameter = max_inscribed_diameter
                    
                    # Calculate circle boundary overlap (arc length where green circle touches red boundary)
                    circle_overlap = self.calculate_inscribed_circle_boundary_overlap(
                        regressed_grid, max_circle_center, max_circle_radius, pixel_width, X, Y
                    )
                    overlap_length_mm = circle_overlap
            else:
                # No significant holes found - set defaults
                inner_perimeter = 0
                max_inscribed_diameter = 0
                max_circle_center = None
                max_circle_radius = 0
                min_enclosing_diameter = 0
                min_enclosing_center = None
                min_enclosing_radius = 0
                fuel_perimeter = 0
                overlap_length_mm = 0
            
            # Plot initial shape - filled
            ax_plot.imshow(self.grid, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                          origin='lower', cmap='Blues', alpha=0.4)
            
            # Plot initial shape - contour
            ax_plot.contour(X, Y, self.grid, levels=[0.5], colors=['blue'], 
                           linewidths=2.5, linestyles='--')
            
            # Plot regressed shape - filled
            ax_plot.imshow(regressed_grid, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                          origin='lower', cmap='Reds', alpha=0.4)
            
            # Plot regressed shape - contour
            ax_plot.contour(X, Y, regressed_grid, levels=[0.5], colors=['red'], 
                           linewidths=2.5)
            
            # Draw the largest inscribed circle if it exists
            if max_circle_center is not None and max_circle_radius > 0:
                inscribed_circle = plt.Circle(max_circle_center, max_circle_radius, 
                                             fill=False, color='green', linewidth=2.5, 
                                             linestyle=':', label='Max Inscribed Circle')
                ax_plot.add_patch(inscribed_circle)
            
            # Draw the minimum enclosing circle if it exists
            if min_enclosing_center is not None and min_enclosing_radius > 0:
                enclosing_circle = plt.Circle(min_enclosing_center, min_enclosing_radius, 
                                             fill=False, color='orange', linewidth=2, 
                                             linestyle='-.', label='Min Enclosing Circle')
                ax_plot.add_patch(enclosing_circle)
            
            # Visualize which points on the green circle are touching fuel
            if max_circle_center is not None and max_circle_radius > 0:
                from scipy import interpolate
                
                # Create interpolation function for fuel grid
                grid_y = np.arange(regressed_grid.shape[0])
                grid_x = np.arange(regressed_grid.shape[1])
                f = interpolate.RegularGridInterpolator((grid_y, grid_x), regressed_grid, bounds_error=False, fill_value=0)
                
                # Sample circle perimeter
                pixel_width = (X[0, 1] - X[0, 0])
                pixel_height = (Y[1, 0] - Y[0, 0])
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
                    
                    if len(all_fuel_array) > 0:
                        # Use 5th percentile of all detected fuel values (extremely aggressive - catches all weak signals)
                        adaptive_threshold = np.percentile(all_fuel_array, 5)
                        # Ensure minimum threshold to avoid pure zeros
                        adaptive_threshold = max(0.001, adaptive_threshold)
                    else:
                        adaptive_threshold = 0.001
                    
                    # Find touching points and plot them
                    touching_x = []
                    touching_y = []
                    touching_angles = []
                    for angle in angles:
                        x_pix = center_pixel_x + radius_pixels * np.cos(angle)
                        y_pix = center_pixel_y + radius_pixels * np.sin(angle)
                        
                        if 0 <= y_pix < regressed_grid.shape[0] and 0 <= x_pix < regressed_grid.shape[1]:
                            fuel_value = f([[y_pix, x_pix]])[0]
                            if fuel_value > adaptive_threshold:  # Use adaptive threshold
                                x_world = X[0, 0] + x_pix * pixel_width
                                y_world = Y[0, 0] + y_pix * pixel_height
                                touching_x.append(x_world)
                                touching_y.append(y_world)
                                touching_angles.append(angle)
                    
                    # Debug: Analyze contact points per extrusion (5 arms at 72° intervals)
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
                    
                    print(f"  Threshold: {adaptive_threshold:.6f} | Total: {len(touching_x)}/{len(angles)} | Arms: {extrusion_contacts} | Largest Arm {largest_arm_idx}: {largest_arm_overlap:.2f} mm")
                    
                    # Plot touching points in cyan with larger size for visibility
                    if touching_x:
                        ax_plot.plot(touching_x, touching_y, 'c.', markersize=5, alpha=0.8, label='Contact Points')
            
            # Create manual legend entries
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='Initial'),
                Line2D([0], [0], color='red', linewidth=2.5, linestyle='-', label='After Regression'),
                Line2D([0], [0], color='green', linewidth=2.5, linestyle=':', label='Max Inscribed Circle'),
                Line2D([0], [0], color='orange', linewidth=2, linestyle='-.', label='Min Enclosing Circle'),
                Line2D([0], [0], color='cyan', marker='.', linestyle='None', markersize=8, label='Contact Points')
            ]
            
            ax_plot.set_aspect('equal')
            ax_plot.grid(True, alpha=0.3)
            ax_plot.set_xlabel('Distance (mm)', fontsize=12)
            ax_plot.set_ylabel('Distance (mm)', fontsize=12)
            
            # Build title with largest arm info
            largest_arm_text = f' | Largest Arm: {largest_arm_overlap:.2f} mm' if largest_arm_overlap > 0 else ''
            ax_plot.set_title(f'Fuel Grain Regression\nTime: {time_val:.2f} sec | Regression: {regression_distance:.4f} mm\nArea: {fuel_area:.2f} mm² | Hole Perimeter: {inner_perimeter:.2f} mm | Max Circle: {max_inscribed_diameter:.2f} mm | Circle-Boundary Touch: {overlap_length_mm:.2f} mm{largest_arm_text}', 
                             fontsize=11, fontweight='bold')
            ax_plot.legend(handles=legend_elements, loc='upper right', fontsize=11)
            
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        # Initial plot
        update(0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_area_vs_regression(self, max_regression_distance=None):
        """
        Plot fuel grain area vs regression distance with polynomial fit and R² value.
        
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
        pixel_area = abs(pixel_width * pixel_height)
        
        # Determine max regression distance
        if max_regression_distance is None:
            max_regression_distance = self.id_radius
        
        # Create array of regression distances
        num_points = 100
        regression_distances = np.linspace(0, max_regression_distance, num_points)
        
        # Array to store results
        areas = np.zeros(num_points)
        
        print(f"Calculating area vs regression (this may take a moment)...")
        
        # Calculate areas for each regression distance
        valid_count = 0
        for i, reg_dist in enumerate(regression_distances):
            regressed_grid, _ = self.fast_marching_method(reg_dist, X, Y)
            areas[i] = np.sum(regressed_grid) * pixel_area
            valid_count = i + 1
            
            # Stop if fuel area reaches 0
            if areas[i] <= 0:
                print(f"  Fuel completely burned at regression distance {reg_dist:.4f} mm")
                break
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}")
        
        # Trim arrays to valid data only
        regression_distances = regression_distances[:valid_count]
        areas = areas[:valid_count]
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, areas, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances, areas, 'o-', linewidth=3, markersize=6, 
               label='Fuel Area', color='steelblue', alpha=0.7)
        ax.plot(regression_distances, fit, '--', linewidth=2.5, color='coral', 
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel('Regression Distance (mm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fuel Grain Area (mm²)', fontsize=13, fontweight='bold')
        ax.set_title(f'Fuel Grain Area vs Regression Distance\n{Path(self.obj_file).name}', 
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
        
        stats_text = f'Initial Area: {initial_area:.2f} mm²\n'
        stats_text += f'Final Area: {final_area:.2f} mm²\n'
        stats_text += f'Area Burned: {initial_area - final_area:.2f} mm²\n'
        stats_text += f'Max Regression: {max_regression_distance:.2f} mm\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += eq
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nArea vs Regression Summary:")
        print(f"  Initial area: {initial_area:.2f} mm²")
        print(f"  Final area: {final_area:.2f} mm²")
        print(f"  Area burned: {initial_area - final_area:.2f} mm²")
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
        
        # Remove zero perimeter points
        valid_mask = perimeters > 0
        regression_distances = regression_distances[valid_mask]
        perimeters = perimeters[valid_mask]
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, perimeters, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances, perimeters, 'o-', linewidth=3, markersize=6,
               label='Inner Hole Perimeter', color='darkgreen', alpha=0.7)
        ax.plot(regression_distances, fit, '--', linewidth=2.5, color='orange',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel('Regression Distance (mm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Inner Hole Perimeter (mm)', fontsize=13, fontweight='bold')
        ax.set_title(f'Inner Hole Perimeter vs Regression Distance\n{Path(self.obj_file).name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        initial_perimeter = perimeters[0]
        final_perimeter = perimeters[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"P = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"P = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Perimeter: {initial_perimeter:.2f} mm\n'
        stats_text += f'Final Perimeter: {final_perimeter:.2f} mm\n'
        stats_text += f'Perimeter Change: {final_perimeter - initial_perimeter:.2f} mm\n'
        stats_text += f'Max Regression: {max_regression_distance:.2f} mm\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += eq
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nPerimeter vs Regression Summary:")
        print(f"  Initial perimeter: {initial_perimeter:.2f} mm")
        print(f"  Final perimeter: {final_perimeter:.2f} mm")
        print(f"  Perimeter change: {final_perimeter - initial_perimeter:.2f} mm")
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
        
        # Remove zero diameter points
        valid_mask = max_inscribed_diameters > 0
        regression_distances = regression_distances[valid_mask]
        max_inscribed_diameters = max_inscribed_diameters[valid_mask]
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, max_inscribed_diameters, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances, max_inscribed_diameters, 'o-', linewidth=3, markersize=6,
               label='Max Inscribed Circle Diameter', color='darkblue', alpha=0.7)
        ax.plot(regression_distances, fit, '--', linewidth=2.5, color='cyan',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel('Regression Distance (mm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Max Inscribed Circle Diameter (mm)', fontsize=13, fontweight='bold')
        ax.set_title(f'Max Inscribed Circle Diameter vs Regression Distance\n{Path(self.obj_file).name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        initial_max_inscribed = max_inscribed_diameters[0]
        final_max_inscribed = max_inscribed_diameters[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"D = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"D = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Max Inscribed: {initial_max_inscribed:.2f} mm\n'
        stats_text += f'Final Max Inscribed: {final_max_inscribed:.2f} mm\n'
        stats_text += f'Diameter Change: {final_max_inscribed - initial_max_inscribed:.2f} mm\n'
        stats_text += f'Max Regression: {max_regression_distance:.2f} mm\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += eq
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nMax Inscribed Circle vs Regression Summary:")
        print(f"  Initial diameter: {initial_max_inscribed:.2f} mm")
        print(f"  Final diameter: {final_max_inscribed:.2f} mm")
        print(f"  Diameter change: {final_max_inscribed - initial_max_inscribed:.2f} mm")
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
        
        # Remove zero diameter points
        valid_mask = min_enclosing_diameters > 0
        regression_distances = regression_distances[valid_mask]
        min_enclosing_diameters = min_enclosing_diameters[valid_mask]
        
        # Fit polynomial with adaptive degree based on R² threshold
        print(f"Fitting polynomial (target R² = 0.99)...")
        coeffs, poly, r2, degree_used = self.fit_polynomial_with_r2_threshold(
            regression_distances, min_enclosing_diameters, target_r2=0.99, min_degree=2, max_degree=8
        )
        fit = poly(regression_distances)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances, min_enclosing_diameters, 'o-', linewidth=3, markersize=6,
               label='Min Enclosing Circle Diameter', color='darkgreen', alpha=0.7)
        ax.plot(regression_distances, fit, '--', linewidth=2.5, color='lightgreen',
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel('Regression Distance (mm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Min Enclosing Circle Diameter (mm)', fontsize=13, fontweight='bold')
        ax.set_title(f'Min Enclosing Circle Diameter vs Regression Distance\n{Path(self.obj_file).name}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        initial_min_enclosing = min_enclosing_diameters[0]
        final_min_enclosing = min_enclosing_diameters[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"D = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"D = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Min Enclosing: {initial_min_enclosing:.2f} mm\n'
        stats_text += f'Final Min Enclosing: {final_min_enclosing:.2f} mm\n'
        stats_text += f'Diameter Change: {final_min_enclosing - initial_min_enclosing:.2f} mm\n'
        stats_text += f'Max Regression: {max_regression_distance:.2f} mm\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += eq
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nMin Enclosing Circle vs Regression Summary:")
        print(f"  Initial diameter: {initial_min_enclosing:.2f} mm")
        print(f"  Final diameter: {final_min_enclosing:.2f} mm")
        print(f"  Diameter change: {final_min_enclosing - initial_min_enclosing:.2f} mm")
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
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances, overlaps, 'o-', linewidth=2, markersize=4, 
               label='Raw Measurement', color='lightblue', alpha=0.5)
        ax.plot(regression_distances, overlaps_smooth, 's-', linewidth=2.5, markersize=5,
               label='Smoothed Data', color='darkblue', alpha=0.8)
        ax.plot(regression_distances, fit, '--', linewidth=2.5, color='cyan', 
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel('Regression Distance (mm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Circle Boundary Overlap (mm)', fontsize=13, fontweight='bold')
        ax.set_title(f'Inscribed Circle Boundary Overlap vs Regression Distance\n{Path(self.obj_file).name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='upper right')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        initial_overlap = overlaps[0]
        final_overlap = overlaps[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"L = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"L = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Overlap: {initial_overlap:.2f} mm\n'
        stats_text += f'Final Overlap: {final_overlap:.2f} mm\n'
        stats_text += f'Overlap Change: {final_overlap - initial_overlap:.2f} mm\n'
        stats_text += f'Max Regression: {max_regression_distance:.2f} mm\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += eq
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nCircle Boundary Overlap vs Regression Summary:")
        print(f"  Initial overlap: {initial_overlap:.2f} mm")
        print(f"  Final overlap: {final_overlap:.2f} mm")
        print(f"  Overlap change: {final_overlap - initial_overlap:.2f} mm")
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
        
        # Remove zero overlap points
        valid_mask = largest_arm_overlaps > 0
        regression_distances = regression_distances[valid_mask]
        largest_arm_overlaps = largest_arm_overlaps[valid_mask]
        
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
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(regression_distances, largest_arm_overlaps, 'o-', linewidth=2, markersize=4, 
               label='Raw Measurement', color='lightcoral', alpha=0.5)
        ax.plot(regression_distances, overlaps_smooth, 's-', linewidth=2.5, markersize=5,
               label='Smoothed Data', color='darkred', alpha=0.8)
        ax.plot(regression_distances, fit, '--', linewidth=2.5, color='gold', 
               alpha=0.8, label=f'Polynomial Fit (degree {degree_used})')
        
        ax.set_xlabel('Regression Distance (mm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Largest Arm Contact Arc Length (mm)', fontsize=13, fontweight='bold')
        ax.set_title(f'Largest Extrusion Arm Contact Length vs Regression Distance\n{Path(self.obj_file).name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='upper right')
        
        # Format the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics to the plot
        initial_overlap = largest_arm_overlaps[0]
        final_overlap = largest_arm_overlaps[-1]
        
        # Format equation based on degree
        if degree_used == 2:
            eq = f"L = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.2f}\nR² = {r2:.6f}"
        elif degree_used == 3:
            eq = f"L = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.2f}\nR² = {r2:.6f}"
        else:
            eq = f"Degree {degree_used} polynomial\nR² = {r2:.6f}"
        
        stats_text = f'Initial Arm Contact: {initial_overlap:.2f} mm\n'
        stats_text += f'Final Arm Contact: {final_overlap:.2f} mm\n'
        stats_text += f'Contact Change: {final_overlap - initial_overlap:.2f} mm\n'
        stats_text += f'Max Regression: {max_regression_distance:.2f} mm\n'
        stats_text += f'Polynomial Degree: {degree_used}\n\n'
        stats_text += eq
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.85),
               family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nLargest Arm Contact Length vs Regression Summary:")
        print(f"  Initial contact: {initial_overlap:.2f} mm")
        print(f"  Final contact: {final_overlap:.2f} mm")
        print(f"  Contact change: {final_overlap - initial_overlap:.2f} mm")
        print(f"  Polynomial degree: {degree_used}")
        print(f"  R² (goodness of fit): {r2:.6f}")
        
        return coeffs


def main():
    """Main execution function"""
    # Configuration
    current_dir = Path(__file__).resolve().parent
    obj_file = current_dir / "goddard.obj"
    # obj_file = r'C:\Users\gosha\Desktop\MotorModelP2\cads\goddard.obj'
    outer_diameter_inches = 6.74  # Will be overridden by actual geometry
    regression_rate = 3  # mm/sec
    time_seconds = 40  # seconds
    cross_section_axis = 2  # 0=X, 1=Y, 2=Z (Z is top-down view)
    
    # Check if OBJ file exists
    if not Path(obj_file).exists():
        print(f"OBJ file not found: {obj_file}")
        print("Please specify a valid OBJ file path")
        return
    
    # Create simulator and run
    simulator = FuelGrainRegressionSimulator(obj_file, outer_diameter_inches=outer_diameter_inches, 
                                            resolution=500, cross_section_axis=cross_section_axis)
    
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


if __name__ == '__main__':
    main()
