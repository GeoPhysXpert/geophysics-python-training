import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class GravityAnomalyCalculator:
    """
    A class for calculating gravity anomalies from various subsurface bodies.
    
    This class provides methods to calculate gravity anomalies for:
    - Spherical bodies
    - Horizontal cylindrical bodies  
    - Rectangular prismatic bodies
    """
    
    def __init__(self):
        """Initialize the gravity anomaly calculator."""
        self.G = 6.674e-11  # Universal gravitational constant (m³/kg·s²)
        self.body_types = {
            '1': 'Sphere',
            '2': 'Horizontal Cylinder', 
            '3': 'Rectangular Prism'
        }
    
    def sphere_anomaly(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Calculate gravity anomaly for a spherical body.
        
        Args:
            x: Array of horizontal distances (m)
            params: Dictionary containing sphere parameters
                   - center_x: Horizontal position of sphere center (m)
                   - depth: Depth to sphere center (m)
                   - radius: Sphere radius (m)
                   - density_contrast: Density contrast (kg/m³)
        
        Returns:
            Array of gravity anomaly values (mGal)
        """
        center_x = params['center_x']
        depth = params['depth']
        radius = params['radius']
        rho = params['density_contrast']
        
        # Volume of sphere
        volume = (4/3) * np.pi * radius**3
        
        # Distance from observation points to sphere center
        r = np.sqrt((x - center_x)**2 + depth**2)
        
        # Gravity anomaly (vertical component)
        g_z = self.G * rho * volume * depth / (r**3)
        
        # Convert to mGal (1 m/s² = 10⁵ mGal)
        return g_z * 1e5
    
    def cylinder_anomaly(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Calculate gravity anomaly for a horizontal cylindrical body.
        
        Args:
            x: Array of horizontal distances (m)
            params: Dictionary containing cylinder parameters
                   - center_x: Horizontal position of cylinder center (m)
                   - depth: Depth to cylinder center (m)
                   - radius: Cylinder radius (m)
                   - density_contrast: Density contrast (kg/m³)
        
        Returns:
            Array of gravity anomaly values (mGal)
        """
        center_x = params['center_x']
        depth = params['depth']
        radius = params['radius']
        rho = params['density_contrast']
        
        # Distance from observation points to cylinder center
        r_squared = (x - center_x)**2 + depth**2
        
        # Gravity anomaly for infinite horizontal cylinder
        g_z = 2 * self.G * rho * np.pi * radius**2 * depth / r_squared
        
        # Convert to mGal
        return g_z * 1e5
    
    def prism_anomaly(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Calculate gravity anomaly for a rectangular prism.
        
        Args:
            x: Array of horizontal distances (m)
            params: Dictionary containing prism parameters
                   - center_x: Horizontal position of prism center (m)
                   - depth_top: Depth to top of prism (m)
                   - depth_bottom: Depth to bottom of prism (m)
                   - half_width: Half-width of prism (m)
                   - density_contrast: Density contrast (kg/m³)
        
        Returns:
            Array of gravity anomaly values (mGal)
        """
        center_x = params['center_x']
        z1 = params['depth_top']
        z2 = params['depth_bottom']
        a = params['half_width']
        rho = params['density_contrast']
        
        # Prism boundaries
        x1 = center_x - a
        x2 = center_x + a
        
        g_z = np.zeros_like(x)
        
        for i, xi in enumerate(x):
            # Calculate gravity anomaly using analytical formula
            term1 = self._prism_term(xi, x2, z2) - self._prism_term(xi, x1, z2)
            term2 = self._prism_term(xi, x2, z1) - self._prism_term(xi, x1, z1)
            g_z[i] = 2 * self.G * rho * (term1 - term2)
        
        # Convert to mGal
        return g_z * 1e5
    
    def _prism_term(self, x: float, xi: float, z: float) -> float:
        """
        Helper function for rectangular prism calculation.
        
        Args:
            x: Observation point
            xi: Prism boundary
            z: Depth
        
        Returns:
            Calculated term for prism formula
        """
        r = np.sqrt((x - xi)**2 + z**2)
        if r == 0:
            return 0
        return (x - xi) * np.log(z + r) - z * np.arctan((x - xi) / z)
    
    def get_user_input(self) -> Tuple[str, Dict[str, float], Tuple[float, float, int]]:
        """
        Get user input for body type, parameters, and survey configuration.
        
        Returns:
            Tuple containing body type, parameters dictionary, and survey config
        """
        print("\n" + "="*60)
        print("         GRAVITY ANOMALY CALCULATOR")
        print("="*60)
        print("\nAvailable subsurface body types:")
        for key, value in self.body_types.items():
            print(f"{key}. {value}")
        
        # Get body type selection
        while True:
            choice = input("\nSelect body type (1-3): ").strip()
            if choice in self.body_types:
                body_type = self.body_types[choice]
                break
            print("Invalid choice. Please select 1, 2, or 3.")
        
        print(f"\nSelected: {body_type}")
        print("-" * 40)
        
        # Get parameters based on body type
        params = {}
        
        if body_type == 'Sphere':
            params = self._get_sphere_params()
        elif body_type == 'Horizontal Cylinder':
            params = self._get_cylinder_params()
        elif body_type == 'Rectangular Prism':
            params = self._get_prism_params()
        
        # Get survey configuration
        survey_config = self._get_survey_config()
        
        return body_type, params, survey_config
    
    def _get_sphere_params(self) -> Dict[str, float]:
        """Get parameters specific to spherical body."""
        print("Enter sphere parameters:")
        params = {}
        params['center_x'] = float(input("Horizontal position of center (m): "))
        params['depth'] = float(input("Depth to center (m): "))
        params['radius'] = float(input("Radius (m): "))
        params['density_contrast'] = float(input("Density contrast (kg/m³): "))
        return params
    
    def _get_cylinder_params(self) -> Dict[str, float]:
        """Get parameters specific to cylindrical body."""
        print("Enter horizontal cylinder parameters:")
        params = {}
        params['center_x'] = float(input("Horizontal position of center (m): "))
        params['depth'] = float(input("Depth to center (m): "))
        params['radius'] = float(input("Radius (m): "))
        params['density_contrast'] = float(input("Density contrast (kg/m³): "))
        return params
    
    def _get_prism_params(self) -> Dict[str, float]:
        """Get parameters specific to rectangular prism."""
        print("Enter rectangular prism parameters:")
        params = {}
        params['center_x'] = float(input("Horizontal position of center (m): "))
        params['depth_top'] = float(input("Depth to top (m): "))
        params['depth_bottom'] = float(input("Depth to bottom (m): "))
        params['half_width'] = float(input("Half-width (m): "))
        params['density_contrast'] = float(input("Density contrast (kg/m³): "))
        return params
    
    def _get_survey_config(self) -> Tuple[float, float, int]:
        """Get survey line configuration."""
        print("\nEnter survey line configuration:")
        x_start = float(input("Start position (m): "))
        x_end = float(input("End position (m): "))
        n_points = int(input("Number of observation points: "))
        return x_start, x_end, n_points
    
    def calculate_anomaly(self, body_type: str, x: np.ndarray, 
                         params: Dict[str, float]) -> np.ndarray:
        """
        Calculate gravity anomaly based on body type.
        
        Args:
            body_type: Type of subsurface body
            x: Array of observation points
            params: Parameters for the body
        
        Returns:
            Array of gravity anomaly values
        """
        if body_type == 'Sphere':
            return self.sphere_anomaly(x, params)
        elif body_type == 'Horizontal Cylinder':
            return self.cylinder_anomaly(x, params)
        elif body_type == 'Rectangular Prism':
            return self.prism_anomaly(x, params)
        else:
            raise ValueError(f"Unknown body type: {body_type}")
    
    def plot_anomaly(self, x: np.ndarray, g_z: np.ndarray, body_type: str, 
                    params: Dict[str, float]) -> None:
        """
        Create a professional plot of the gravity anomaly.
        
        Args:
            x: Array of observation points
            g_z: Array of gravity anomaly values
            body_type: Type of subsurface body
            params: Parameters used for calculation
        """
        plt.figure(figsize=(12, 8))
        
        # Main anomaly plot
        plt.subplot(2, 1, 1)
        plt.plot(x, g_z, 'b-', linewidth=2, label='Gravity Anomaly')
        plt.fill_between(x, g_z, alpha=0.3, color='lightblue')
        plt.xlabel('Distance (m)')
        plt.ylabel('Gravity Anomaly (mGal)')
        plt.title(f'Gravity Anomaly - {body_type}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Cross-section sketch
        plt.subplot(2, 1, 2)
        self._plot_cross_section(body_type, params)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_cross_section(self, body_type: str, params: Dict[str, float]) -> None:
        """Plot a cross-sectional view of the subsurface body."""
        center_x = params['center_x']
        
        if body_type == 'Sphere':
            depth = params['depth']
            radius = params['radius']
            
            # Draw sphere
            circle = plt.Circle((center_x, -depth), radius, 
                              fill=True, color='red', alpha=0.7, 
                              label='Sphere')
            plt.gca().add_patch(circle)
            plt.xlim(center_x - 3*radius, center_x + 3*radius)
            plt.ylim(-depth - 2*radius, 10)
            
        elif body_type == 'Horizontal Cylinder':
            depth = params['depth']
            radius = params['radius']
            
            # Draw cylinder cross-section (circle)
            circle = plt.Circle((center_x, -depth), radius, 
                              fill=True, color='green', alpha=0.7,
                              label='Cylinder')
            plt.gca().add_patch(circle)
            plt.xlim(center_x - 3*radius, center_x + 3*radius)
            plt.ylim(-depth - 2*radius, 10)
            
        elif body_type == 'Rectangular Prism':
            depth_top = params['depth_top']
            depth_bottom = params['depth_bottom']
            half_width = params['half_width']
            
            # Draw rectangle
            rect = plt.Rectangle((center_x - half_width, -depth_bottom), 
                               2*half_width, depth_bottom - depth_top,
                               fill=True, color='orange', alpha=0.7,
                               label='Prism')
            plt.gca().add_patch(rect)
            plt.xlim(center_x - 3*half_width, center_x + 3*half_width)
            plt.ylim(-depth_bottom - half_width, 10)
        
        # Draw surface
        x_surf = plt.xlim()
        plt.plot(x_surf, [0, 0], 'k-', linewidth=3, label='Surface')
        
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        plt.title('Cross-Section View')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().set_aspect('equal')
    
    def display_summary(self, body_type: str, params: Dict[str, float], 
                       x: np.ndarray, g_z: np.ndarray) -> None:
        """
        Display a summary of the calculation results.
        
        Args:
            body_type: Type of subsurface body
            params: Parameters used
            x: Observation points
            g_z: Calculated anomaly values
        """
        print("\n" + "="*60)
        print("                CALCULATION SUMMARY")
        print("="*60)
        print(f"Body Type: {body_type}")
        print(f"Density Contrast: {params['density_contrast']:.2f} kg/m³")
        
        if body_type in ['Sphere', 'Horizontal Cylinder']:
            print(f"Center Position: {params['center_x']:.2f} m")
            print(f"Depth: {params['depth']:.2f} m")
            print(f"Radius: {params['radius']:.2f} m")
        elif body_type == 'Rectangular Prism':
            print(f"Center Position: {params['center_x']:.2f} m")
            print(f"Top Depth: {params['depth_top']:.2f} m")
            print(f"Bottom Depth: {params['depth_bottom']:.2f} m")
            print(f"Half-width: {params['half_width']:.2f} m")
        
        print(f"\nSurvey Configuration:")
        print(f"Distance Range: {x[0]:.2f} to {x[-1]:.2f} m")
        print(f"Number of Points: {len(x)}")
        
        print(f"\nAnomaly Statistics:")
        print(f"Maximum Anomaly: {np.max(g_z):.4f} mGal")
        print(f"Minimum Anomaly: {np.min(g_z):.4f} mGal")
        print(f"Peak-to-Peak: {np.max(g_z) - np.min(g_z):.4f} mGal")
        print("="*60)
    
    def run(self) -> None:
        """Main execution function."""
        try:
            # Get user input
            body_type, params, survey_config = self.get_user_input()
            
            # Create observation points
            x_start, x_end, n_points = survey_config
            x = np.linspace(x_start, x_end, n_points)
            
            # Calculate gravity anomaly
            print("\nCalculating gravity anomaly...")
            g_z = self.calculate_anomaly(body_type, x, params)
            
            # Display results
            self.display_summary(body_type, params, x, g_z)
            
            # Plot results
            print("\nGenerating plot...")
            self.plot_anomaly(x, g_z, body_type, params)
            
            # Option to save data
            save_data = input("\nSave data to file? (y/n): ").lower().strip()
            if save_data == 'y':
                filename = input("Enter filename (without extension): ").strip()
                self.save_data(filename, x, g_z, body_type, params)
                print(f"Data saved to {filename}.txt")
            
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your inputs and try again.")
    
    def save_data(self, filename: str, x: np.ndarray, g_z: np.ndarray, 
                  body_type: str, params: Dict[str, float]) -> None:
        """
        Save calculated data to a text file.
        
        Args:
            filename: Output filename (without extension)
            x: Observation points
            g_z: Gravity anomaly values
            body_type: Type of body
            params: Parameters used
        """
        with open(f"{filename}.txt", 'w') as f:
            f.write("# Gravity Anomaly Calculation Results\n")
            f.write(f"# Body Type: {body_type}\n")
            f.write(f"# Parameters: {params}\n")
            f.write("# Distance(m)\tGravity_Anomaly(mGal)\n")
            for xi, gi in zip(x, g_z):
                f.write(f"{xi:.6f}\t{gi:.6f}\n")


def main():
    """Main function to run the gravity anomaly calculator."""
    calculator = GravityAnomalyCalculator()
    calculator.run()


if __name__ == "__main__":
    main()