import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class SeismicRefractionModel:
    """
    Comprehensive seismic refraction modeling tool implementing:
    - Layered Earth setup
    - Raypath logic with Snell's Law
    - Travel time calculations
    - t-x curve plotting
    """
    
    def __init__(self):
        self.layers = []
        self.velocities = []
        self.thicknesses = []
        self.depths = []
        self.n_layers = 0
        
    def setup_layers(self, velocities: List[float], thicknesses: List[float] = None):
        """
        Phase 1: Layered Earth Setup
        
        Args:
            velocities: List of P-wave velocities for each layer (m/s)
            thicknesses: List of layer thicknesses (m). Last layer assumed infinite.
        """
        self.velocities = np.array(velocities)
        self.n_layers = len(velocities)
        
        if thicknesses is None:
            # Default thicknesses for demonstration
            self.thicknesses = np.array([10, 20, 30] + [np.inf])[:self.n_layers]
        else:
            self.thicknesses = np.array(thicknesses)
            
        # Ensure last layer is infinite
        if len(self.thicknesses) == self.n_layers and self.thicknesses[-1] != np.inf:
            self.thicknesses[-1] = np.inf
            
        # Calculate cumulative depths
        self.depths = np.zeros(self.n_layers)
        for i in range(1, self.n_layers):
            if self.thicknesses[i-1] != np.inf:
                self.depths[i] = self.depths[i-1] + self.thicknesses[i-1]
            else:
                self.depths[i] = np.inf
                
        print(f"Earth Model Setup:")
        print(f"Number of layers: {self.n_layers}")
        for i in range(self.n_layers):
            depth_str = f"{self.depths[i]:.1f}" if self.depths[i] != np.inf else "∞"
            thick_str = f"{self.thicknesses[i]:.1f}" if self.thicknesses[i] != np.inf else "∞"
            print(f"Layer {i+1}: V = {self.velocities[i]:.0f} m/s, "
                  f"Depth = {depth_str} m, Thickness = {thick_str} m")
    
    def calculate_critical_angle(self, v1: float, v2: float) -> float:
        """
        Calculate critical angle using Snell's Law
        
        Args:
            v1: Velocity of upper layer
            v2: Velocity of lower layer
            
        Returns:
            Critical angle in radians (returns π/2 if v2 <= v1)
        """
        if v2 <= v1:
            return np.pi/2  # No critical angle exists
        return np.arcsin(v1/v2)
    
    def calculate_direct_wave_time(self, offset: float) -> float:
        """
        Phase 2 & 3: Calculate travel time for direct wave
        
        Args:
            offset: Source-receiver distance (m)
            
        Returns:
            Travel time (s)
        """
        return offset / self.velocities[0]
    
    def calculate_refracted_wave_time(self, offset: float, refractor_layer: int) -> Tuple[float, bool]:
        """
        Phase 2 & 3: Calculate travel time for refracted (head) wave
        
        Args:
            offset: Source-receiver distance (m)
            refractor_layer: Index of refracting layer
            
        Returns:
            Tuple of (travel_time, is_valid)
        """
        if refractor_layer >= self.n_layers or refractor_layer < 1:
            return np.inf, False
            
        # Check if critical angles exist for all layers above refractor
        for i in range(refractor_layer):
            v_upper = self.velocities[i]
            v_refractor = self.velocities[refractor_layer]
            if v_refractor <= v_upper:
                return np.inf, False
                
        # Calculate total vertical travel time (down and up)
        total_vertical_time = 0
        total_horizontal_distance = 0
        
        for i in range(refractor_layer):
            if self.thicknesses[i] == np.inf:
                return np.inf, False
                
            # Critical angle for this interface
            critical_angle = self.calculate_critical_angle(
                self.velocities[i], self.velocities[refractor_layer]
            )
            
            # Vertical travel time in this layer
            cos_critical = np.cos(critical_angle)
            layer_vertical_time = 2 * self.thicknesses[i] / (self.velocities[i] * cos_critical)
            total_vertical_time += layer_vertical_time
            
            # Horizontal distance covered in this layer
            tan_critical = np.tan(critical_angle)
            layer_horizontal_distance = 2 * self.thicknesses[i] * tan_critical
            total_horizontal_distance += layer_horizontal_distance
        
        # Remaining horizontal distance traveled at refractor velocity
        remaining_horizontal = offset - total_horizontal_distance
        
        if remaining_horizontal < 0:
            return np.inf, False  # Ray doesn't reach this far
            
        horizontal_time = remaining_horizontal / self.velocities[refractor_layer]
        
        total_time = total_vertical_time + horizontal_time
        return total_time, True
    
    def calculate_reflected_wave_time(self, offset: float, reflector_layer: int) -> Tuple[float, bool]:
        """
        Phase 2 & 3: Calculate travel time for reflected wave (optional extension)
        
        Args:
            offset: Source-receiver distance (m)
            reflector_layer: Index of reflecting interface
            
        Returns:
            Tuple of (travel_time, is_valid)
        """
        if reflector_layer >= self.n_layers - 1 or reflector_layer < 0:
            return np.inf, False
            
        if self.thicknesses[reflector_layer] == np.inf:
            return np.inf, False
            
        # Depth to reflector
        reflector_depth = self.depths[reflector_layer] + self.thicknesses[reflector_layer]
        
        # Calculate reflection raypath
        # Using average velocity for simplicity (can be improved with ray tracing)
        avg_velocity = np.mean(self.velocities[:reflector_layer+1])
        
        # Two-way travel distance
        raypath_distance = np.sqrt(offset**2/4 + reflector_depth**2) * 2
        
        travel_time = raypath_distance / avg_velocity
        return travel_time, True
    
    def find_crossover_distance(self, refractor_layer: int, 
                              max_offset: float = 1000, 
                              n_points: int = 1000) -> Tuple[float, float]:
        """
        Phase 3: Find crossover distance where head wave overtakes direct wave
        
        Args:
            refractor_layer: Index of refracting layer
            max_offset: Maximum offset to search
            n_points: Number of points to evaluate
            
        Returns:
            Tuple of (crossover_distance, crossover_time)
        """
        offsets = np.linspace(1, max_offset, n_points)
        
        for offset in offsets:
            direct_time = self.calculate_direct_wave_time(offset)
            refracted_time, is_valid = self.calculate_refracted_wave_time(offset, refractor_layer)
            
            if is_valid and refracted_time < direct_time:
                return offset, refracted_time
                
        return np.inf, np.inf
    
    def calculate_tx_curves(self, max_offset: float = 500, n_points: int = 100) -> Dict:
        """
        Phase 3: Calculate t-x data for all wave types
        
        Args:
            max_offset: Maximum source-receiver distance
            n_points: Number of offset points to calculate
            
        Returns:
            Dictionary containing t-x data
        """
        offsets = np.linspace(1, max_offset, n_points)
        
        results = {
            'offsets': offsets,
            'direct': np.zeros(n_points),
            'refracted': {},
            'reflected': {},
            'crossovers': {}
        }
        
        # Direct wave
        for i, offset in enumerate(offsets):
            results['direct'][i] = self.calculate_direct_wave_time(offset)
        
        # Refracted waves for each possible refractor
        for layer in range(1, self.n_layers):
            refracted_times = np.full(n_points, np.inf)
            
            for i, offset in enumerate(offsets):
                time, is_valid = self.calculate_refracted_wave_time(offset, layer)
                if is_valid:
                    refracted_times[i] = time
                    
            results['refracted'][layer] = refracted_times
            
            # Find crossover distance
            crossover_dist, crossover_time = self.find_crossover_distance(layer, max_offset)
            results['crossovers'][layer] = (crossover_dist, crossover_time)
        
        # Reflected waves (optional)
        for layer in range(self.n_layers - 1):
            reflected_times = np.full(n_points, np.inf)
            
            for i, offset in enumerate(offsets):
                time, is_valid = self.calculate_reflected_wave_time(offset, layer)
                if is_valid:
                    reflected_times[i] = time
                    
            results['reflected'][layer] = reflected_times
            
        return results
    
    def plot_tx_curves(self, max_offset: float = 500, n_points: int = 100, 
                      show_reflected: bool = False, figsize: Tuple[int, int] = (12, 8)):
        """
        Phase 4: Plot t-x curves with crossover points
        
        Args:
            max_offset: Maximum offset for plotting
            n_points: Number of points to calculate
            show_reflected: Whether to show reflected arrivals
            figsize: Figure size tuple
        """
        # Calculate t-x data
        data = self.calculate_tx_curves(max_offset, n_points)
        
        plt.figure(figsize=figsize)
        
        # Plot direct wave
        plt.plot(data['offsets'], data['direct'], 'b-', linewidth=2, 
                label='Direct Wave', alpha=0.8)
        
        # Plot refracted waves
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (layer, times) in enumerate(data['refracted'].items()):
            valid_mask = times < np.inf
            if np.any(valid_mask):
                color = colors[i % len(colors)]
                plt.plot(data['offsets'][valid_mask], times[valid_mask], 
                        color=color, linewidth=2, alpha=0.8,
                        label=f'Refracted Wave (Layer {layer+1})')
                
                # Mark crossover point
                crossover_dist, crossover_time = data['crossovers'][layer]
                if crossover_dist < np.inf:
                    plt.plot(crossover_dist, crossover_time, 'o', 
                            color=color, markersize=8, markeredgecolor='black',
                            markeredgewidth=1, label=f'Crossover {layer+1}')
        
        # Plot reflected waves (optional)
        if show_reflected:
            for i, (layer, times) in enumerate(data['reflected'].items()):
                valid_mask = times < np.inf
                if np.any(valid_mask):
                    color = colors[i % len(colors)]
                    plt.plot(data['offsets'][valid_mask], times[valid_mask], 
                            '--', color=color, linewidth=1.5, alpha=0.6,
                            label=f'Reflected Wave (Interface {layer+1})')
        
        plt.xlabel('Offset Distance (m)', fontsize=12)
        plt.ylabel('Travel Time (s)', fontsize=12)
        plt.title('Seismic Refraction t-x Curves', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Print crossover information
        print("\nCrossover Analysis:")
        for layer, (dist, time) in data['crossovers'].items():
            if dist < np.inf:
                print(f"Layer {layer+1} crossover: {dist:.1f} m at {time:.4f} s")
            else:
                print(f"Layer {layer+1}: No crossover within {max_offset} m")
        
        plt.show()
        
        return data

def get_user_input_model():
    """
    Interactive function to get Earth model parameters from user input
    
    Returns:
        Tuple of (velocities, thicknesses) lists
    """
    print("\n" + "=" * 60)
    print("🌍 INTERACTIVE EARTH MODEL SETUP")
    print("=" * 60)
    print("Create your custom layered Earth model for seismic refraction analysis")
    print("Note: Velocities should generally increase with depth for proper refraction")
    print("-" * 60)
    
    # Get number of layers
    while True:
        try:
            n_layers = int(input("\nEnter number of layers (2-10): "))
            if 2 <= n_layers <= 10:
                break
            else:
                print("❌ Please enter a number between 2 and 10")
        except ValueError:
            print("❌ Please enter a valid integer")
    
    velocities = []
    thicknesses = []
    
    print(f"\n📝 Enter parameters for {n_layers} layers:")
    print("💡 Tip: Typical velocities - Soil: 300-800 m/s, Rock: 1500-6000 m/s")
    print("-" * 50)
    
    for i in range(n_layers):
        print(f"\n🏔️  LAYER {i+1}:")
        
        # Get velocity
        while True:
            try:
                if i == 0:
                    velocity = float(input(f"  Velocity (m/s): "))
                else:
                    prev_vel = velocities[-1]
                    velocity = float(input(f"  Velocity (m/s) [previous: {prev_vel}]: "))
                    
                if velocity <= 0:
                    print("  ❌ Velocity must be positive")
                    continue
                elif i > 0 and velocity <= velocities[-1]:
                    response = input(f"  ⚠️  Velocity ({velocity}) ≤ previous layer ({velocities[-1]}). Continue? (y/n): ")
                    if response.lower() != 'y':
                        continue
                        
                velocities.append(velocity)
                break
            except ValueError:
                print("  ❌ Please enter a valid number")
        
        # Get thickness (last layer is infinite)
        if i == n_layers - 1:
            thicknesses.append(np.inf)
            print(f"  Thickness: ∞ (bottom layer)")
        else:
            while True:
                try:
                    thickness = float(input(f"  Thickness (m): "))
                    if thickness <= 0:
                        print("  ❌ Thickness must be positive")
                        continue
                    thicknesses.append(thickness)
                    break
                except ValueError:
                    print("  ❌ Please enter a valid number")
    
    # Display summary
    print("\n" + "=" * 50)
    print("📋 MODEL SUMMARY:")
    print("=" * 50)
    total_depth = 0
    for i in range(n_layers):
        if i < n_layers - 1:
            depth_range = f"{total_depth:.1f} - {total_depth + thicknesses[i]:.1f} m"
            total_depth += thicknesses[i]
        else:
            depth_range = f"{total_depth:.1f} - ∞ m"
            
        thick_str = f"{thicknesses[i]:.1f} m" if thicknesses[i] != np.inf else "∞"
        print(f"Layer {i+1}: {velocities[i]:>8.0f} m/s | {thick_str:>8} | Depth: {depth_range}")
    
    return velocities, thicknesses

def get_plotting_parameters():
    """
    Get plotting parameters from user
    
    Returns:
        Tuple of (max_offset, n_points, show_reflected)
    """
    print("\n" + "=" * 50)
    print("📊 PLOTTING PARAMETERS")
    print("=" * 50)
    
    # Max offset
    while True:
        try:
            default_offset = 500
            offset_input = input(f"Maximum offset distance (m) [default: {default_offset}]: ").strip()
            max_offset = float(offset_input) if offset_input else default_offset
            if max_offset <= 0:
                print("❌ Offset must be positive")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Number of points
    while True:
        try:
            default_points = 200
            points_input = input(f"Number of calculation points [default: {default_points}]: ").strip()
            n_points = int(points_input) if points_input else default_points
            if n_points < 10:
                print("❌ Use at least 10 points")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid integer")
    
    # Show reflected waves
    while True:
        show_ref_input = input("Show reflected waves? (y/n) [default: n]: ").strip().lower()
        if show_ref_input in ['', 'n', 'no']:
            show_reflected = False
            break
        elif show_ref_input in ['y', 'yes']:
            show_reflected = True
            break
        else:
            print("❌ Please enter 'y' or 'n'")
    
    return max_offset, n_points, show_reflected

def interactive_modeling():
    """
    Complete interactive seismic refraction modeling workflow
    """
    print("🎯 INTERACTIVE SEISMIC REFRACTION MODELING")
    print("Build custom Earth models and analyze seismic wave propagation")
    
    while True:
        try:
            # Get user input for Earth model
            velocities, thicknesses = get_user_input_model()
            
            # Create and setup model
            model = SeismicRefractionModel()
            model.setup_layers(velocities, thicknesses)
            
            # Get plotting parameters
            max_offset, n_points, show_reflected = get_plotting_parameters()
            
            # Generate and plot results
            print("\n🔄 Calculating travel times and generating plots...")
            data = model.plot_tx_curves(max_offset=max_offset, 
                                      n_points=n_points, 
                                      show_reflected=show_reflected)
            
            # Manual calculation example
            print("\n" + "=" * 50)
            print("🧮 MANUAL CALCULATION EXAMPLE")
            print("=" * 50)
            
            while True:
                try:
                    test_offset = float(input(f"Enter offset for detailed calculation (1-{max_offset} m): "))
                    if 1 <= test_offset <= max_offset:
                        break
                    else:
                        print(f"❌ Please enter a value between 1 and {max_offset}")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            print(f"\n📍 Results for offset = {test_offset:.1f} m:")
            print("-" * 30)
            
            direct_time = model.calculate_direct_wave_time(test_offset)
            print(f"Direct wave:      {direct_time:.4f} s")
            
            for layer in range(1, model.n_layers):
                ref_time, valid = model.calculate_refracted_wave_time(test_offset, layer)
                if valid and ref_time < np.inf:
                    print(f"Refracted (L{layer+1}):   {ref_time:.4f} s")
                else:
                    print(f"Refracted (L{layer+1}):   No arrival")
            
            # Ask if user wants to continue
            print("\n" + "=" * 50)
            continue_input = input("🔄 Create another model? (y/n): ").strip().lower()
            if continue_input not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting program...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with valid inputs.")
    
    print("\n✅ Thank you for using the Seismic Refraction Modeling Tool!")

# Example usage and demonstration
def demonstrate_seismic_modeling():
    """
    Demonstrate the seismic refraction modeling tool with example data
    """
    print("=" * 60)
    print("SEISMIC REFRACTION MODELING DEMONSTRATION")
    print("=" * 60)
    
    # Create model instance
    model = SeismicRefractionModel()
    
    # Example 1: Three-layer model
    print("\n🌍 EXAMPLE 1: Three-Layer Earth Model")
    print("-" * 40)
    
    velocities = [500, 1500, 3000]  # m/s
    thicknesses = [20, 40, np.inf]  # m
    
    model.setup_layers(velocities, thicknesses)
    
    # Plot t-x curves
    print("\n📊 Generating t-x curves...")
    data = model.plot_tx_curves(max_offset=300, n_points=200)
    
    # Example 2: Four-layer model with higher velocities
    print("\n🌍 EXAMPLE 2: Four-Layer Earth Model")
    print("-" * 40)
    
    model2 = SeismicRefractionModel()
    velocities2 = [600, 1200, 2500, 4500]  # m/s
    thicknesses2 = [10, 25, 50, np.inf]    # m
    
    model2.setup_layers(velocities2, thicknesses2)
    
    print("\n📊 Generating t-x curves with reflections...")
    data2 = model2.plot_tx_curves(max_offset=400, n_points=200, show_reflected=True)
    
    # Demonstrate manual calculations
    print("\n🧮 MANUAL CALCULATION EXAMPLES:")
    print("-" * 40)
    
    offset = 100  # m
    print(f"\nFor offset = {offset} m:")
    
    direct_time = model.calculate_direct_wave_time(offset)
    print(f"Direct wave time: {direct_time:.4f} s")
    
    for layer in range(1, model.n_layers):
        ref_time, valid = model.calculate_refracted_wave_time(offset, layer)
        if valid:
            print(f"Refracted wave (Layer {layer+1}): {ref_time:.4f} s")
        else:
            print(f"Refracted wave (Layer {layer+1}): Invalid/No arrival")

def main_menu():
    """
    Main menu system for the seismic refraction modeling tool
    """
    while True:
        print("\n" + "=" * 60)
        print("🌍 SEISMIC REFRACTION MODELING TOOL")
        print("=" * 60)
        print("Choose your modeling approach:")
        print()
        print("1. 🎯 Interactive Mode - Create custom Earth models")
        print("2. 📚 Demo Mode - Run predefined examples")
        print("3. 🔧 Quick Mode - Single model setup")
        print("4. ❌ Exit")
        print("-" * 60)
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                interactive_modeling()
            elif choice == '2':
                demonstrate_seismic_modeling()
            elif choice == '3':
                quick_modeling()
            elif choice == '4':
                print("\n👋 Thank you for using the Seismic Refraction Modeling Tool!")
                break
            else:
                print("❌ Please enter a valid choice (1-4)")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting program...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

def quick_modeling():
    """
    Quick modeling mode with minimal input
    """
    print("\n🔧 QUICK MODELING MODE")
    print("-" * 30)
    
    try:
        # Get basic parameters
        n_layers = int(input("Number of layers (2-5): "))
        if not 2 <= n_layers <= 5:
            print("Using 3 layers (default)")
            n_layers = 3
        
        print(f"\nEnter {n_layers} velocities (m/s), separated by spaces:")
        print("Example: 500 1500 3000")
        vel_input = input("Velocities: ").strip().split()
        velocities = [float(v) for v in vel_input]
        
        if len(velocities) != n_layers:
            print("❌ Number of velocities doesn't match layer count")
            return
        
        print(f"\nEnter {n_layers-1} thicknesses (m), separated by spaces:")
        print("Example: 20 40")
        thick_input = input("Thicknesses: ").strip().split()
        thicknesses = [float(t) for t in thick_input] + [np.inf]
        
        if len(thicknesses) != n_layers:
            print("❌ Number of thicknesses doesn't match")
            return
        
        # Create and run model
        model = SeismicRefractionModel()
        model.setup_layers(velocities, thicknesses)
        model.plot_tx_curves(max_offset=400, n_points=150)
        
    except (ValueError, IndexError) as e:
        print(f"❌ Invalid input format: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main_menu()