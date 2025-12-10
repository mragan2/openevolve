import math
import time

class MTDCSimulation:
    def __init__(self):
        # Physical Constants (Normalized for simulation)
        self.lambda_0 = 0.7       # Dark Energy density
        self.graviton_mass = 1.0  # Initial normalized mass
        self.decay_rate = 0.05    # Gamma factor
        self.time_steps = 20
        
    def decay_function(self, t):
        """
        Simulates the decay of the massive graviton over time.
        Formula: m_g(t) = m_0 * e^(-Gamma * t)
        """
        return self.graviton_mass * math.exp(-self.decay_rate * t)

    def effective_lambda(self, current_mass):
        """
        Calculates effective Cosmological Constant.
        In MTDC, Lambda adjusts as the graviton mass decays.
        Lambda_eff = Lambda_0 + (Coupling * Mass^2)
        """
        coupling_constant = 0.1
        return self.lambda_0 + (coupling_constant * (current_mass ** 2))

    def run(self):
        print(f"{'Time':<10} | {'Graviton Mass (m_g)':<20} | {'Eff. Lambda':<20}")
        print("-" * 55)

        for t in range(self.time_steps + 1):
            # Calculate current state
            m_g = self.decay_function(t)
            lambda_eff = self.effective_lambda(m_g)
            
            # Print state
            print(f"{t:<10} | {m_g:.4f}{' ':14} | {lambda_eff:.4f}")
            
            # Check stability (The logic we tested earlier)
            if lambda_eff < 0.7:
                print("!! WARNING: Vacuum Instability Detected !!")
                break
                
            time.sleep(0.1) # Small pause for effect

        print("-" * 55)
        print("MTDC Simulation Cycle Complete.")

if __name__ == "__main__":
    sim = MTDCSimulation()
    sim.run()
