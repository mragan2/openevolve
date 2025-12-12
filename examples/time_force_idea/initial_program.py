"""
Zabawkowy model: "Czas to tylko liczba".
IDEA: Startujemy od zera. Niech ewolucja wymyśli, czym jest czas.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any, Dict
import random
import math

# EVOLVE-BLOCK-START
SIMPLE_IDEA = "Czas płynie do przodu, zazwyczaj."

@dataclass
class SystemState:
    """
    Prosty stan cząstki.
    """
    x: float = 10.0      # Pozycja
    v: float = 0.0       # Prędkość
    t: float = 0.0       # Czas
    entropy: float = 0.0 # Miejsce na eksperymenty

class BasicPhysics:
    """
    Bardzo prosta, naiwna fizyka.
    Brak relatywistyki, brak limitów, brak ochrony.
    """
    def __init__(self, gravity: float = 1.0):
        self.gravity = gravity

    def update(self, state: SystemState, dt: float) -> SystemState:
        # 1. Zwykła fizyka (F = ma)
        # Siła ciągnie do zera (jak grawitacja)
        force = -self.gravity / (state.x * state.x + 0.1) # +0.1 żeby nie dzielić przez zero (jeszcze)
        
        # 2. Naiwna integracja
        new_v = state.v + force * dt
        new_x = state.x + new_v * dt
        
        # 3. Czas jest liniowy (NUDNE! Ewolucja powinna to zmienić)
        new_t = state.t + dt
        
        # 4. Entropia jest stała (NUDNE!)
        new_entropy = state.entropy

        return SystemState(x=new_x, v=new_v, t=new_t, entropy=new_entropy)

# Funkcje pomocnicze, które ewolucja może wykorzystać lub usunąć
def calculate_energy(state: SystemState) -> float:
    return 0.5 * state.v**2

def strange_attractor(x: float) -> float:
    return math.sin(x)

# Setup symulacji
def run_simulation(steps: int = 100) -> List[SystemState]:
    physics = BasicPhysics(gravity=5.0)
    current_state = SystemState(x=10.0, v=0.5, t=0.0)
    history = [current_state]

    for _ in range(steps):
        current_state = physics.update(current_state, dt=0.1)
        history.append(current_state)
        
    return history
# EVOLVE-BLOCK-END

def demo():
    results = run_simulation(20)
    final = results[-1]
    return {
        "final_pos": final.x,
        "final_time": final.t,
        "is_stable": abs(final.x) < 1000.0
    }

if __name__ == "__main__":
    print(demo())