"""
Zabawkowy model: "czas jest siłą".
IDEA: "Czas działa jak siła, która popycha stan układu w przyszłość."
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol, List
import random
import math

# EVOLVE-BLOCK-START
SIMPLE_IDEA = "Na horyzoncie zdarzeń czas, jaki znamy, kończy się."

@dataclass(frozen=True)
class SystemState:
    """
    Reprezentacja stanu układu dynamicznego.
    Attributes:
        value: Odległość od osobliwości (przestrzeń)
        time: Czas lokalny układu
        velocity: Prędkość radialna (zmiana odległości w czasie)
    """
    value: float = 10.0
    time: float = 0.0
    velocity: float = 0.0

class Force(Protocol):
    def apply(self, state: SystemState, dt: float) -> SystemState:
        ...

class TimeForce:
    def __init__(self, strength: float = 1.0) -> None:
        self.strength = float(strength)

    def apply(self, state: SystemState, dt: float) -> SystemState:
        new_time = state.time + dt
        acceleration = self.strength
        new_velocity = state.velocity + acceleration * dt
        new_value = state.value + new_velocity * dt
        return SystemState(value=new_value, time=new_time, velocity=new_velocity)

class BlurredTimeForce(TimeForce):
    """
    Siła czasu z elementem stochastycznym (rozmyciem).
    Filozofia Boone'a: Czas nie jest precyzyjny, jest 'rozmyty'.
    """
    def __init__(self, strength: float = 1.0, blur_factor: float = 0.1) -> None:
        super().__init__(strength)
        self.blur_factor = blur_factor

    def apply(self, state: SystemState, dt: float) -> SystemState:
        # Dodajemy szum do kroku czasowego
        noise = random.gauss(0, self.blur_factor * dt)
        blurred_dt = max(0.0, dt + noise) # Czas nie może płynąć ujemnie w tym modelu (chyba że w czarnej dziurze)
        
        # Aplikujemy logikę bazową z rozmytym czasem
        new_time = state.time + blurred_dt
        acceleration = self.strength
        new_velocity = state.velocity + acceleration * blurred_dt
        new_value = state.value + new_velocity * blurred_dt
        return SystemState(value=new_value, time=new_time, velocity=new_velocity)

class EventHorizonForce(TimeForce):
    """
    Implementacja Czarnej Dziury i załamania czasoprzestrzeni.
    """
    def __init__(self, strength: float = 1.0, horizon_radius: float = 5.0) -> None:
        super().__init__(strength)
        self.horizon_radius = horizon_radius

    def apply(self, state: SystemState, dt: float) -> SystemState:
        dist = state.value
        rs = self.horizon_radius

        if dist > rs:
            # NA ZEWNĄTRZ: Dylatacja czasu
            # Używamy max(0.0, ...) dla stabilności numerycznej
            dilation_factor = max(0.0, (dist - rs) / dist) if dist > 0 else 0.0
            effective_dt = dt * dilation_factor
            
        elif dist == rs:
            # NA HORYZONCIE
            effective_dt = 0.0
            
        else:
            # WEWNĄTRZ: Odwrócenie czasu
            effective_dt = -dt

        acceleration = self.strength
        new_time = state.time + effective_dt
        
        # Fizyka Newtonowska z odwróconym czasem (wewnątrz horyzontu)
        new_velocity = state.velocity + acceleration * effective_dt
        new_value = state.value + new_velocity * effective_dt

        return SystemState(value=new_value, time=new_time, velocity=new_velocity)

class Observer:
    """
    Subiektywny obserwator odczuwający upływ czasu.
    """
    def __init__(self, perception_bias: float = 0.0):
        self.perception_bias = perception_bias

    def perceive_time(self, objective_dt: float) -> float:
        """Zwraca subiektywnie odczuty czas."""
        # Bias > 0: czas płynie szybciej
        # Bias < 0: czas płynie wolniej (nuda/strach)
        return objective_dt * (1.0 + self.perception_bias)

class Integrator:
    def __init__(self, forces: List[Force]) -> None:
        self.forces = forces
    
    def integrate(self, state: SystemState, dt: float) -> SystemState:
        current_state = state
        for force in self.forces:
            current_state = force.apply(current_state, dt)
        return current_state

# SETUP
_DEFAULT_FORCE = EventHorizonForce(strength=1.0, horizon_radius=5.0)
_DEFAULT_INTEGRATOR = Integrator([_DEFAULT_FORCE])

def simulate_step(state: SystemState, dt: float) -> SystemState:
    return _DEFAULT_INTEGRATOR.integrate(state, dt)

def run_simulation(initial_state: SystemState, dt: float, steps: int) -> List[SystemState]:
    states = [initial_state]
    current = initial_state
    for _ in range(steps):
        current = simulate_step(current, dt)
        states.append(current)
    return states

def build_model(horizon: float = 5.0) -> tuple[Force, Integrator]:
    force = EventHorizonForce(strength=1.0, horizon_radius=horizon)
    integrator = Integrator([force])
    return force, integrator
# EVOLVE-BLOCK-END

def test_time_reversal_inside_horizon() -> bool:
    """Test czy czas cofa się wewnątrz czarnej dziury."""
    horizon = 5.0
    state = SystemState(value=2.0, time=10.0, velocity=0.0)
    force = EventHorizonForce(strength=1.0, horizon_radius=horizon)
    result = force.apply(state, dt=1.0)
    return result.time < 10.0

def test_blurred_time_force_variability() -> bool:
    """Test czy BlurredTimeForce faktycznie wprowadza zmienność."""
    state = SystemState(value=10.0, time=0.0, velocity=0.0)
    force = BlurredTimeForce(strength=1.0, blur_factor=0.3)
    times = []
    for _ in range(100):
        result = force.apply(state, dt=1.0)
        times.append(result.time)
        state = result
    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    return variance > 0.01

def test_observer_subjectivity() -> Dict[str, Any]:
    """Test subiektywnego doświadczenia czasu."""
    objective_dt = 1.0
    observers = [
        Observer(perception_bias=0.0),
        Observer(perception_bias=0.2),
        Observer(perception_bias=-0.2),
    ]
    results = {}
    for i, observer in enumerate(observers):
        results[f"observer_{i}"] = observer.perceive_time(objective_dt)
    return results

def demo() -> Dict[str, Any]:
    initial = SystemState(value=10.0, time=0.0, velocity=0.0)
    states = run_simulation(initial, dt=0.1, steps=100)
    return {
        "final_time": states[-1].time,
        "test_reversal": test_time_reversal_inside_horizon(),
        "test_blur": test_blurred_time_force_variability(),
        "test_observer": test_observer_subjectivity()
    }

if __name__ == "__main__":
    print(demo())