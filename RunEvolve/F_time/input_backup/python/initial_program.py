"""
Zabawkowy model: "czas jest siłą".

Cel:
- Mamy BARDZO prosty punkt startowy:
    IDEA: "Czas działa jak siła, która popycha stan układu w przyszłość."
- OpenEvolve ma ewoluować ten kod tak, aby:
    * powstała coraz bardziej złożona architektura (klasy, metody, warstwy),
    * kod był dobrze udokumentowany (docstringi, komentarze),
    * zachowana była centralna idea: "czas jako siła".

To NIE jest model fizyczny – to generator struktur kodu z jednej idei.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict


# EVOLVE-BLOCK-START
SIMPLE_IDEA = "Czas jest siłą, która zmienia stan układu."

@dataclass
class SystemState:
    """Minimalny stan układu, który może być modyfikowany przez 'siłę czasu'."""
    value: float = 0.0
    time: float = 0.0


class TimeForce:
    """
    Bardzo prosty seed:
    - F_time = 1.0 (stała "siła czasu"),
    - apply() tylko zwiększa value o dt * F_time.
    OpenEvolve ma prawo całkowicie przepisać tę klasę.
    """

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = float(strength)

    def apply(self, state: SystemState, dt: float) -> SystemState:
        """Zastosuj prostą 'siłę czasu' do stanu."""
        state.time += dt
        state.value += self.strength * dt
        return state


def simulate_step(state: SystemState, dt: float) -> SystemState:
    """
    Najprostsza pętla aktualizacji:
    - tworzy TimeForce,
    - aplikuje ją raz.
    OpenEvolve może przekształcić to w dużo bardziej złożony silnik.
    """
    tf = TimeForce()
    return tf.apply(state, dt)
# EVOLVE-BLOCK-END


def demo() -> Dict[str, Any]:
    """Prosty test ręczny – OpenEvolve może go rozbudować w testy jednostkowe."""
    s = SystemState(value=0.0, time=0.0)
    s = simulate_step(s, 0.1)
    return {"value": s.value, "time": s.time}


if __name__ == "__main__":
    print("Demo:", demo())
