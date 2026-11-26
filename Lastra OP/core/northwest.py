"""Método de la esquina noroeste."""
from __future__ import annotations

from core._allocation import asignar
from core.models import TransportProblem, TransportSolution


def _seleccionar(c, s, d, rows, cols):
    # La esquina noroeste es la primera fila y la primera columna aún activas.
    return rows[0], cols[0], {}


def esquina_noroeste(problema: TransportProblem) -> TransportSolution:
    """Requiere un problema balanceado (ver core.balancing.balancear)."""
    return asignar(problema, "Esquina noroeste", _seleccionar)
