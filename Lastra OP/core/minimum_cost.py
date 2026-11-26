"""Método del costo mínimo.

Desempate: entre celdas activas con el mismo costo se elige la primera en
orden de fila (fila menor y, dentro de ella, columna menor). np.argmin sobre
la submatriz aplanada devuelve justo esa celda.
"""
from __future__ import annotations

import numpy as np

from core._allocation import asignar
from core.models import TransportProblem, TransportSolution


def _seleccionar(c, s, d, rows, cols):
    sub = c[np.ix_(rows, cols)]
    r, q = divmod(int(np.argmin(sub)), len(cols))
    return rows[r], cols[q], {}


def costo_minimo(problema: TransportProblem) -> TransportSolution:
    """Requiere un problema balanceado (ver core.balancing.balancear)."""
    return asignar(problema, "Costo mínimo", _seleccionar)
