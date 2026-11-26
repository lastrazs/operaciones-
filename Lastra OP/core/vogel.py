"""Método de aproximación de Vogel (VAM).

Penalización de una fila o columna: diferencia entre sus dos costos activos
más bajos. Si solo le queda una celda activa, la penalización es ese costo
(en ese punto la asignación ya está forzada, así que la regla no cambia el costo).

Desempates, en este orden:
1. Mayor penalización. Si una fila y una columna empatan, gana la fila.
2. Entre filas (o columnas) empatadas, la de menor índice.
3. Dentro de la línea elegida, la celda de menor costo; si empatan, la de menor índice.
"""
from __future__ import annotations

import numpy as np

from core._allocation import asignar
from core.models import TransportProblem, TransportSolution


def _penalizacion(costos: np.ndarray) -> float:
    if costos.size == 1:
        return float(costos[0])
    a, b = np.partition(costos, 1)[:2]
    return float(b - a)


def _seleccionar(c, s, d, rows, cols):
    sub = c[np.ix_(rows, cols)]
    pen_filas = [_penalizacion(sub[r, :]) for r in range(len(rows))]
    pen_cols = [_penalizacion(sub[:, q]) for q in range(len(cols))]
    max_f, max_c = max(pen_filas), max(pen_cols)

    if max_f >= max_c:
        r = pen_filas.index(max_f)
        q = int(np.argmin(sub[r, :]))
        linea = ("fila", rows[r], max_f)
    else:
        q = pen_cols.index(max_c)
        r = int(np.argmin(sub[:, q]))
        linea = ("columna", cols[q], max_c)

    details = {
        "penalizaciones_filas": dict(zip(rows, pen_filas)),
        "penalizaciones_columnas": dict(zip(cols, pen_cols)),
        "linea_elegida": linea,
    }
    return rows[r], cols[q], details


def vogel(problema: TransportProblem) -> TransportSolution:
    """Requiere un problema balanceado (ver core.balancing.balancear)."""
    return asignar(problema, "Vogel", _seleccionar)
