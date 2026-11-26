"""Motor común de los métodos iniciales (noroeste, costo mínimo, Vogel).

Los tres siguen el mismo ciclo: elegir una celda, asignar min(oferta, demanda),
tachar la fila o la columna agotada y registrar el paso. Solo cambia la regla
de selección, que cada módulo pasa como función.

Degeneración: si en un paso se agotan a la vez la oferta y la demanda, se tacha
solo la fila (o la columna si ya no quedan más filas). La columna sigue activa
con demanda 0 y luego recibe una asignación de 0. Así toda solución inicial
tiene exactamente m + n - 1 celdas básicas, lo que necesitará MODI/stepping-stone.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from core.models import Step, TransportProblem, TransportSolution
from core.validation import TOL, requerir_balanceado

# seleccionar(costos, oferta_restante, demanda_restante, filas_activas, columnas_activas)
#   -> (i, j, detalles_para_el_paso)
Selector = Callable[[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]],
                    tuple[int, int, dict[str, Any]]]


def asignar(problema: TransportProblem, metodo: str, seleccionar: Selector) -> TransportSolution:
    requerir_balanceado(problema)
    c = problema.costs
    m, n = c.shape
    s, d = problema.supply.copy(), problema.demand.copy()
    rows, cols = list(range(m)), list(range(n))
    x = np.zeros((m, n))
    basic: list[tuple[int, int]] = []
    steps: list[Step] = []

    while rows and cols:
        i, j, details = seleccionar(c, s, d, rows, cols)
        q = min(s[i], d[j])
        x[i, j] += q
        s[i] -= q
        d[j] -= q
        basic.append((i, j))

        oferta_agotada, demanda_agotada = s[i] <= TOL, d[j] <= TOL
        o, dst = problema.origins[i], problema.destinations[j]
        if oferta_agotada and (not demanda_agotada or len(rows) > 1):
            s[i] = 0.0
            rows.remove(i)
            cierre = f"se agota la oferta de {o}"
            if demanda_agotada:
                cierre += f" (degenerado: {dst} queda activo con demanda 0)"
        else:
            d[j] = 0.0
            cols.remove(j)
            cierre = f"se cubre la demanda de {dst}"

        steps.append(Step(
            description=f"Asignar {q:g} unidades a {o} -> {dst} (costo {c[i, j]:g}); {cierre}.",
            cell=(i, j),
            quantity=float(q),
            details={**details,
                     "oferta_restante": s.tolist(),
                     "demanda_restante": d.tolist()},
        ))

    return TransportSolution(
        method=metodo,
        problem=problema,
        allocation=x,
        total_cost=float((c * x).sum()),
        steps=steps,
        basic_cells=basic,
    )
