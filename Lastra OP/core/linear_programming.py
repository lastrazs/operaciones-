"""Solución óptima por programación lineal con SciPy (linprog, método HiGHS).

min Z = sum c_ij x_ij   s.a.   sum_j x_ij = O_i,   sum_i x_ij = D_j,   x_ij >= 0
Es la referencia del óptimo contra la que se comparan los métodos iniciales.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from core.models import Step, TransportProblem, TransportSolution
from core.validation import requerir_balanceado


def programacion_lineal(problema: TransportProblem) -> TransportSolution:
    """Requiere un problema balanceado (ver core.balancing.balancear)."""
    requerir_balanceado(problema)
    c = problema.costs
    m, n = c.shape

    # Variable x_ij en la posición i*n + j.
    A_oferta = np.kron(np.eye(m), np.ones((1, n)))
    A_demanda = np.kron(np.ones((1, m)), np.eye(n))
    A_eq = np.vstack([A_oferta, A_demanda])
    b_eq = np.concatenate([problema.supply, problema.demand])

    res = linprog(c.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"linprog no encontró solución: {res.message}")

    x = res.x.reshape(m, n)
    x[np.abs(x) < 1e-9] = 0.0
    basic = [(int(i), int(j)) for i, j in zip(*np.nonzero(x))]
    total = float((c * x).sum())

    steps = [
        Step(f"Formulación: {m * n} variables x_ij >= 0, {m} restricciones de oferta "
             f"y {n} de demanda."),
        Step(f"Resolución con SciPy linprog (HiGHS): {res.message}",
             details={"iteraciones": int(getattr(res, "nit", 0))}),
        Step(f"Costo mínimo Z = {total:g}."),
    ]
    return TransportSolution(
        method="Programación lineal",
        problem=problema,
        allocation=x,
        total_cost=total,
        steps=steps,
        basic_cells=basic,
    )
