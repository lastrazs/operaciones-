"""Validaciones de entrada: costos >= 0, dimensiones coherentes, filas de Markov suman 1.

Cada validador reúne todos los errores y los lanza juntos en un solo
``ValidationError``, para que la interfaz los muestre de una vez.
"""
from __future__ import annotations

import numpy as np

from core.models import MarkovModel, TransportProblem, TransportSolution

TOL = 1e-9            # tolerancia numérica para cantidades de transporte
MARKOV_TOL = 1e-6     # tolerancia para probabilidades escritas a mano


class ValidationError(ValueError):
    """Datos de entrada inválidos. ``errors`` contiene cada mensaje por separado."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def validar_transporte(problema: TransportProblem) -> None:
    c, s, d = problema.costs, problema.supply, problema.demand
    if c.ndim != 2:
        raise ValidationError(["La matriz de costos debe ser bidimensional."])

    errors: list[str] = []
    m, n = c.shape
    if m == 0 or n == 0:
        errors.append("Debe haber al menos un origen y un destino.")
    if s.size != m:
        errors.append(f"La oferta tiene {s.size} valores y la matriz de costos {m} filas.")
    if d.size != n:
        errors.append(f"La demanda tiene {d.size} valores y la matriz de costos {n} columnas.")
    if len(problema.origins) != s.size:
        errors.append("El número de etiquetas de origen no coincide con la oferta.")
    if len(problema.destinations) != d.size:
        errors.append("El número de etiquetas de destino no coincide con la demanda.")
    for nombre, arr in (("costos", c), ("oferta", s), ("demanda", d)):
        if not np.all(np.isfinite(arr)):
            errors.append(f"Hay valores no numéricos o infinitos en {nombre}.")
    if errors:
        raise ValidationError(errors)

    for i, j in zip(*np.nonzero(c < 0)):
        errors.append(
            f"Costo negativo en {problema.origins[i]} -> {problema.destinations[j]}: {c[i, j]:g}."
        )
    for i in np.nonzero(s < 0)[0]:
        errors.append(f"Oferta negativa en {problema.origins[i]}: {s[i]:g}.")
    for j in np.nonzero(d < 0)[0]:
        errors.append(f"Demanda negativa en {problema.destinations[j]}: {d[j]:g}.")
    if not errors and (s.sum() <= 0 or d.sum() <= 0):
        errors.append("La oferta total y la demanda total deben ser positivas.")
    if errors:
        raise ValidationError(errors)


def validar_markov(modelo: MarkovModel, tol: float = MARKOV_TOL) -> None:
    P, v0 = modelo.transition, modelo.initial
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValidationError(["La matriz de transición debe ser cuadrada."])

    errors: list[str] = []
    k = P.shape[0]
    if v0.size != k:
        errors.append(f"La distribución inicial tiene {v0.size} valores y la matriz {k} estados.")
    if len(modelo.states) != k:
        errors.append("El número de etiquetas de estado no coincide con la matriz.")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(v0)):
        errors.append("Hay valores no numéricos o infinitos.")
    if errors:
        raise ValidationError(errors)

    if np.any(P < 0):
        errors.append("La matriz de transición tiene probabilidades negativas.")
    for i, total in enumerate(P.sum(axis=1)):
        if abs(total - 1) > tol:
            errors.append(f"La fila {modelo.states[i]} suma {total:.6g}, debe sumar 1.")
    if np.any(v0 < 0):
        errors.append("La distribución inicial tiene valores negativos.")
    if abs(v0.sum() - 1) > tol:
        errors.append(f"La distribución inicial suma {v0.sum():.6g}, debe sumar 1.")
    if errors:
        raise ValidationError(errors)


def es_factible(problema: TransportProblem, solucion: TransportSolution, tol: float = 1e-6) -> bool:
    """True si la asignación es no negativa y cumple exactamente oferta y demanda."""
    x = solucion.allocation
    return bool(
        x.shape == problema.costs.shape
        and np.all(x >= -tol)
        and np.allclose(x.sum(axis=1), problema.supply, atol=tol)
        and np.allclose(x.sum(axis=0), problema.demand, atol=tol)
    )


def requerir_balanceado(problema: TransportProblem) -> None:
    validar_transporte(problema)
    if not problema.is_balanced:
        raise ValidationError([
            f"El problema no está balanceado (oferta {problema.total_supply:g}, "
            f"demanda {problema.total_demand:g}). Aplica balancear() antes de resolverlo."
        ])
