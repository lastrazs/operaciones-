"""Balanceo del problema de transporte con origen o destino ficticio."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.models import TransportProblem
from core.validation import TOL, ValidationError, validar_transporte

DUMMY_ORIGIN = "Origen ficticio"
DUMMY_DESTINATION = "Destino ficticio"


def balancear(
    problema: TransportProblem,
    costo_ficticio: float | Sequence[float] = 0.0,
) -> TransportProblem:
    """Devuelve un problema nuevo con oferta total igual a demanda total.

    - Oferta > demanda: agrega un destino ficticio (columna) que absorbe el sobrante.
    - Demanda > oferta: agrega un origen ficticio (fila) que representa la demanda insatisfecha.

    ``costo_ficticio`` es un parámetro, no un cero fijo: puede ser un número o
    un vector (uno por origen si se agrega columna, uno por destino si se agrega
    fila), por ejemplo para penalizar la demanda insatisfecha. El problema
    original no se modifica.
    """
    validar_transporte(problema)
    c, s, d = problema.costs.copy(), problema.supply.copy(), problema.demand.copy()
    origins, destinations = list(problema.origins), list(problema.destinations)
    m, n = c.shape
    diff = problema.total_supply - problema.total_demand

    if abs(diff) <= TOL:
        return TransportProblem(c, s, d, origins, destinations,
                                problema.dummy_origin, problema.dummy_destination)

    largo = m if diff > 0 else n
    cf = np.asarray(costo_ficticio, dtype=float)
    try:
        cf = np.broadcast_to(cf, (largo,)).copy()
    except ValueError:
        raise ValidationError([f"El costo ficticio debe ser un número o un vector de {largo} valores."])
    if np.any(cf < 0) or not np.all(np.isfinite(cf)):
        raise ValidationError(["El costo ficticio no puede ser negativo ni infinito."])

    if diff > 0:
        return TransportProblem(
            np.column_stack([c, cf]), s, np.append(d, diff),
            origins, destinations + [DUMMY_DESTINATION],
            problema.dummy_origin, True,
        )
    return TransportProblem(
        np.vstack([c, cf]), np.append(s, -diff), d,
        origins + [DUMMY_ORIGIN], destinations,
        True, problema.dummy_destination,
    )
