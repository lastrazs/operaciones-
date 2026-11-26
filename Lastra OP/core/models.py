"""Estructuras de datos compartidas por todo el proyecto.

Decisión central: los tres métodos heurísticos y la programación lineal
devuelven el mismo ``TransportSolution``. Así la tabla comparativa, las
gráficas y las pruebas tratan a todos los métodos por igual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _default_labels(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{k + 1}" for k in range(n)]


@dataclass
class TransportProblem:
    """Problema de transporte: costos (m x n), oferta (m), demanda (n) y etiquetas."""

    costs: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    origins: list[str] = field(default_factory=list)
    destinations: list[str] = field(default_factory=list)
    dummy_origin: bool = False        # True si balancear() agregó una fila ficticia
    dummy_destination: bool = False   # True si balancear() agregó una columna ficticia

    def __post_init__(self) -> None:
        self.costs = np.asarray(self.costs, dtype=float)
        self.supply = np.asarray(self.supply, dtype=float).ravel()
        self.demand = np.asarray(self.demand, dtype=float).ravel()
        self.origins = list(self.origins) or _default_labels("O", self.supply.size)
        self.destinations = list(self.destinations) or _default_labels("D", self.demand.size)

    @property
    def shape(self) -> tuple[int, int]:
        return self.supply.size, self.demand.size

    @property
    def total_supply(self) -> float:
        return float(self.supply.sum())

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())

    @property
    def is_balanced(self) -> bool:
        return bool(np.isclose(self.total_supply, self.total_demand))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransportProblem":
        return cls(
            costs=data["costs"],
            supply=data["supply"],
            demand=data["demand"],
            origins=data.get("origins", []),
            destinations=data.get("destinations", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "costs": self.costs.tolist(),
            "supply": self.supply.tolist(),
            "demand": self.demand.tolist(),
            "origins": self.origins,
            "destinations": self.destinations,
        }


@dataclass
class Step:
    """Una iteración de un método. La interfaz la muestra sin recalcular nada."""

    description: str
    cell: tuple[int, int] | None = None
    quantity: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportSolution:
    """Resultado común de esquina noroeste, costo mínimo, Vogel y programación lineal."""

    method: str
    problem: TransportProblem
    allocation: np.ndarray
    total_cost: float
    steps: list[Step] = field(default_factory=list)
    basic_cells: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class MarkovModel:
    """Cadena de Markov: matriz de transición P (filas = estado actual) y distribución inicial."""

    transition: np.ndarray
    initial: np.ndarray
    states: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.transition = np.asarray(self.transition, dtype=float)
        self.initial = np.asarray(self.initial, dtype=float).ravel()
        self.states = list(self.states) or _default_labels("E", self.initial.size)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkovModel":
        return cls(
            transition=data["transition"],
            initial=data["initial"],
            states=data.get("states", []),
        )
