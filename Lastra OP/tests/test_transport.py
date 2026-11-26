"""Prueba central: ningún método baja del óptimo y toda solución cumple oferta y demanda."""
import json
from pathlib import Path

import numpy as np
import pytest

from core import (
    TransportProblem, ValidationError, balancear, costo_minimo, es_factible,
    esquina_noroeste, programacion_lineal, vogel,
)

DATA = Path(__file__).resolve().parents[1] / "data"
HEURISTICAS = [esquina_noroeste, costo_minimo, vogel]
TODOS = HEURISTICAS + [programacion_lineal]


def cargar(nombre: str) -> dict:
    return json.loads((DATA / nombre).read_text(encoding="utf-8"))


def caso_base() -> TransportProblem:
    return TransportProblem.from_dict(cargar("example.json")["transport"])


def problemas_aleatorios(cantidad: int = 40, semilla: int = 2026) -> list[TransportProblem]:
    """Problemas pequeños con costos enteros en [0, 15): muchos empates a propósito."""
    rng = np.random.default_rng(semilla)
    problemas = []
    for _ in range(cantidad):
        m, n = rng.integers(2, 6, size=2)
        p = TransportProblem(
            rng.integers(0, 15, (m, n)),
            rng.integers(1, 60, m),
            rng.integers(1, 60, n),
        )
        problemas.append(balancear(p, costo_ficticio=int(rng.integers(0, 10))))
    return problemas


CASOS = [caso_base()] + problemas_aleatorios()


# --- Caso LogiDistribución: valores verificados a mano -----------------------

@pytest.mark.parametrize("metodo, esperado", [
    (esquina_noroeste, 8100),
    (costo_minimo, 6350),
    (vogel, 5850),
    (programacion_lineal, 5850),
])
def test_costos_caso_base(metodo, esperado):
    assert metodo(caso_base()).total_cost == pytest.approx(esperado)


def test_asignacion_esquina_noroeste_caso_base():
    esperado = [[300, 50, 0, 0], [0, 200, 50, 0], [0, 0, 150, 250]]
    np.testing.assert_allclose(esquina_noroeste(caso_base()).allocation, esperado)


# --- Prueba central ----------------------------------------------------------

@pytest.mark.parametrize("problema", CASOS)
def test_ningun_metodo_bajo_el_optimo_y_todos_factibles(problema):
    optimo = programacion_lineal(problema).total_cost
    for metodo in TODOS:
        sol = metodo(problema)
        assert es_factible(problema, sol), sol.method
        assert sol.total_cost >= optimo - 1e-6, sol.method
        assert sol.total_cost == pytest.approx(float((problema.costs * sol.allocation).sum()))


@pytest.mark.parametrize("problema", CASOS)
def test_heuristicas_dan_m_mas_n_menos_1_celdas_basicas(problema):
    m, n = problema.shape
    for metodo in HEURISTICAS:
        sol = metodo(problema)
        assert len(sol.basic_cells) == m + n - 1, sol.method
        assert len(set(sol.basic_cells)) == len(sol.basic_cells), sol.method
        assert len(sol.steps) == len(sol.basic_cells), sol.method


@pytest.mark.parametrize("metodo", TODOS)
def test_resultado_determinista(metodo):
    for problema in CASOS[:10]:
        a, b = metodo(problema), metodo(problema)
        np.testing.assert_array_equal(a.allocation, b.allocation)
        assert [s.description for s in a.steps] == [s.description for s in b.steps]


# --- Desempates documentados -------------------------------------------------

def test_costo_minimo_desempata_por_orden_de_fila():
    # Todas las celdas cuestan lo mismo: debe empezar en (0, 0) como noroeste.
    p = TransportProblem(np.ones((2, 2)), [10, 10], [5, 15])
    assert costo_minimo(p).steps[0].cell == (0, 0)


def test_vogel_desempata_fila_antes_que_columna():
    # Fila 0 y columna 0 tienen penalización 1; gana la fila 0.
    p = TransportProblem([[1, 2], [2, 3]], [10, 10], [10, 10])
    paso = vogel(p).steps[0]
    assert paso.details["linea_elegida"][:2] == ("fila", 0)
    assert paso.cell == (0, 0)


# --- Desbalance ---------------------------------------------------------------

def test_metodos_rechazan_problema_desbalanceado():
    p = TransportProblem.from_dict(cargar("example_unbalanced.json")["transport"])
    for metodo in TODOS:
        with pytest.raises(ValidationError):
            metodo(p)


def test_escenario_desbalanceado_se_resuelve_tras_balancear():
    datos = cargar("example_unbalanced.json")
    p = balancear(TransportProblem.from_dict(datos["transport"]), datos["dummy_cost"])
    optimo = programacion_lineal(p).total_cost
    for metodo in HEURISTICAS:
        sol = metodo(p)
        assert es_factible(p, sol)
        assert sol.total_cost >= optimo - 1e-6
