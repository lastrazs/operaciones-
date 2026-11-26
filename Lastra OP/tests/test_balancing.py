import numpy as np
import pytest

from core import TransportProblem, ValidationError, balancear

COSTOS = [[8, 6], [5, 12]]


def test_problema_balanceado_no_cambia_y_es_copia():
    p = TransportProblem(COSTOS, [10, 20], [15, 15])
    b = balancear(p)
    assert b.shape == (2, 2) and not b.dummy_origin and not b.dummy_destination
    b.costs[0, 0] = 99
    assert p.costs[0, 0] == 8


def test_exceso_de_oferta_agrega_destino_ficticio_con_costo_parametro():
    p = TransportProblem(COSTOS, [30, 20], [15, 15])
    b = balancear(p, costo_ficticio=2.5)
    assert b.shape == (2, 3) and b.dummy_destination
    np.testing.assert_allclose(b.costs[:, -1], [2.5, 2.5])
    assert b.demand[-1] == 20
    assert b.is_balanced
    assert p.shape == (2, 2)  # el original no se modifica


def test_exceso_de_demanda_agrega_origen_ficticio_con_vector_de_costos():
    p = TransportProblem(COSTOS, [10, 10], [15, 15])
    b = balancear(p, costo_ficticio=[100, 50])  # penalización por demanda insatisfecha
    assert b.shape == (3, 2) and b.dummy_origin
    np.testing.assert_allclose(b.costs[-1], [100, 50])
    assert b.supply[-1] == 10
    assert b.is_balanced


def test_costo_ficticio_por_defecto_es_cero():
    b = balancear(TransportProblem(COSTOS, [30, 20], [15, 15]))
    np.testing.assert_allclose(b.costs[:, -1], 0)


@pytest.mark.parametrize("cf", [-1, [1, 2, 3]])
def test_costo_ficticio_invalido(cf):
    with pytest.raises(ValidationError):
        balancear(TransportProblem(COSTOS, [30, 20], [15, 15]), costo_ficticio=cf)
