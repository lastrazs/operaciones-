import json
from pathlib import Path

import pytest

from core import MarkovModel, TransportProblem, ValidationError, validar_markov, validar_transporte

DATA = Path(__file__).resolve().parents[1] / "data"


def datos_ejemplo() -> dict:
    return json.loads((DATA / "example.json").read_text(encoding="utf-8"))


def test_datos_de_ejemplo_son_validos():
    d = datos_ejemplo()
    validar_transporte(TransportProblem.from_dict(d["transport"]))
    validar_markov(MarkovModel.from_dict(d["markov"]))


def test_costo_negativo_se_rechaza_con_ubicacion():
    p = TransportProblem([[1, -2], [3, 4]], [5, 5], [5, 5], ["A", "B"], ["X", "Y"])
    with pytest.raises(ValidationError, match="A -> Y"):
        validar_transporte(p)


@pytest.mark.parametrize("oferta, demanda", [([-1, 5], [2, 2]), ([5, 5], [5, -5])])
def test_oferta_o_demanda_negativa(oferta, demanda):
    with pytest.raises(ValidationError):
        validar_transporte(TransportProblem([[1, 2], [3, 4]], oferta, demanda))


def test_dimensiones_incoherentes():
    with pytest.raises(ValidationError):
        validar_transporte(TransportProblem([[1, 2], [3, 4]], [5, 5, 5], [5, 5]))


def test_se_reportan_todos_los_errores_juntos():
    p = TransportProblem([[-1, -2], [3, 4]], [5, -5], [5, 5])
    with pytest.raises(ValidationError) as exc:
        validar_transporte(p)
    assert len(exc.value.errors) == 3


def test_fila_de_markov_que_no_suma_1():
    m = MarkovModel([[0.5, 0.4], [0.3, 0.7]], [0.5, 0.5], ["A", "B"])
    with pytest.raises(ValidationError, match="fila A"):
        validar_markov(m)


def test_probabilidad_negativa_y_distribucion_inicial_invalida():
    m = MarkovModel([[1.2, -0.2], [0.3, 0.7]], [0.6, 0.6])
    with pytest.raises(ValidationError) as exc:
        validar_markov(m)
    assert len(exc.value.errors) == 2


def test_matriz_no_cuadrada():
    with pytest.raises(ValidationError):
        validar_markov(MarkovModel([[0.5, 0.5]], [1.0]))
