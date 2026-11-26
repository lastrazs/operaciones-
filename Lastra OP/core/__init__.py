"""Lógica matemática del optimizador logístico. No depende de Streamlit."""
from core.balancing import balancear
from core.linear_programming import programacion_lineal
from core.minimum_cost import costo_minimo
from core.models import MarkovModel, Step, TransportProblem, TransportSolution
from core.northwest import esquina_noroeste
from core.validation import ValidationError, es_factible, validar_markov, validar_transporte
from core.vogel import vogel

METODOS_INICIALES = {
    "Esquina noroeste": esquina_noroeste,
    "Costo mínimo": costo_minimo,
    "Vogel": vogel,
}
