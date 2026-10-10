"""Parte 2. Tamaños del efecto e intervalos de confianza clásicos.

Con 57.109 personas casi cualquier diferencia resulta estadísticamente
significativa. El tamaño del efecto es lo que indica si la diferencia importa,
por eso cada contraste informa uno junto con su valor p.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .datos import NIVEL_CONFIANZA

# Umbrales convencionales de Cohen (1988). Para la razón de odds se usan los
# equivalentes de d = 0,2; 0,5; 0,8 con la conversión logística de Chinn (2000):
# OR = exp(d * pi / sqrt(3)). Son referencias orientativas, no reglas: la
# relevancia práctica se discute además en unidades originales (puntos
# porcentuales, años, guaraníes).
UMBRALES = {
    "h": (0.2, 0.5, 0.8),       # diferencia de proporciones (h de Cohen)
    "d": (0.2, 0.5, 0.8),       # diferencia de medias estandarizada
    "r": (0.1, 0.3, 0.5),       # correlaciones, biserial por rangos, V con gl* = 1
    "or": (1.44, 2.48, 4.27),   # razón de odds (se evalúa max(OR, 1/OR))
}


def magnitud(valor: float, tipo: str) -> str:
    """Clasifica un tamaño del efecto en trivial, pequeño, mediano o grande."""
    pequeno, mediano, grande = UMBRALES[tipo]
    v = max(valor, 1 / valor) if tipo == "or" else abs(valor)
    if v < pequeno:
        return "trivial"
    if v < mediano:
        return "pequeño"
    if v < grande:
        return "mediano"
    return "grande"


def h_cohen(p1: float, p2: float) -> float:
    """h de Cohen: diferencia entre dos proporciones en escala arcoseno."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def medidas_2x2(a: int, b: int, c: int, d: int) -> dict:
    """Riesgo relativo y razón de odds de una tabla 2x2.

    Disposición:            evento   no evento
        grupo de interés      a          b
        grupo de referencia   c          d
    """
    p1, p2 = a / (a + b), c / (c + d)
    return {
        "diferencia": p1 - p2,
        "riesgo_relativo": p1 / p2,
        "razon_odds": (a * d) / (b * c),
        "h": h_cohen(p1, p2),
    }


def d_cohen(x, y) -> float:
    """d de Cohen con desvío estándar combinado (media de x menos media de y)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    s_comb = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / s_comb


def probabilidad_superioridad(x, y) -> float:
    """Probabilidad de que un valor de x supere a uno de y (los empates cuentan 1/2).

    Es el estadístico U de Mann-Whitney dividido por n1*n2, es decir, el efecto
    que esa prueba contrasta: 0,5 significa que ningún grupo tiende a tener
    valores mayores que el otro.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rangos = stats.rankdata(np.concatenate([x, y]))
    n1, n2 = len(x), len(y)
    u1 = rangos[:n1].sum() - n1 * (n1 + 1) / 2
    return u1 / (n1 * n2)


def biserial_rangos(prob_superioridad: float) -> float:
    """Correlación biserial por rangos: r = 2A - 1, entre -1 y 1."""
    return 2 * prob_superioridad - 1


def v_cramer(tabla) -> float:
    """V de Cramér: intensidad de la asociación entre dos variables categóricas (0 a 1)."""
    tabla = np.asarray(tabla)
    chi2 = stats.chi2_contingency(tabla, correction=False).statistic
    return np.sqrt(chi2 / (tabla.sum() * (min(tabla.shape) - 1)))


def ic_correlacion_fisher(r: float, n: int, metodo: str = "pearson",
                          nivel: float = NIVEL_CONFIANZA) -> tuple[float, float]:
    """IC de una correlación por la transformación z de Fisher.

    Para Pearson el error estándar de z es 1/sqrt(n-3). Para Spearman se usa la
    corrección de Bonett y Wright (2000): sqrt((1 + r^2/2) / (n-3)).
    """
    z = np.arctanh(r)
    ee = 1 / np.sqrt(n - 3)
    if metodo == "spearman":
        ee *= np.sqrt(1 + r ** 2 / 2)
    critico = stats.norm.ppf(1 - (1 - nivel) / 2)
    return float(np.tanh(z - critico * ee)), float(np.tanh(z + critico * ee))


def spearman_rapido(x, y) -> float:
    """rho de Spearman (Pearson sobre rangos); versión liviana para el bootstrap."""
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])
