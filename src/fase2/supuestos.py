"""Parte 2. Verificación de supuestos previa a cada prueba.

Regla de la fase: ninguna prueba se ejecuta sin verificar antes sus supuestos.
Si un supuesto no se cumple, se pasa a la alternativa que no lo necesita y se
deja constancia de la decisión.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .datos import AZUL, GRIS, NARANJA, SEMILLA

N_MAX_SHAPIRO = 5000


def prueba_normalidad(x, n_max: int = N_MAX_SHAPIRO, semilla: int = SEMILLA) -> dict:
    """Shapiro-Wilk más asimetría y curtosis.

    SciPy advierte que el valor p de Shapiro-Wilk no es exacto con más de 5000
    observaciones. Por eso, cuando la muestra es mayor, la prueba se aplica sobre
    una submuestra aleatoria reproducible de 5000 casos. Con muestras grandes la
    prueba detecta desvíos mínimos, así que siempre se lee junto con el gráfico
    Q-Q, la asimetría y la curtosis.
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    muestra = x
    if len(x) > n_max:
        muestra = np.random.default_rng(semilla).choice(x, size=n_max, replace=False)
    w, p = stats.shapiro(muestra)
    return {
        "n": len(x),
        "n_shapiro": len(muestra),
        "W": w,
        "p": p,
        "asimetria": stats.skew(x),
        "curtosis_exceso": stats.kurtosis(x),
    }


def tabla_normalidad(muestras: dict, alfa: float) -> pd.DataFrame:
    """Aplica ``prueba_normalidad`` a varias muestras y resume la decisión."""
    filas = []
    for nombre, x in muestras.items():
        r = prueba_normalidad(x)
        r = {"muestra": nombre, **r, "normalidad": "se rechaza" if r["p"] < alfa else "no se rechaza"}
        filas.append(r)
    return pd.DataFrame(filas).set_index("muestra")


def grafico_qq(muestras: dict, titulo: str, n_max: int = N_MAX_SHAPIRO, semilla: int = SEMILLA) -> plt.Figure:
    """Gráficos Q-Q normales (uno por muestra).

    Si los puntos siguen la recta, la distribución es compatible con la normal.
    Se grafica la misma submuestra que usa Shapiro-Wilk para no superponer
    decenas de miles de puntos.
    """
    fig, ejes = plt.subplots(1, len(muestras), figsize=(4.2 * len(muestras), 3.8), squeeze=False)
    for eje, (nombre, x) in zip(ejes[0], muestras.items()):
        x = pd.Series(x).dropna().to_numpy(dtype=float)
        if len(x) > n_max:
            x = np.random.default_rng(semilla).choice(x, size=n_max, replace=False)
        (teoricos, observados), (pendiente, ordenada, _) = stats.probplot(x, dist="norm")
        eje.scatter(teoricos, observados, s=6, color=AZUL, alpha=0.5, linewidths=0, label="Datos")
        eje.plot(teoricos, ordenada + pendiente * teoricos, color=NARANJA, linewidth=2, label="Normal teórica")
        eje.set_title(nombre)
        eje.set_xlabel("Cuantiles teóricos (normal estándar)")
        eje.set_ylabel("Cuantiles observados")
    ejes[0][0].legend(loc="upper left")
    fig.suptitle(titulo, fontweight="bold", color=GRIS)
    fig.tight_layout()
    return fig


def prueba_levene(*grupos) -> dict:
    """Homogeneidad de varianzas (Levene centrado en la mediana, Brown-Forsythe).

    Se centra en la mediana porque es la versión robusta cuando las
    distribuciones no son normales.
    """
    grupos = [pd.Series(g).dropna().to_numpy(dtype=float) for g in grupos]
    w, p = stats.levene(*grupos, center="median")
    return {
        "W": w,
        "gl1": len(grupos) - 1,
        "gl2": sum(len(g) for g in grupos) - len(grupos),
        "p": p,
        "razon_varianzas": max(g.var(ddof=1) for g in grupos) / min(g.var(ddof=1) for g in grupos),
    }


def condicion_proporciones(exitos, totales, etiquetas, minimo: int = 10) -> pd.DataFrame:
    """Condición de tamaño para la aproximación normal de una proporción.

    La prueba z usa la normal como aproximación de la binomial; es adecuada si
    en cada grupo hay al menos ``minimo`` éxitos y ``minimo`` fracasos.
    """
    exitos, totales = np.asarray(exitos), np.asarray(totales)
    tabla = pd.DataFrame({
        "n": totales,
        "exitos": exitos,
        "fracasos": totales - exitos,
        "proporcion": exitos / totales,
    }, index=etiquetas)
    tabla["cumple"] = (tabla["exitos"] >= minimo) & (tabla["fracasos"] >= minimo)
    return tabla


def frecuencias_esperadas(tabla: pd.DataFrame) -> dict:
    """Condición de Cochran para chi-cuadrado: esperadas >= 5 (y ninguna < 1)."""
    esperadas = stats.contingency.expected_freq(tabla.to_numpy())
    return {
        "esperada_minima": esperadas.min(),
        "celdas_menores_a_5": int((esperadas < 5).sum()),
        "cumple": bool((esperadas >= 5).all()),
        "esperadas": pd.DataFrame(esperadas, index=tabla.index, columns=tabla.columns),
    }
