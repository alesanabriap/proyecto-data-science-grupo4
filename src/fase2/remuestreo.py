"""Parte 1. Remuestreo y dependencia entre observaciones.

La EPHC no es una muestra aleatoria simple de personas: se seleccionan unidades
primarias de muestreo (UPM), dentro de ellas viviendas, y se entrevista a todos
los integrantes del hogar. Las personas de un mismo hogar o de una misma UPM se
parecen entre sí, de modo que el supuesto de independencia de las pruebas
clásicas no se cumple estrictamente. Este módulo permite:

* medir esa dependencia (correlación intraclase y efecto de diseño);
* estimar intervalos de confianza por bootstrap, remuestreando personas o
  conglomerados completos;
* obtener errores estándar robustos a conglomerados para una diferencia de
  proporciones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .datos import N_BOOTSTRAP, NIVEL_CONFIANZA, SEMILLA


def correlacion_intraclase(valores, conglomerados) -> dict:
    """Correlación intraclase (estimador ANOVA de un factor) y efecto de diseño.

    ICC = proporción de la variabilidad total que se debe a diferencias entre
    conglomerados. Efecto de diseño aproximado (Kish): 1 + (m - 1) * ICC, donde
    m es el tamaño medio del conglomerado. Un efecto de diseño de 2 significa
    que la varianza real duplica la que supone una muestra aleatoria simple.
    """
    datos = pd.DataFrame({"y": np.asarray(valores, dtype=float), "g": np.asarray(conglomerados)})
    grupos = datos.groupby("g")["y"]
    n_i = grupos.size().to_numpy()
    media_i = grupos.mean().to_numpy()
    k, n_total = len(n_i), n_i.sum()
    media = datos["y"].mean()

    cm_entre = (n_i * (media_i - media) ** 2).sum() / (k - 1)
    cm_dentro = ((datos["y"] - grupos.transform("mean")) ** 2).sum() / (n_total - k)
    n0 = (n_total - (n_i ** 2).sum() / n_total) / (k - 1)   # tamaño "promedio" con grupos desiguales
    icc = (cm_entre - cm_dentro) / (cm_entre + (n0 - 1) * cm_dentro)
    tam_medio = n_total / k
    return {
        "conglomerados": k,
        "tamano_medio": tam_medio,
        "icc": icc,
        "efecto_diseno": 1 + (tam_medio - 1) * icc,
    }


def ic_bootstrap(estadistico, *arreglos, conglomerados=None, n_rep: int = N_BOOTSTRAP,
                 nivel: float = NIVEL_CONFIANZA, semilla: int = SEMILLA) -> dict:
    """Intervalo de confianza bootstrap por el método del percentil.

    Parameters
    ----------
    estadistico : función que recibe los arreglos remuestreados y devuelve un número.
    arreglos    : uno o más arreglos alineados (misma longitud, una fila por persona).
    conglomerados : si es None se remuestrean personas (bootstrap simple). Si se
        pasa un arreglo de identificadores, se remuestrean conglomerados completos
        con reemplazo y se conservan todas sus personas (bootstrap por conglomerados),
        lo que respeta la dependencia dentro de cada conglomerado.
    """
    arreglos = [np.asarray(a) for a in arreglos]
    n = len(arreglos[0])
    rng = np.random.default_rng(semilla)

    if conglomerados is not None:
        codigos = pd.factorize(np.asarray(conglomerados))[0]
        orden = np.argsort(codigos, kind="stable")
        cortes = np.flatnonzero(np.diff(codigos[orden])) + 1
        indices_por_grupo = np.split(orden, cortes)   # posiciones de las personas de cada conglomerado
        k = len(indices_por_grupo)

    replicas = np.empty(n_rep)
    for b in range(n_rep):
        if conglomerados is None:
            idx = rng.integers(0, n, size=n)
        else:
            elegidos = rng.integers(0, k, size=k)
            idx = np.concatenate([indices_por_grupo[g] for g in elegidos])
        replicas[b] = estadistico(*[a[idx] for a in arreglos])

    cola = (1 - nivel) / 2
    inf, sup = np.quantile(replicas, [cola, 1 - cola])
    return {
        "estimacion": float(estadistico(*arreglos)),
        "inf": float(inf),
        "sup": float(sup),
        "error_estandar": float(replicas.std(ddof=1)),
        "replicas": replicas,
    }


def p_desde_error_estandar(estimacion: float, error_estandar: float, valor_nulo: float = 0.0,
                           alternativa: str = "bilateral") -> tuple[float, float]:
    """Prueba de Wald: z = (estimación - valor nulo) / EE, con su valor p.

    Se usa para obtener un valor p robusto a conglomerados a partir del error
    estándar bootstrap por conglomerados.
    """
    z = (estimacion - valor_nulo) / error_estandar
    if alternativa == "bilateral":
        p = 2 * stats.norm.sf(abs(z))
    elif alternativa == "mayor":
        p = stats.norm.sf(z)
    elif alternativa == "menor":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternativa debe ser 'bilateral', 'mayor' o 'menor'")
    return float(z), float(p)


def diferencia_proporciones_robusta(resultado, grupo, conglomerados,
                                    nivel: float = NIVEL_CONFIANZA) -> dict:
    """Diferencia de proporciones (grupo=1 menos grupo=0) con error estándar robusto.

    Una regresión lineal de la variable binaria sobre el indicador de grupo tiene
    como pendiente exactamente la diferencia de proporciones. Al pedir la matriz
    de covarianzas agrupada por conglomerado, el error estándar deja de suponer
    que las personas de una misma UPM son independientes.
    """
    y = np.asarray(resultado, dtype=float)
    x = sm.add_constant(np.asarray(grupo, dtype=float))
    ajuste = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(np.asarray(conglomerados))[0]})
    diferencia, ee = ajuste.params[1], ajuste.bse[1]
    z = diferencia / ee
    critico = stats.norm.ppf(1 - (1 - nivel) / 2)
    return {
        "diferencia": float(diferencia),
        "error_estandar": float(ee),
        "z": float(z),
        "p": float(2 * stats.norm.sf(abs(z))),
        "inf": float(diferencia - critico * ee),
        "sup": float(diferencia + critico * ee),
    }
