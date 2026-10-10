"""Parte 3. Consolidación: comparaciones múltiples y tabla resumen de contrastes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .datos import ALFA, DIR_SALIDA, N_CONTRASTES, formatear_p


def cargar_contrastes(partes=(1, 2, 3)) -> pd.DataFrame:
    """Une los resultados que cada parte guardó en output/fase2."""
    tablas = []
    for parte in partes:
        ruta = DIR_SALIDA / f"contrastes_parte{parte}.csv"
        if not ruta.exists():
            raise FileNotFoundError(
                f"Falta {ruta.name}. Ejecute antes el notebook de la Parte {parte}."
            )
        tablas.append(pd.read_csv(ruta))
    contrastes = pd.concat(tablas, ignore_index=True).sort_values("id").reset_index(drop=True)
    if len(contrastes) != N_CONTRASTES:
        raise ValueError(
            f"Se declararon {N_CONTRASTES} contrastes confirmatorios y se encontraron {len(contrastes)}."
        )
    return contrastes


def ajustar_p(valores_p, metodo: str) -> np.ndarray:
    """Valores p ajustados. ``metodo``: 'bonferroni', 'holm' o 'fdr_bh' (Benjamini-Hochberg)."""
    return multipletests(np.asarray(valores_p, dtype=float), method=metodo)[1]


def aplicar_correcciones(contrastes: pd.DataFrame, alfa: float = ALFA) -> pd.DataFrame:
    """Agrega los valores p ajustados y la decisión de cada contraste.

    Regla declarada de antemano: la decisión se toma con el valor p robusto a
    conglomerados, ajustado por Bonferroni sobre la familia de contrastes
    confirmatorios. Los ajustes de Holm y de Benjamini-Hochberg y el valor p
    clásico se informan solo para transparencia.
    """
    tabla = contrastes.copy()
    tabla["p_clasico_bonferroni"] = ajustar_p(tabla["p_clasico"], "bonferroni")
    tabla["p_robusto_bonferroni"] = ajustar_p(tabla["p_robusto"], "bonferroni")
    tabla["p_robusto_holm"] = ajustar_p(tabla["p_robusto"], "holm")
    tabla["p_robusto_bh"] = ajustar_p(tabla["p_robusto"], "fdr_bh")
    tabla["decision"] = np.where(
        tabla["p_robusto_bonferroni"] < alfa, "Se rechaza H0", "No se rechaza H0"
    )
    tabla["decision_clasica"] = np.where(
        tabla["p_clasico_bonferroni"] < alfa, "Se rechaza H0", "No se rechaza H0"
    )
    return tabla


def _numero(x: float, decimales: int = 3) -> str:
    return f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def tabla_para_informe(tabla: pd.DataFrame) -> pd.DataFrame:
    """Versión legible de la tabla resumen (la que se muestra y se lleva al informe)."""
    filas = []
    for _, f in tabla.iterrows():
        gl = "" if pd.isna(f["gl"]) else f" (gl = {int(f['gl'])})"
        filas.append({
            "Hipótesis": f"{f['id']}. {f['hipotesis']}",
            "Prueba": f["prueba"],
            "Supuestos verificados": f["supuestos"],
            "Estadístico": f"{f['estadistico']} = {_numero(f['valor_estadistico'], 2)}{gl}",
            "p clásico": formatear_p(f["p_clasico"]),
            "p robusto": formatear_p(f["p_robusto"]),
            "p robusto (Bonferroni)": formatear_p(f["p_robusto_bonferroni"]),
            "IC 95 %": f"{f['ic_parametro']}: [{_numero(f['ic_inf'])}; {_numero(f['ic_sup'])}] ({f['ic_metodo']})",
            "Tamaño del efecto": f"{f['efecto']} = {_numero(f['valor_efecto'])} ({f['magnitud']})",
            "Decisión": f["decision"],
        })
    return pd.DataFrame(filas)


def exportar(tabla: pd.DataFrame, legible: pd.DataFrame) -> dict[str, Path]:
    """Guarda la tabla completa (CSV) y la legible (CSV y Markdown)."""
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    rutas = {
        "completa": DIR_SALIDA / "tabla_resumen_contrastes.csv",
        "legible_csv": DIR_SALIDA / "tabla_resumen_contrastes_legible.csv",
        "legible_md": DIR_SALIDA / "tabla_resumen_contrastes.md",
    }
    tabla.to_csv(rutas["completa"], index=False)
    legible.to_csv(rutas["legible_csv"], index=False)
    encabezado = "| " + " | ".join(legible.columns) + " |"
    separador = "|" + "|".join(["---"] * len(legible.columns)) + "|"
    cuerpo = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in legible.to_numpy()]
    rutas["legible_md"].write_text("\n".join([encabezado, separador, *cuerpo]) + "\n", encoding="utf-8")
    return rutas
