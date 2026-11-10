"""Cruce ecológico departamento: enfermedad (personas, 2024) / NBI (hogares, 2022)."""
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .comun import RAIZ, SALIDA, OBJETIVO, ponderada

ARCHIVO = RAIZ / "data/externos/nbi_departamental_2022.csv"
DIMENSIONES = ["nbi_una_o_mas_pct", "nbi_vivienda_pct", "nbi_sanitaria_pct",
               "nbi_educacion_pct", "nbi_subsistencia_pct"]

def normalizar(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()

def ejecutar(df):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fuente = pd.read_csv(ARCHIVO)
    if len(fuente) != 18 or fuente.departamento.duplicated().any():
        raise ValueError("La tabla del NBI debe incluir Asunción y 17 departamentos.")
    filas = []
    for d, grupo in df.groupby("departamento", dropna=False):
        filas.append({"departamento_ephc": d, "clave": normalizar(d),
                      "encuestados": len(grupo),
                      "prevalencia_2024_pct": 100 * ponderada(grupo[OBJETIVO], grupo.factor_expansion)})
    epi = pd.DataFrame(filas)
    fuente["clave"] = fuente.departamento.map(normalizar)
    merged = epi.merge(fuente, how="left", on="clave", validate="one_to_one", indicator=True)
    if not (merged._merge == "both").all():
        raise ValueError(f"Departamentos EPHC sin NBI: {merged.loc[merged._merge != 'both', 'departamento_ephc'].tolist()}")
    merged = merged.drop(columns=["_merge", "clave"])
    merged.to_csv(SALIDA / "cruce_departamental_nbi.csv", index=False)
    correlaciones = []
    for col in DIMENSIONES:
        rho, p = spearmanr(merged[col], merged.prevalencia_2024_pct)
        correlaciones.append({"indicador_nbi": col, "n_departamentos": len(merged),
                              "rho_spearman": float(rho), "p_exploratorio": float(p)})
    pd.DataFrame(correlaciones).to_csv(SALIDA / "correlaciones_nbi.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(merged.nbi_una_o_mas_pct, merged.prevalencia_2024_pct)
    for _, r in merged.iterrows():
        ax.annotate(str(r.departamento_ephc), (r.nbi_una_o_mas_pct, r.prevalencia_2024_pct),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Hogares con al menos una NBI en 2022 (%)")
    ax.set_ylabel("Personas que declararon enfermedad en 2024 (%)")
    ax.set_title("Cruce exploratorio NBI–enfermedad por departamento")
    fig.tight_layout()
    fig.savefig(SALIDA / "nbi_vs_enfermedad_departamento.png", dpi=180)
    plt.close(fig)
    return {"departamentos": len(merged), "correlaciones": correlaciones,
            "advertencia": "Cruce ecológico 2022/2024: no implica causalidad ni riesgo individual."}
