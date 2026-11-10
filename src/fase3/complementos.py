"""Análisis complementarios de la Fase 3 para el informe.

No modifica el pipeline oficial (`ejecutar.py`): reutiliza sus funciones y escribe
en `output/fase3/complementos/`. Cubre los puntos de la guía que el pipeline no
reporta:

1. Búsqueda en grilla de hiperparámetros con la misma validación temporal
   (T1 -> T2 y T1+T2 -> T3), sin usar T4.
2. Brecha de sobreajuste: AUC en entrenamiento, validación y prueba.
3. Importancia de variables por permutación (caída del AUC ponderado en T4).
4. Regresión logística interpretable (OR con errores robustos por UPM).
5. Análisis de errores: matriz de confusión ponderada, errores por subgrupo y
   los casos peor predichos.

Uso: python -m src.fase3.complementos   (requiere haber ejecutado src.fase3.ejecutar)
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from . import ejecutar as ej

OUT = ej.OUT / "complementos"
GRILLAS = {
    "Regresion_logistica": {"C": [0.01, 0.1, 1.0, 10.0]},
    "Random_Forest": {"max_depth": [8, 13, None], "min_samples_leaf": [10, 35, 100]},
    "Hist_Gradient_Boosting": {"learning_rate": [0.03, 0.06, 0.1], "max_leaf_nodes": [7, 15, 31]},
}
OFICIAL = {
    "Regresion_logistica": {"C": 1.0},
    "Random_Forest": {"max_depth": 13, "min_samples_leaf": 35},
    "Hist_Gradient_Boosting": {"learning_rate": 0.06, "max_leaf_nodes": 15},
}


def construir(nombre: str, params: dict):
    if nombre == "Regresion_logistica":
        modelo = LogisticRegression(max_iter=700, **params)
    elif nombre == "Random_Forest":
        modelo = RandomForestClassifier(n_estimators=120, random_state=ej.SEED, n_jobs=-1, **params)
    else:
        modelo = HistGradientBoostingClassifier(max_iter=150, l2_regularization=4, random_state=ej.SEED, **params)
    return Pipeline([("procesar", ej.preprocessing(ej.NUM, ej.CAT)), ("modelo", modelo)])


def auc_ponderado(modelo, datos):
    p = modelo.predict_proba(datos[ej.COLUMNS])[:, 1]
    return float(roc_auc_score(datos[ej.LABEL], p, sample_weight=datos.peso))


def grilla(df):
    filas = []
    for nombre, grid in GRILLAS.items():
        claves = list(grid)
        for valores in itertools.product(*grid.values()):
            params = dict(zip(claves, valores))
            aucs = []
            for fin, siguiente in [(1, 2), (2, 3)]:
                tr, va = df[df.trimestre <= fin], df[df.trimestre == siguiente]
                m = construir(nombre, params).fit(tr[ej.COLUMNS], tr[ej.LABEL], modelo__sample_weight=tr.peso)
                aucs.append(auc_ponderado(m, va))
            filas.append({"modelo": nombre, "parametros": json.dumps(params), "auc_T2": aucs[0], "auc_T3": aucs[1],
                          "auc_validacion": float(np.mean(aucs)), "es_oficial": params == OFICIAL[nombre]})
    return pd.DataFrame(filas).sort_values(["modelo", "auc_validacion"], ascending=[True, False])


def importancia(modelo, prueba, n_rep=10):
    rng = np.random.default_rng(ej.SEED)
    base = auc_ponderado(modelo, prueba)
    filas = []
    for var in ej.COLUMNS:
        caidas = []
        for _ in range(n_rep):
            x = prueba.copy()
            x[var] = prueba[var].to_numpy()[rng.permutation(len(prueba))]
            caidas.append(base - auc_ponderado(modelo, x))
        filas.append({"variable": var, "caida_auc": float(np.mean(caidas)), "desvio": float(np.std(caidas, ddof=1))})
    return pd.DataFrame(filas).sort_values("caida_auc", ascending=False)


def logistica_interpretable(dev):
    d = dev.copy()
    d["anios_estudio"] = d["anios_estudio"].fillna(d["anios_estudio"].median())
    d["log_ingreso"] = d["log_ingreso"].fillna(d["log_ingreso"].median())
    d["edad_10"] = d["edad"] / 10
    d["log2_ingreso"] = d["log_ingreso"] / np.log(2)      # +1 = el doble de ingreso
    f = ("sin_seguro ~ edad_10 + anios_estudio + log2_ingreso + C(sexo, Treatment('Hombre'))"
         " + C(area, Treatment('Urbana')) + C(departamento, Treatment('Asunción'))")
    m = smf.logit(f, data=d).fit(disp=False, cov_type="cluster", cov_kwds={"groups": d["UPM"]})
    ic = m.conf_int()
    return pd.DataFrame({"OR": np.exp(m.params), "IC95_inf": np.exp(ic[0]), "IC95_sup": np.exp(ic[1]), "p": m.pvalues})


def errores(modelo, prueba):
    p = modelo.predict_proba(prueba[ej.COLUMNS])[:, 1]
    e = prueba.assign(prob=p, pred=(p >= 0.5).astype(int))
    e["error_abs"] = (e[ej.LABEL] - e["prob"]).abs()
    e["tipo"] = np.select([(e.sin_seguro == 1) & (e.pred == 1), (e.sin_seguro == 0) & (e.pred == 0),
                           (e.sin_seguro == 0) & (e.pred == 1)], ["VP", "VN", "FP"], "FN")
    w = e.peso
    matriz = e.groupby("tipo").peso.sum() / w.sum() * 100
    e["quintil_muestra"] = pd.qcut(e.ingreso_per_capita.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    e["grupo_edad"] = pd.cut(e.edad, [17, 29, 44, 59, 130], labels=["18-29", "30-44", "45-59", "60+"])
    sub = []
    for var in ["area", "sexo", "grupo_edad", "quintil_muestra"]:
        for cat, g in e.groupby(var, observed=True):
            ws = g.peso
            sin = g[g.sin_seguro == 1]; con = g[g.sin_seguro == 0]
            sub.append({"variable": var, "categoria": str(cat), "n": len(g),
                        "sin_seguro_pct": 100 * np.average(g.sin_seguro, weights=ws),
                        "acierto_pct": 100 * np.average(g.sin_seguro == g.pred, weights=ws),
                        "detecta_sin_seguro_pct": 100 * np.average(sin.pred == 1, weights=sin.peso) if len(sin) else np.nan,
                        "detecta_con_seguro_pct": 100 * np.average(con.pred == 0, weights=con.peso) if len(con) else np.nan})
    # Casos peor predichos: el 1 % con mayor error absoluto
    corte = e.error_abs.quantile(0.99)
    peores = e[e.error_abs >= corte]
    perfil = lambda g: pd.Series({"n": len(g), "edad_media": g.edad.mean(), "anios_estudio_media": g.anios_estudio.mean(),
                                  "ingreso_mediano": g.ingreso_per_capita.median(), "rural_pct": 100 * (g.area == "Rural").mean(),
                                  "mujer_pct": 100 * (g.sexo == "Mujer").mean(), "prob_media": g.prob.mean()})
    comp = pd.DataFrame({"Todos T4": perfil(e), "Peores FP (tenía seguro)": perfil(peores[peores.tipo == "FP"]),
                         "Peores FN (no tenía seguro)": perfil(peores[peores.tipo == "FN"])}).T
    tipo_seguro = (peores[peores.tipo == "FP"].S01A.map({1: "IPS", 2: "Privado individual", 3: "Privado laboral", 4: "Privado familiar",
                                                         5: "Sanidad Militar", 6: "Sanidad Policial"}).value_counts(normalize=True) * 100)
    fp_todos = (e[e.tipo == "FP"].S01A.map({1: "IPS", 2: "Privado individual", 3: "Privado laboral", 4: "Privado familiar",
                                           5: "Sanidad Militar", 6: "Sanidad Policial"}).value_counts(normalize=True) * 100)
    ejemplos = peores.sort_values("error_abs", ascending=False).head(10)[
        ["tipo", "prob", "edad", "sexo", "area", "departamento", "anios_estudio", "ingreso_per_capita", "S01A"]]
    return matriz, pd.DataFrame(sub), comp, tipo_seguro, fp_todos, ejemplos, float(corte)


def cobertura_por_quintil(df):
    """Cobertura ponderada de adultos por quintil de ingreso, con dos definiciones de quintil.

    - quintil_oficial_INE: variable quintili de la REG02 (quintiles de la población total, ponderados).
    - quintil_muestral_adultos: quintiles con igual número de adultos de la muestra, que es la
      definición usada en el piloto y en la presentación de la Fase 3.
    """
    claves = ["UPM", "NVIVI", "NHOGA", "L02"]
    raw = pd.read_csv(ej.RAW, sep=";", encoding="utf-8-sig", dtype=str, low_memory=False)[claves + ["quintili"]]
    raw[claves] = raw[claves].apply(pd.to_numeric)
    d = df.merge(raw, on=claves, how="left")
    d["quintil_oficial_INE"] = pd.to_numeric(d.quintili, errors="coerce")
    d["quintil_muestral_adultos"] = pd.qcut(d.ingreso_per_capita.rank(method="first"), 5, labels=False) + 1
    filas = []
    for definicion in ["quintil_oficial_INE", "quintil_muestral_adultos"]:
        for q, g in d.dropna(subset=[definicion]).groupby(definicion):
            filas.append({"definicion": definicion, "quintil": int(q), "adultos": len(g),
                          "con_seguro_pct": 100 * (1 - np.average(g.sin_seguro, weights=g.peso))})
    return pd.DataFrame(filas)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, _ = ej.load()
    dev, test = df[df.trimestre < 4].copy(), df[df.trimestre == 4].copy()

    cobertura_por_quintil(df).to_csv(OUT / "cobertura_por_quintil.csv", index=False)

    g = grilla(df)
    g.to_csv(OUT / "grilla_hiperparametros.csv", index=False)
    mejores = g.groupby("modelo").head(1)
    oficiales = g[g.es_oficial]

    ganador, cols = ej.fit_model("Hist_Gradient_Boosting", dev, False)
    brecha = []
    rank = pd.read_csv(ej.OUT / "ranking_validacion.csv")
    comp = pd.read_csv(ej.OUT / "comparacion_test.csv")
    for nombre in ["Regresion_logistica", "Random_Forest", "Hist_Gradient_Boosting"]:
        c = comp[(comp.modelo == nombre) & (~comp.con_cluster)].set_index("conjunto").auc_roc
        v = rank[(rank.modelo == nombre) & (~rank.con_cluster)].auc_validacion.iloc[0]
        brecha.append({"modelo": nombre, "auc_entrenamiento": c["entrenamiento"], "auc_validacion": v, "auc_prueba_T4": c["prueba_T4"],
                       "brecha_entrenamiento_validacion": c["entrenamiento"] - v, "brecha_validacion_prueba": v - c["prueba_T4"]})
    pd.DataFrame(brecha).to_csv(OUT / "brecha_sobreajuste.csv", index=False)

    imp = importancia(ganador, test)
    imp.to_csv(OUT / "importancia_permutacion.csv", index=False)
    rl, _ = ej.fit_model("Regresion_logistica", dev, False)
    imp_rl = importancia(rl, test)
    imp_rl.to_csv(OUT / "importancia_permutacion_logistica.csv", index=False)

    ors = logistica_interpretable(dev)
    ors.to_csv(OUT / "logistica_or.csv")

    matriz, sub, perfil_peores, tipo_fp_peores, tipo_fp, ejemplos, corte = errores(ganador, test)
    matriz.to_csv(OUT / "matriz_confusion_pct.csv")
    sub.to_csv(OUT / "errores_subgrupos.csv", index=False)
    perfil_peores.to_csv(OUT / "perfil_peores_casos.csv")
    pd.DataFrame({"peores_FP": tipo_fp_peores, "todos_FP": tipo_fp}).to_csv(OUT / "tipo_seguro_falsos_positivos.csv")
    ejemplos.to_csv(OUT / "ejemplos_peores_casos.csv", index=False)

    print(mejores.round(4).to_string()); print(oficiales.round(4).to_string())
    print(pd.DataFrame(brecha).round(4).to_string()); print(imp.round(4).to_string()); print(imp_rl.round(4).to_string())
    print(ors.round(4).to_string()); print(matriz.round(2)); print(sub.round(1).to_string())
    print("corte peor 1 %:", round(corte, 3)); print(perfil_peores.round(2).to_string())
    print(pd.DataFrame({"peores_FP": tipo_fp_peores, "todos_FP": tipo_fp}).round(1)); print(ejemplos.round(2).to_string())


if __name__ == "__main__":
    main()
