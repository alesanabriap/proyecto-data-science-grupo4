"""Clustering exploratorio sin utilizar la respuesta enfermedad."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from .comun import SALIDA, SEMILLA, PERFIL_NUM, PERFIL_CAT, OBJETIVO, ponderada, preprocesador

ATRIBUTOS = PERFIL_NUM + PERFIL_CAT

def ejecutar(df):
    SALIDA.mkdir(parents=True, exist_ok=True)
    tr, te = df[df.trimestre < 4].copy(), df[df.trimestre == 4].copy()
    prep = preprocesador(PERFIL_NUM, PERFIL_CAT)
    X = prep.fit_transform(tr[ATRIBUTOS])
    X_test = prep.transform(te[ATRIBUTOS])
    ix = np.random.default_rng(SEMILLA).choice(len(tr), size=min(2200, len(tr)), replace=False)
    scores, modelos = [], {}
    for k in (3, 4, 5, 6):
        m = MiniBatchKMeans(n_clusters=k, random_state=SEMILLA, n_init=5,
                            batch_size=2048, max_iter=160)
        m.fit(X)
        labels = m.predict(X[ix])
        score = silhouette_score(X[ix], labels) if len(set(labels)) > 1 else np.nan
        scores.append({"k": k, "silhouette": float(score)})
        modelos[k] = m
    comparacion = pd.DataFrame(scores)
    comparacion.to_csv(SALIDA / "silhouette_k.csv", index=False)
    k = int(comparacion.sort_values("silhouette", ascending=False).iloc[0].k)
    m = modelos[k]
    tr["perfil"], te["perfil"] = m.predict(X), m.predict(X_test)
    pd.concat([tr[["persona_id", "trimestre", "perfil"]],
               te[["persona_id", "trimestre", "perfil"]]], ignore_index=True).to_csv(
        SALIDA / "perfiles_por_persona.csv", index=False)
    resumen = []
    for data, conjunto in ((tr, "entrenamiento"), (te, "prueba_T4")):
        for perfil, grupo in data.groupby("perfil"):
            w = grupo.factor_expansion
            resumen.append({
                "conjunto": conjunto, "perfil": int(perfil), "n": len(grupo),
                "peso_poblacional": float(w.sum()),
                "enfermedad_pct": 100 * ponderada(grupo[OBJETIVO], w),
                "edad_media": ponderada(grupo.edad, w),
                "quintil_media": ponderada(grupo.quintil_ingreso.fillna(grupo.quintil_ingreso.median()), w),
                "seguro_pct": 100 * ponderada((grupo.tiene_seguro == "1").astype(int), w),
                "rural_pct": 100 * ponderada((grupo.area == "Rural").astype(int), w),
            })
    tabla = pd.DataFrame(resumen)
    tabla.to_csv(SALIDA / "resumen_perfiles.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    sub = tabla[tabla.conjunto == "prueba_T4"].sort_values("enfermedad_pct")
    ax.barh([f"Perfil {int(i)}" for i in sub.perfil], sub.enfermedad_pct)
    ax.set_xlabel("Enfermedad reciente autorreportada (%) — ponderado")
    ax.set_title("Enfermedad según perfil sociodemográfico · T4 2024")
    fig.tight_layout()
    fig.savefig(SALIDA / "enfermedad_por_perfil.png", dpi=180)
    plt.close(fig)
    joblib.dump({"preparador": prep, "kmeans": m, "columnas": ATRIBUTOS},
                SALIDA / "segmentador.joblib", compress=3)
    return {"clusters": k, "silhouette": float(comparacion[comparacion.k == k].iloc[0].silhouette),
            "perfiles_test": tabla[tabla.conjunto == "prueba_T4"].to_dict(orient="records")}
