"""Clasificación binaria con validación temporal y comparación contra baseline."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from .comun import NUMERICAS, CATEGORICAS, OBJETIVO, SALIDA, SEMILLA, ponderada, preprocesador

CONFIGS = {
    "logistica": [{"C": .3, "max_iter": 600}, {"C": 3., "max_iter": 600}],
    "bosque": [
        {"n_estimators": 110, "max_depth": 10, "min_samples_leaf": 35, "n_jobs": -1, "random_state": SEMILLA},
        {"n_estimators": 110, "max_depth": 18, "min_samples_leaf": 35, "n_jobs": -1, "random_state": SEMILLA},
    ],
    "naive_bayes": [{"var_smoothing": 1e-8}, {"var_smoothing": 1e-6}],
}

def construir(nombre, config):
    clases = {"logistica": LogisticRegression, "bosque": RandomForestClassifier,
              "naive_bayes": GaussianNB}
    return Pipeline([("preparar", preprocesador()), ("modelo", clases[nombre](**config))])

def entrenar(modelo, df):
    return modelo.fit(
        df[NUMERICAS+CATEGORICAS], df[OBJETIVO].astype(int),
        modelo__sample_weight=df.factor_expansion.to_numpy(dtype=float),
    )

def metricas(modelo, df, etiqueta):
    y = df[OBJETIVO].astype(int).to_numpy()
    w = df.factor_expansion.to_numpy(dtype=float)
    x = df[NUMERICAS+CATEGORICAS]
    pred = modelo.predict(x)
    proba = modelo.predict_proba(x)[:, 1]
    return {
        "muestra": etiqueta, "n": len(df), "prevalencia_ponderada": ponderada(y, w),
        "accuracy": accuracy_score(y, pred, sample_weight=w),
        "balanced_accuracy": balanced_accuracy_score(y, pred, sample_weight=w),
        "precision": precision_score(y, pred, sample_weight=w, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=w, zero_division=0),
        "f1": f1_score(y, pred, sample_weight=w, zero_division=0),
        "auc_roc": roc_auc_score(y, proba, sample_weight=w) if len(np.unique(y)) > 1 else np.nan,
    }

def ejecutar(df):
    SALIDA.mkdir(parents=True, exist_ok=True)
    # Origen expansivo: T1->T2 y (T1,T2)->T3. No se usa T4 para seleccionar.
    validacion = []
    for nombre, configs in CONFIGS.items():
        for j, config in enumerate(configs):
            for hasta, siguiente in ((1, 2), (2, 3)):
                tr = df[df.trimestre <= hasta]
                va = df[df.trimestre == siguiente]
                modelo = construir(nombre, config)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    entrenar(modelo, tr)
                validacion.append({"modelo": nombre, "config_id": j,
                    "params": json.dumps(config), "train_trimestres": f"T1..T{hasta}",
                    **metricas(modelo, va, f"T{siguiente}")})
    tabla = pd.DataFrame(validacion)
    tabla.to_csv(SALIDA/"validacion_temporal.csv", index=False)
    candidatos = tabla.groupby(["modelo", "config_id"], as_index=False)["auc_roc"].mean()
    elegido = candidatos.sort_values(["auc_roc","modelo"], ascending=[False,True]).iloc[0]
    ganador, indice = str(elegido.modelo), int(elegido.config_id)
    train = df[df.trimestre <= 3]
    test = df[df.trimestre == 4]
    modelos = {"baseline": Pipeline([
        ("preparar", preprocesador()), ("modelo", DummyClassifier(strategy="most_frequent")),
    ])}
    for nombre in CONFIGS:
        mejor = candidatos[candidatos.modelo == nombre].sort_values("auc_roc", ascending=False).iloc[0]
        modelos[nombre] = construir(nombre, CONFIGS[nombre][int(mejor.config_id)])
    resultados = []
    for nombre, modelo in modelos.items():
        entrenar(modelo, train)
        for sub, etiqueta in ((train,"entrenamiento"), (test,"prueba_T4")):
            resultados.append({"modelo": nombre, **metricas(modelo, sub, etiqueta)})
    pd.DataFrame(resultados).to_csv(SALIDA/"metricas_clasificacion.csv", index=False)

    modelo = modelos[ganador]
    joblib.dump({"modelo": modelo, "columnas": NUMERICAS+CATEGORICAS,
        "ganador": ganador, "periodo_entrenado": "T1-T3_2024"},
        SALIDA/"clasificador.joblib", compress=3)
    pred = modelo.predict(test[NUMERICAS+CATEGORICAS])
    y = test[OBJETIVO].astype(int).to_numpy()
    w = test.factor_expansion.to_numpy(dtype=float)
    cm = confusion_matrix(y, pred, labels=[0,1], sample_weight=w)
    fig, ax = plt.subplots(figsize=(6.7,4.8))
    ConfusionMatrixDisplay(cm, display_labels=["Sano/a","Enfermo/a"]).plot(
        ax=ax, cmap="Blues", values_format=".0f", colorbar=False)
    ax.set_title(f"Clasificación T4 — {ganador} (ponderado)")
    fig.tight_layout()
    fig.savefig(SALIDA/"matriz_confusion.png", dpi=180)
    plt.close(fig)
    errores = []
    for atributo in ("sexo","area","quintil_ingreso","departamento"):
        for grupo, sub in test.groupby(atributo, dropna=False):
            if len(sub)<30:
                continue
            sub_pred = pred[test.index.get_indexer(sub.index)]
            errores.append({"variable":atributo,"categoria":str(grupo),"n":len(sub),
                "prevalencia_real": ponderada(sub[OBJETIVO],sub.factor_expansion),
                "prevalencia_predicha":ponderada(sub_pred,sub.factor_expansion),
                "recall":recall_score(sub[OBJETIVO],sub_pred,sample_weight=sub.factor_expansion,zero_division=0)})
    pd.DataFrame(errores).to_csv(SALIDA/"errores_subgrupos.csv",index=False)
    with (SALIDA/"seleccion_modelo.json").open("w",encoding="utf-8") as f:
        json.dump({"modelo":ganador,"config":CONFIGS[ganador][indice],
            "criterio":"AUC-ROC ponderado promedio en validación T2 y T3",
            "auc_validacion":float(elegido.auc_roc),
            "advertencia":"Prueba T4 reservada hasta selección final."},
            f,ensure_ascii=False,indent=2)
    return {"modelo":ganador,"validacion_auc":float(elegido.auc_roc),
        "test": next(x for x in resultados if x["modelo"]==ganador and x["muestra"]=="prueba_T4")}
