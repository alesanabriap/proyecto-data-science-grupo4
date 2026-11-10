"""Carga, validación del diseño y preprocesamiento sin fuga de información."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "data/processed/ephc_salud_2024_limpio.csv"
CARPETA_TEMP = RAIZ / "data/temporal"
SALIDA = RAIZ / "output/fase3"
CLAVE = ["UPM", "NVIVI", "NHOGA", "L02"]
OBJETIVO = "enfermedad_reciente"
NUMERICAS = ["edad", "quintil_ingreso", "anios_estudio"]
CATEGORICAS = ["sexo", "area", "tiene_seguro", "estado_civil", "departamento"]
PERFIL_NUM = ["edad", "quintil_ingreso", "anios_estudio"]
PERFIL_CAT = ["sexo", "area", "tiene_seguro"]
SEMILLA = 42

def cargar() -> pd.DataFrame:
    df = pd.read_csv(ENTRADA, encoding="utf-8-sig", low_memory=False)
    faltan = set(CLAVE + [OBJETIVO, "factor_expansion", "departamento"] + NUMERICAS + CATEGORICAS) - set(df.columns)
    if faltan:
        raise ValueError(f"Columnas faltantes en EPHC procesada: {sorted(faltan)}")
    if df[OBJETIVO].isna().any() or not df[OBJETIVO].isin([0, 1]).all():
        raise ValueError("La variable enfermedad_reciente debe ser binaria y completa.")
    if (df["factor_expansion"] <= 0).any() or df["factor_expansion"].isna().any():
        raise ValueError("factor_expansion debe ser positivo y estar completo.")
    partes = []
    for t in range(1, 5):
        archivo = CARPETA_TEMP / f"REG02_EPHC_2024_T{t}.csv"
        claves = pd.read_csv(archivo, sep=";", encoding="utf-8-sig", usecols=CLAVE)
        claves["trimestre"] = t
        partes.append(claves)
    entrevista = pd.concat(partes, ignore_index=True)
    if entrevista.duplicated(CLAVE).any():
        raise ValueError("Claves de personas duplicadas entre trimestres.")
    df = df.merge(entrevista, on=CLAVE, how="left", validate="one_to_one")
    if df["trimestre"].isna().any():
        raise ValueError("Hay personas sin trimestre de entrevista.")
    df["trimestre"] = df["trimestre"].astype(int)
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAS:
        df[c] = df[c].fillna("No declarado").astype(str)
    df["upm_global"] = df["UPM"].astype(str)
    return df

def validar_division_temporal(df: pd.DataFrame):
    if set(df["trimestre"]) != {1, 2, 3, 4}:
        raise AssertionError("Se esperan cuatro trimestres de la EPHC 2024.")
    train = set(df.loc[df.trimestre < 4, "upm_global"])
    test = set(df.loc[df.trimestre == 4, "upm_global"])
    if train & test:
        raise AssertionError("UPM presentes en entrenamiento y prueba.")

def ponderada(y, w) -> float:
    return float(np.average(np.asarray(y, dtype=float), weights=np.asarray(w, dtype=float)))

def preprocesador(numericas=None, categoricas=None):
    numericas = list(NUMERICAS if numericas is None else numericas)
    categoricas = list(CATEGORICAS if categoricas is None else categoricas)
    num = Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                    ("escalar", StandardScaler())])
    cat = Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                    ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, numericas), ("cat", cat, categoricas)], sparse_threshold=0)
