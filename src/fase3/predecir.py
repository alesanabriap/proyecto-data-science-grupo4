"""Inferencia del modelo oficial de Fase 3.

Uso:
  python -m src.fase3.predecir --entrada data/nuevos_adultos.csv --salida predicciones.csv

Entrada: variables con nombres de columnas como el dataset modelado:
edad, anios_estudio, log_ingreso, sexo, area, departamento.
Si solo existe ingreso_per_capita en guaraníes, se calcula log1p automáticamente.

La salida estima ausencia de cobertura médica; NO diagnostica ni predice enfermedad.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .ejecutar import OUT, NUM, CAT, SEGM

def predecir(entrada: str, salida: str, modelo_ruta: str | None = None) -> int:
    path=Path(modelo_ruta) if modelo_ruta else OUT/"mejor_clasificador.joblib"
    paquete=joblib.load(path)
    df=pd.read_csv(entrada)
    if "log_ingreso" not in df and "ingreso_per_capita" in df:
        ingreso=pd.to_numeric(df["ingreso_per_capita"], errors="coerce")
        if ingreso.dropna().lt(0).any():
            raise ValueError("El ingreso per cápita no puede ser negativo.")
        df["log_ingreso"]=np.log1p(ingreso)
    faltantes=sorted(set(paquete["features"])-set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas para inferencia: {faltantes}")
    if bool(paquete.get("cluster",False)):
        raise ValueError("Modelo con cluster: requiere asignación de perfiles; no soportado por este script.")
    modelo=paquete["model"]
    prob=modelo.predict_proba(df[paquete["features"]])[:,1]
    df["probabilidad_sin_seguro"]=prob
    df["clasificacion_sin_seguro"]=(prob>=0.5).astype(int)
    df.to_csv(salida,index=False)
    return len(df)

def main():
    parser=argparse.ArgumentParser(description="Inferencia ausencia de seguro — Fase 3")
    parser.add_argument("--entrada",required=True)
    parser.add_argument("--salida",default="predicciones_cobertura.csv")
    parser.add_argument("--modelo",default=None)
    args=parser.parse_args()
    n=predecir(args.entrada,args.salida,args.modelo)
    print(f"{n} registros procesados. Resultado: {args.salida}")

if __name__=="__main__":
    main()
