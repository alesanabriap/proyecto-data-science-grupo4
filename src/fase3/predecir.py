"""python -m src.fase3.predecir --entrada nuevos.csv --salida predicciones.csv"""
import argparse
import joblib
import pandas as pd
from .comun import SALIDA

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--entrada",required=True)
    parser.add_argument("--salida",default="predicciones.csv")
    args=parser.parse_args()
    paquete=joblib.load(SALIDA/"clasificador.joblib")
    datos=pd.read_csv(args.entrada)
    faltan=set(paquete["columnas"])-set(datos.columns)
    if faltan:
        raise ValueError(f"Columnas no encontradas: {sorted(faltan)}")
    for c in ("sexo","area","tiene_seguro","estado_civil","departamento"):
        datos[c]=datos[c].fillna("No declarado").astype(str)
    p=paquete["modelo"].predict_proba(datos[paquete["columnas"]])[:,1]
    datos["probabilidad_declaracion_enfermedad"]=p
    datos["clase_estimada"]=(p>=.5).astype(int)
    datos.to_csv(args.salida,index=False)
    print(f"Guardadas {len(datos)} predicciones en {args.salida}")

if __name__ == "__main__":
    main()
