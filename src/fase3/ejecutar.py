"""Desde la raíz: python -m src.fase3.ejecutar."""
import json
from .comun import SALIDA, cargar, validar_division_temporal
from . import perfiles, clasificacion, nbi

def main():
    df = cargar()
    validar_division_temporal(df)
    resultados = {
        "n_registros": int(len(df)),
        "objetivo": "Enfermedad autorreportada en últimos 90 días, no incidencia futura",
        "perfiles": perfiles.ejecutar(df),
        "clasificacion": clasificacion.ejecutar(df),
        "nbi_departamental": nbi.ejecutar(df),
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    with (SALIDA/"resumen_ejecucion.json").open("w",encoding="utf-8") as f:
        json.dump(resultados,f,ensure_ascii=False,indent=2)
    print(json.dumps(resultados,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
