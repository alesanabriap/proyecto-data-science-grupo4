from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def calcular_serie_temporal(carpeta: str = "../data/temporal") -> pd.DataFrame:
    carpeta = Path(carpeta)
    archivos = {t: carpeta / f"REG02_EPHC_2024_T{t}.csv" for t in range(1, 5)}
    faltantes = [str(p) for p in archivos.values() if not p.exists()]
    if faltantes:
        raise FileNotFoundError("Faltan archivos trimestrales: " + ", ".join(faltantes))

    partes = []
    for trimestre, ruta in archivos.items():
        df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", low_memory=False)
        factor = "Factor" if "Factor" in df.columns else "factor" if "factor" in df.columns else "FACTOR"
        tmp = df[["AREA", factor]].copy()
        tmp["trimestre"] = trimestre
        tmp["factor_expansion"] = pd.to_numeric(tmp[factor], errors="coerce")
        tmp["area"] = tmp["AREA"].map({1: "Urbana", 6: "Rural"})
        partes.append(tmp[["trimestre", "area", "factor_expansion"]])

    df = pd.concat(partes, ignore_index=True)
    serie = df.groupby(["trimestre", "area"], as_index=False)["factor_expansion"].sum()
    serie["porcentaje"] = (
        serie["factor_expansion"]
        / serie.groupby("trimestre")["factor_expansion"].transform("sum")
        * 100
    )
    return serie[["trimestre", "area", "porcentaje"]]


def graficar_serie(serie: pd.DataFrame, salida: str = "../output/graficos/10_serie_temporal_area.png") -> None:
    plt.figure(figsize=(9, 5))
    for area, g in serie.groupby("area"):
        plt.plot(g["trimestre"], g["porcentaje"], marker="o", label=area)
    plt.xticks([1, 2, 3, 4], ["T1", "T2", "T3", "T4"])
    plt.xlabel("Trimestre 2024")
    plt.ylabel("Población estimada (%)")
    plt.title("Composición ponderada urbana/rural por trimestre - EPHC 2024")
    plt.legend(title="Área")
    plt.tight_layout()
    plt.savefig(salida, dpi=180)
    plt.close()


if __name__ == "__main__":
    serie = calcular_serie_temporal()
    print(serie.round(2))
    graficar_serie(serie)
