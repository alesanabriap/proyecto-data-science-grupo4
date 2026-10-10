"""Parte 1. Carga del dataset analítico y utilidades compartidas de la Fase 2.

Todas las partes importan este módulo para trabajar exactamente sobre los mismos
datos, con el mismo nivel de significancia y la misma semilla aleatoria.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Rutas (relativas a la raíz del repositorio, sin rutas absolutas)
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "data" / "processed" / "ephc_salud_2024_limpio.csv"
DIR_TEMPORAL = RAIZ / "data" / "temporal"
CLAVE_PERSONA = ["UPM", "NVIVI", "NHOGA", "L02"]
DIR_SALIDA = RAIZ / "output" / "fase2"
DIR_GRAFICOS = DIR_SALIDA / "graficos"

# --------------------------------------------------------------------------- #
# Decisiones declaradas ANTES de ejecutar cualquier prueba
# --------------------------------------------------------------------------- #
ALFA = 0.05               # nivel de significancia de toda la fase
NIVEL_CONFIANZA = 0.95    # intervalos de confianza
SEMILLA = 42              # reproducibilidad de submuestras y bootstrap
N_BOOTSTRAP = 2000        # réplicas bootstrap
N_CONTRASTES = 6          # contrastes confirmatorios (familia para Bonferroni)

REGISTROS_ESPERADOS = 57_109

# Grupos de edad: se usan porque la relación edad-enfermedad no es lineal
# (ver Parte 2). Los cortes siguen etapas del ciclo de vida.
CORTES_EDAD = [-1, 4, 14, 29, 44, 59, 130]
ETIQUETAS_EDAD = ["0-4", "5-14", "15-29", "30-44", "45-59", "60+"]

# H4 se evalúa en personas que ya completaron su trayectoria educativa.
EDAD_MINIMA_EDUCACION = 25

# Colores (paleta categórica validada para daltonismo; el color nunca es el
# único canal: todos los gráficos llevan etiquetas o leyenda).
AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS = "#52514e"
GRIS_CLARO = "#d9d8d4"


def cargar_trimestres() -> pd.DataFrame:
    """Trimestre de entrevista de cada persona, tomado de los REG02 trimestrales.

    Los cuatro archivos trimestrales contienen exactamente las mismas personas que
    la base anual (la base anual es la unión de los cuatro trimestres). No traen las
    variables de salud, pero sí la clave de persona, de modo que basta enlazarlos
    para saber en qué trimestre se entrevistó a cada una.
    """
    partes = []
    for trimestre in range(1, 5):
        ruta = DIR_TEMPORAL / f"REG02_EPHC_2024_T{trimestre}.csv"
        parte = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", usecols=CLAVE_PERSONA)
        parte["trimestre"] = trimestre
        partes.append(parte)
    trimestres = pd.concat(partes, ignore_index=True)
    if trimestres.duplicated(CLAVE_PERSONA).any():
        raise ValueError("Hay personas repetidas entre los archivos trimestrales.")
    return trimestres


def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el dataset limpio de la Fase 1 y agrega las variables derivadas de la Fase 2.

    El archivo de la Fase 1 no se modifica: las variables nuevas viven solo en
    memoria y se reconstruyen por código en cada ejecución.
    """
    df = pd.read_csv(ruta)

    # Controles de integridad: si algo cambió en la Fase 1, se detecta aquí.
    if len(df) != REGISTROS_ESPERADOS:
        raise ValueError(f"Se esperaban {REGISTROS_ESPERADOS} registros y hay {len(df)}.")
    if not df["persona_id"].is_unique:
        raise ValueError("La clave persona_id no es única.")
    if not set(df["enfermedad_reciente"].unique()) <= {0, 1}:
        raise ValueError("enfermedad_reciente debe ser binaria (0/1).")

    df["cobertura"] = df["tiene_seguro"].map({0: "Sin seguro", 1: "Con seguro"})
    df["enfermedad"] = df["enfermedad_reciente"].map({0: "Sano/a", 1: "Enfermo/a"})
    df["grupo_edad"] = pd.cut(df["edad"], bins=CORTES_EDAD, labels=ETIQUETAS_EDAD)
    # Identificador de hogar: las personas de un mismo hogar no son independientes.
    df["hogar_id"] = (
        df["UPM"].astype(str) + "-" + df["NVIVI"].astype(str) + "-" + df["NHOGA"].astype(str)
    )

    # Trimestre de entrevista (corrección a la Fase 1): enlace uno a uno por la clave de persona.
    df = df.merge(cargar_trimestres(), on=CLAVE_PERSONA, how="left", validate="one_to_one")
    if df["trimestre"].isna().any():
        raise ValueError("Hay personas sin trimestre de entrevista.")
    df["trimestre"] = df["trimestre"].astype(int)
    return df


def aplicar_estilo() -> None:
    """Estilo común de los gráficos: ejes discretos, grilla tenue, texto legible."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRIS,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIS_CLARO,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def ejes_con_coma(eje, decimales_x: int | None = None, decimales_y: int | None = None) -> None:
    """Muestra los números de los ejes con coma decimal, como en el resto del informe."""
    from matplotlib.ticker import FuncFormatter

    def formateador(decimales):
        return FuncFormatter(lambda valor, _: f"{valor:.{decimales}f}".replace(".", ",").replace("-", "−"))

    if decimales_x is not None:
        eje.xaxis.set_major_formatter(formateador(decimales_x))
    if decimales_y is not None:
        eje.yaxis.set_major_formatter(formateador(decimales_y))


def guardar_figura(fig: plt.Figure, nombre: str) -> Path:
    """Guarda la figura en output/fase2/graficos y devuelve la ruta."""
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_GRAFICOS / nombre
    fig.savefig(ruta)
    return ruta


def ruta_salida(nombre: str) -> Path:
    """Ruta de un archivo de resultados en output/fase2 (crea la carpeta si no existe)."""
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    return DIR_SALIDA / nombre


def formatear_p(p: float) -> str:
    """Valor p legible: nunca se informa 'p = 0'."""
    if pd.isna(p):
        return "n/c"
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


class RegistroContrastes:
    """Acumula los resultados de los contrastes de una parte y los guarda en CSV.

    Cada parte escribe ``output/fase2/contrastes_parteN.csv``. La Parte 3 lee los
    tres archivos para aplicar la corrección por comparaciones múltiples y armar
    la tabla resumen, de modo que ningún número se copia a mano.
    """

    COLUMNAS = [
        "id", "parte", "hipotesis", "variables", "familia", "prueba", "supuestos",
        "n", "estadistico", "valor_estadistico", "gl",
        "p_clasico", "p_robusto",
        "efecto", "valor_efecto", "magnitud",
        "ic_parametro", "ic_inf", "ic_sup", "ic_metodo",
    ]

    def __init__(self, parte: int):
        self.parte = parte
        self.filas: list[dict] = []

    def agregar(self, **campos) -> None:
        desconocidos = set(campos) - set(self.COLUMNAS)
        if desconocidos:
            raise KeyError(f"Campos no previstos en el registro: {sorted(desconocidos)}")
        fila = {columna: campos.get(columna, np.nan) for columna in self.COLUMNAS}
        fila["parte"] = self.parte
        # Si la celda se vuelve a ejecutar, se reemplaza el contraste en vez de duplicarlo.
        self.filas = [f for f in self.filas if f["id"] != fila["id"]]
        self.filas.append(fila)

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas, columns=self.COLUMNAS)

    def guardar(self) -> Path:
        DIR_SALIDA.mkdir(parents=True, exist_ok=True)
        ruta = DIR_SALIDA / f"contrastes_parte{self.parte}.csv"
        self.tabla().to_csv(ruta, index=False)
        return ruta
