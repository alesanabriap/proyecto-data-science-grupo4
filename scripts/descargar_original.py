"""Descarga el CSV original si falta (archivo omitido del ZIP por >25 MB)."""
from pathlib import Path
from urllib.request import urlopen
from shutil import copyfileobj
DEST = Path(__file__).resolve().parents[1] / "data/raw/REG02_EPHC_ANUAL_2024.csv"
URL = "https://raw.githubusercontent.com/alesanabriap/proyecto-data-science-grupo4/main/data/raw/REG02_EPHC_ANUAL_2024.csv"
if __name__ == "__main__":
    if DEST.exists() and DEST.stat().st_size > 25_000_000:
        print("El CSV original ya está disponible:", DEST)
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=180) as src, DEST.open("wb") as dst:
            copyfileobj(src, dst)
        print("Descargado:", DEST, DEST.stat().st_size, "bytes")
