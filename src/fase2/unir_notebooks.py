"""Parte 1. Une los tres notebooks de la Fase 2 en un único notebook de entrega.

Cada integrante trabaja en su propio archivo (``notebooks/fase2_parteN_*.ipynb``)
para no pisar los cambios de los demás. La consigna pide un notebook ejecutado
y su exportación a HTML, así que este script concatena las tres partes, en
orden, en ``notebooks/fase2.ipynb``. Ese archivo no se edita a mano: se regenera.

Uso, desde la raíz del repositorio::

    python -m src.fase2.unir_notebooks
"""
from __future__ import annotations

from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[2]
DIR_NOTEBOOKS = RAIZ / "notebooks"
PARTES = [
    "fase2_parte1_proporciones.ipynb",
    "fase2_parte2_numericas.ipynb",
    "fase2_parte3_asociacion_regresion.ipynb",
]
SALIDA = DIR_NOTEBOOKS / "fase2.ipynb"

PORTADA = """# Proyecto Integrador de Ciencia de Datos · Fase 2: análisis inferencial
## Asociación entre cobertura de seguro médico y enfermedad reciente en Paraguay: análisis de la EPHC 2024

**Grupo 4**
Kyara Ivanna Popov Araujo · Zinri Alice Bobadilla Peralta · Alejandro Sanabria Penayo

Este notebook reúne, en orden, las tres partes de la Fase 2. Se genera con
`python -m src.fase2.unir_notebooks` a partir de los notebooks de cada integrante
y se ejecuta de principio a fin.

| Sección | Contenido |
|---|---|
| Parte 1 | Correcciones de la Fase 1, marco inferencial, independencia de las observaciones, H1 y H5 |
| Parte 2 | H2 y H4 |
| Parte 3 | H3, H6, comparaciones múltiples, tabla resumen y conclusión |
"""


def unir() -> Path:
    unido = nbformat.v4.new_notebook()
    unido.cells.append(nbformat.v4.new_markdown_cell(PORTADA))
    for nombre in PARTES:
        parte = nbformat.read(DIR_NOTEBOOKS / nombre, as_version=4)
        if not unido.metadata:
            unido.metadata = parte.metadata
        for celda in parte.cells:
            if celda.cell_type == "code":      # el notebook unido se ejecuta de nuevo, sin salidas previas
                celda.outputs = []
                celda.execution_count = None
            unido.cells.append(celda)
    for i, celda in enumerate(unido.cells):    # identificadores propios y estables, sin duplicados entre partes
        celda["id"] = f"celda-{i:03d}"
    nbformat.validate(unido)
    nbformat.write(unido, SALIDA)
    return SALIDA


if __name__ == "__main__":
    ruta = unir()
    print(f"Notebook unificado: {ruta.relative_to(RAIZ)} ({len(nbformat.read(ruta, as_version=4).cells)} celdas)")
