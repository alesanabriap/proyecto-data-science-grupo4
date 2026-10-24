"""Genera entregas autónomas de Fase 1 y Fase 2 desde el repositorio."""
from pathlib import Path
import shutil, zipfile, subprocess, json
import fitz
import nbformat

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)
NAMES = "Popov-Bobadilla-Sanabria"
COMMON = [
    "README.md", "requirements.txt", "data/raw/diccionario_EPHC_ANUAL_2024.xls",
    "data/processed/ephc_salud_2024_limpio.csv",
    "data/processed/diccionario_datos_final.csv",
    "data/processed/diccionario_datos_seleccionados.csv",
    "data/processed/bitacora_limpieza.csv",
]
TEMP = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "data/temporal").glob("*")) if p.is_file()]
SOURCE = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").rglob("*.py"))]
F1 = COMMON + TEMP + SOURCE + [
    "docs/Comprension_del_Problema.pdf", "docs/Informe_Fase1_Grupo4.pdf",
    "notebooks/fase1.ipynb", "docs/fase1_FINAL.html",
] + [str(p.relative_to(ROOT)) for p in sorted((ROOT/"output/graficos").glob("*")) if p.is_file()]
F2 = COMMON + TEMP + SOURCE + [
    "docs/Informe_Fase1_Grupo4.pdf", # respaldo de correcciones
    "docs/Informe_Fase2_Grupo4.pdf", "notebooks/fase2.ipynb",
    "notebooks/fase2_parte1_proporciones.ipynb",
    "notebooks/fase2_parte2_numericas.ipynb",
    "notebooks/fase2_parte3_asociacion_regresion.ipynb",
    "docs/fase2.html",
] + [str(p.relative_to(ROOT)) for p in sorted((ROOT/"output/fase2").rglob("*")) if p.is_file()]
# Regla de la guía: originales >25 MB no se incluyen en ZIP.
# Conservamos referencias para recuperar el original exacto del repositorio.
ORIGINAL = "data/raw/REG02_EPHC_ANUAL_2024.csv"
report = []
for phase, files, pages in [
    (1, F1, {"docs/Comprension_del_Problema.pdf": (2,3), "docs/Informe_Fase1_Grupo4.pdf": (1,20)}),
    (2, F2, {"docs/Informe_Fase2_Grupo4.pdf": (1,15)}),
]:
    unique = sorted(set(files))
    missing = [x for x in unique if not (ROOT / x).is_file()]
    if missing:
        raise FileNotFoundError(f"Fase {phase}: faltan archivos: {missing}")
    warnings = []
    for rel, (lo, hi) in pages.items():
        try:
            with fitz.open(ROOT / rel) as pdf:
                n = len(pdf)
                if not lo <= n <= hi:
                    warnings.append(f"{rel}: {n} páginas (requerido {lo}-{hi})")
        except Exception as e:
            warnings.append(f"PDF no validable {rel}: {e}")
    notebook = ROOT / f"notebooks/fase{phase}.ipynb"
    n = nbformat.read(notebook, as_version=4)
    errors = [(i, out.get("ename")) for i,c in enumerate(n.cells) if c.cell_type == "code"
              for out in c.get("outputs", []) if out.output_type == "error"]
    unexecuted = [i for i,c in enumerate(n.cells) if c.cell_type == "code" and c.execution_count is None]
    if errors: warnings.append(f"Errores guardados en notebook: {errors}")
    if unexecuted: warnings.append(f"Celdas con execution_count vacío: {unexecuted}")
    bundle_name = f"Fase{phase}_DataScience_{NAMES}.zip"
    with zipfile.ZipFile(DIST / bundle_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in unique:
            z.write(ROOT / rel, rel)
        intro = (
           f"ENTREGA FASE {phase} - GRUPO 4\n"
           "Integrantes: Kyara Ivanna Popov Araujo, Zinri Alice Bobadilla Peralta, Alejandro Sanabria Penayo\n"
           "Repositorio: https://github.com/alesanabriap/proyecto-data-science-grupo4\n"
           f"Archivo original omitido por superar 25 MB: {ORIGINAL}\n"
           "Descargar el CSV original del repositorio a esa ubicación, o desde el INE siguiendo README.md.\n"
           "Los notebooks ya contienen resultados guardados en el repositorio.\n"
           "Instalar dependencias con: pip install -r requirements.txt\n"
           "Ejecutar desde la raíz del ZIP con Jupyter, y guardar las salidas regeneradas.\n"
           "Para regenerar el notebook de Fase 2 desde sus partes, seguir README.md.\n"
           + ("\nADVERTENCIAS DE VALIDACIÓN:\n" + "\n".join(warnings) if warnings else "\nSin advertencias de estructura.\n")
        )
        z.writestr("INSTRUCCIONES_ENTREGA.txt", intro)
    with zipfile.ZipFile(DIST/bundle_name) as z:
        bad = z.testzip()
        if bad: raise ValueError(f"ZIP dañado: {bad}")
    report.append({"fase": phase, "archivo": bundle_name, "archivos": len(unique)+1, "advertencias": warnings})
(DIST / "validacion_entregas.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
