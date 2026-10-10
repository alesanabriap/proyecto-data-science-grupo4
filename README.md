# Proyecto Integrador de Ciencia de Datos - Grupo 4

**Título:** Asociación entre cobertura de seguro médico y enfermedad reciente en Paraguay: análisis de la EPHC 2024.

## Integrantes
- Kyara Ivanna Popov Araujo
- Zinri Alice Bobadilla Peralta
- Alejandro Sanabria Penayo

## Fuente
Instituto Nacional de Estadística (INE Paraguay), Encuesta Permanente de Hogares Continua (EPHC) 2024.

Portal oficial: https://www.ine.gov.py/microdato/

La versión anual incluida en este repositorio corresponde al archivo oficial suministrado para la entrega y contiene la columna `FACTOR` (Factor Revisión 2025). En el procesamiento se normaliza a `factor_expansion`.

## Estructura
- `data/raw/`: REG02 anual original y diccionario oficial.
- `data/temporal/`: cuatro REG02 trimestrales oficiales de 2024.
- `data/processed/`: dataset analítico, bitácora y diccionarios.
- `notebooks/`: notebook ejecutable de Fase 1 y notebooks de Fase 2 (uno por integrante más el unificado).
- `src/`: utilidad reproducible para la serie temporal (`analisis_temporal.py`) y paquete `fase2/` con el código compartido de la inferencia.
- `output/graficos/`: figuras del EDA (Fase 1).
- `output/fase2/`: resultados de los contrastes, tabla resumen y figuras de la Fase 2.
- `docs/`: informes en PDF y notebooks exportados a HTML.

## Reproducción
1. Crear un entorno virtual.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Verificar que existan:
   - `data/raw/REG02_EPHC_ANUAL_2024.csv`
   - `data/temporal/REG02_EPHC_2024_T1.csv`
   - `data/temporal/REG02_EPHC_2024_T2.csv`
   - `data/temporal/REG02_EPHC_2024_T3.csv`
   - `data/temporal/REG02_EPHC_2024_T4.csv`
4. Ejecutar `notebooks/fase1.ipynb` de principio a fin.
5. Opcionalmente ejecutar `python src/analisis_temporal.py` desde `src/` para regenerar la figura temporal.

## Fase 2: análisis inferencial

### Organización del trabajo
La Fase 2 contrasta seis hipótesis. El código está dividido en tres partes, una por integrante, con archivos separados para que cada uno trabaje y haga sus commits sin conflictos.

| Parte | Responsable | Notebook | Módulos de `src/fase2/` | Contenido |
|---|---|---|---|---|
| 1 | Kyara Ivanna Popov Araujo | `fase2_parte1_proporciones.ipynb` | `datos.py`, `remuestreo.py`, `unir_notebooks.py` | Correcciones de la Fase 1, marco inferencial, independencia de las observaciones, H1 (seguro) y H5 (área) |
| 2 | Zinri Alice Bobadilla Peralta | `fase2_parte2_numericas.ipynb` | `supuestos.py`, `efectos.py` | H2 (edad) y H4 (educación e ingreso) |
| 3 | Alejandro Sanabria Penayo | `fase2_parte3_asociacion_regresion.ipynb` | `resumen.py` | H3 (quintil), H6 (seguro controlando otras variables), comparaciones múltiples, tabla resumen y conclusión |

### Reproducción de la Fase 2
Requiere el dataset `data/processed/ephc_salud_2024_limpio.csv` que genera el notebook de la Fase 1 y los cuatro archivos de `data/temporal/`. Desde la raíz del repositorio:

```bash
pip install -r requirements.txt

# 1. Ejecutar las tres partes en orden (la Parte 3 lee los resultados de las otras dos)
jupyter nbconvert --to notebook --execute --inplace notebooks/fase2_parte1_proporciones.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/fase2_parte2_numericas.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/fase2_parte3_asociacion_regresion.ipynb

# 2. Unir las partes en el notebook de entrega, ejecutarlo y exportarlo a HTML
python -m src.fase2.unir_notebooks
jupyter nbconvert --to notebook --execute --inplace notebooks/fase2.ipynb
jupyter nbconvert --to html notebooks/fase2.ipynb --output-dir docs --output fase2.html
```

La ejecución completa tarda alrededor de dos minutos (el bootstrap usa 2.000 réplicas). La semilla aleatoria está fijada en `src/fase2/datos.py`, por lo que los resultados son idénticos en cada ejecución.

`notebooks/fase2.ipynb` es un archivo generado: los cambios se hacen en los notebooks de cada parte y luego se vuelve a unir.

El informe en PDF se compila desde su fuente en Markdown con pandoc y XeLaTeX, también desde la raíz del repositorio (las figuras se leen de `output/fase2/graficos/`):

```bash
pandoc docs/Informe_Fase2_Grupo4.md -o docs/Informe_Fase2_Grupo4.pdf --pdf-engine=xelatex
```

### Resultados de la Fase 2
- `output/fase2/tabla_resumen_contrastes.csv`: tabla resumen completa (también en `.md` y en versión legible).
- `output/fase2/contrastes_parteN.csv`: resultados que registra cada parte.
- `output/fase2/graficos/`: nueve figuras.
- `docs/Informe_Fase2_Grupo4.pdf`: informe de la fase. Su fuente es `docs/Informe_Fase2_Grupo4.md`.
- `docs/fase2.html`: notebook ejecutado.

### Decisiones metodológicas de la Fase 2
- Nivel de significancia: 0,05, declarado antes de ejecutar las pruebas.
- Las observaciones no son independientes (personas del mismo hogar y de la misma UPM). Cada contraste informa el valor p clásico y un valor p robusto a conglomerados; la decisión usa el robusto.
- Corrección por comparaciones múltiples: Bonferroni sobre los seis contrastes confirmatorios y Benjamini-Hochberg en las comparaciones post hoc.
- Las pruebas se aplican sobre la muestra sin ponderar; la estimación ponderada se usa como análisis de sensibilidad en H1.
- La hipótesis sobre educación e ingreso (H4) se evalúa en personas de 25 años o más.
- El trimestre de entrevista se recupera enlazando los archivos trimestrales y se usa como control de sensibilidad en H5 y H6.

## Nota sobre la serie temporal
Los REG02 trimestrales suministrados no contienen las variables de salud `S01A` y `S03`. Por ello no se inventa una serie trimestral de enfermedad o cobertura. La visualización temporal utiliza la composición ponderada urbana/rural (`AREA`) como variable contextual común a los cuatro trimestres y relevante para el análisis principal.

**Corrección aplicada en la Fase 2:** los cuatro REG02 trimestrales contienen exactamente las mismas 57.744 personas que la base anual. Aunque no traen `S01A` ni `S03`, al enlazarlos con la base anual por la clave `UPM + NVIVI + NHOGA + L02` se recupera el trimestre de entrevista de cada persona y, con él, la proporción de enfermedad por trimestre. El enlace se hace por código en `src/fase2/datos.py` (función `cargar_trimestres`).

## Decisiones metodológicas principales
- `S03=1` se considera enfermedad reciente.
- `S03=3` se considera sano/a.
- `S03=2` (accidente) se excluye del dataset analítico principal.
- `S03=9` se excluye por código no válido.
- `tiene_seguro=1` para `S01A` de 1 a 6; `0` para `S01A=7` (el código 8 no existe en la EPHC 2024).
- Los faltantes de educación, ingreso y quintil no se imputan en Fase 1.
- El factor de expansión se usa cuando el resultado pretende representar población.
- No se realizan afirmaciones causales.
