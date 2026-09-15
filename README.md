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
- `notebooks/`: notebook ejecutable de Fase 1.
- `src/`: utilidad reproducible para la serie temporal.
- `output/graficos/`: figuras del EDA.
- `docs/`: informe final y checklist.

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

## Nota sobre la serie temporal
Los REG02 trimestrales suministrados no contienen las variables de salud `S01A` y `S03`. Por ello no se inventa una serie trimestral de enfermedad o cobertura. La visualización temporal utiliza la composición ponderada urbana/rural (`AREA`) como variable contextual común a los cuatro trimestres y relevante para el análisis principal.

## Decisiones metodológicas principales
- `S03=1` se considera enfermedad reciente.
- `S03=3` se considera sano/a.
- `S03=2` (accidente) se excluye del dataset analítico principal.
- `S03=9` se excluye por código no válido.
- `tiene_seguro=1` para `S01A` 1-6 u 8; `0` para `S01A=7`.
- Los faltantes de educación, ingreso y quintil no se imputan en Fase 1.
- El factor de expansión se usa cuando el resultado pretende representar población.
- No se realizan afirmaciones causales.
