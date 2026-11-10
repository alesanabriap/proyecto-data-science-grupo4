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

La semilla aleatoria está fijada en `src/fase2/datos.py`, por lo que los resultados son idénticos en cada ejecución.

`notebooks/fase2.ipynb` es un archivo generado: los cambios se hacen en los notebooks de cada parte y luego se vuelve a unir.

## Fase 3 oficial — cobertura médica y perfiles socioeconómicos (EPHC 2024)

**Enfoque aprobado:** la Fase 3 busca identificar perfiles de adultos con y sin cobertura de seguro médico y evaluar modelos predictivos de la **ausencia de cobertura**. El trabajo anterior sobre enfermedad reciente se conserva en las Fases 1 y 2 como diagnóstico que motivó la corrección de la pregunta; no constituye la Fase 3 vigente.

- [Metodología y reproducción de Fase 3](docs/FASE3_METODOLOGIA.md)
- [Resultados reales y conclusión oficial](docs/FASE3_RESULTADOS.md)
- [Notebook oficial](notebooks/fase3.ipynb)
- [Pipeline de ejecución automática](.github/workflows/fase3.yml)

```bash
pip install -r requirements.txt
pytest -q tests/test_fase3.py
python -m src.fase3.ejecutar
python -m src.fase3.complementos   # grilla, sobreajuste, importancia y peores casos
jupyter nbconvert --to notebook --execute --inplace notebooks/fase3.ipynb
```

Los resultados reproducibles se generan en `output/fase3/` y GitHub Actions los publica como archivo descargable denominado `fase3-cobertura-2024`, junto con los modelos y notebook HTML.

**Resultado:** en T4 el HistGradientBoosting sin clusters alcanzó AUC-ROC 0,793 y accuracy 75,4 %, frente a 67,9 % de baseline. Los tres clusters describen desigualdades de cobertura pero no mejoraron la capacidad predictiva.
