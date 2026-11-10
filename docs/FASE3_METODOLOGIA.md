# Fase 3 — perfiles sociodemográficos y clasificación de enfermedad reciente

## Pregunta reformulada (por devolución docente)

**¿Qué perfiles sociodemográficos y socioeconómicos presentan mayor prevalencia de enfermedad reciente en Paraguay y en qué medida sus características permiten clasificar esa condición según la EPHC 2024?**

Se mantiene el conjunto EPHC 2024 de las fases previas; el seguro pasa a ser predictor en lugar del eje exclusivo. El NBI del Censo 2022 amplía el contexto territorial.

La Fase 3 **exige clasificación supervisada** y comparaciones de modelos. El clustering solo descubre perfiles y **no sustituye** el entrenamiento de los tres clasificadores. Las Fases 1 y 2 siguen intactas.

## Datos y variables

**Objetivo:** \`enfermedad_reciente\` (1: declaró enfermedad en últimos 90 días, 0: declaró estar sano). No representa incidencia, diagnóstico clínico ni probabilidad futura de enfermar.

**Para formar clusters:** \`edad\`, \`quintil_ingreso\`, \`anios_estudio\`, \`sexo\`, \`area\`, \`tiene_seguro\`. No interviene \`enfermedad_reciente\` en su creación.

**Para clasificar:** las variables anteriores + \`estado_civil\`, \`departamento\`. Ingreso y quintil son sustitutos próximos: se elige quintil para no duplicar señal. La educación tiene faltantes en menores; se utiliza imputación dentro del pipeline. No entran \`persona_id\`, \`UPM\`, \`trimestre\` ni \`factor_expansion\` como atributos.

**Pesos:** \`factor_expansion\` para tasas y métricas poblacionales aproximadas, no como predictor.

## Métodos

- Agrupación: MiniBatchKMeans (k = 3, 4, 5, 6); selección por coeficiente silhouette sobre una submuestra reproducible del periodo de desarrollo. Su interpretación y estabilidad deben discutirse posteriormente.
- Clasificación: DummyClassifier (mayoría, baseline), regresión logística, Random Forest y Gaussian Naive Bayes. Dos configuraciones por algoritmo, predefinidas.
- Preprocesamiento: imputación de medianas, escalado y codificación one-hot, ajustados solo en entrenamiento. Separación temporal sin mezclar entrevistas posteriores.
- Validación de origen expansivo: T1 → T2 y T1+T2 → T3. Elección por AUC-ROC ponderada media en validación.
- Prueba final: modelo reentrenado con T1–T3, evaluado **una sola vez** en T4. Se comprueba ausencia de UPM repetidas entre desarrollo y prueba.
- Métricas ponderadas: accuracy, balanced accuracy, precision, recall, F1, AUC-ROC y matriz de confusión. Comparación entrenamiento/prueba y métricas por subgrupos.
- NBI: 18 registros oficiales (Asunción y 17 departamentos) correspondientes a hogares en 2022. Cruce agregado por departamentos disponibles en EPHC 2024. Spearman exploratoria para cinco dimensiones; **no** asignar NBI departamental a cada hogar o persona. El resultado no demuestra causalidad.

## Fuente y trazabilidad del NBI

INE, *Necesidades Básicas Insatisfechas (NBI) Paraguay 2022*, tríptico, tabla “Hogares por Necesidades Básicas Insatisfechas según departamento y distrito”.

https://www.ine.gov.py/Publicaciones/Biblioteca/documento/262/NIB%20-%20TRIPTICO.pdf

La transcripción departamental está en \`data/externos/nbi_departamental_2022.csv\`. La EPHC no cubre Boquerón ni Alto Paraguay en este dataset: esas filas se conservan en la tabla externa, pero no se usan para atribuir prevalencias de enfermedad sin encuesta.

## Reproducción

Desde la raíz del repositorio, usando Python 3.10+:

\`\`\`bash
python -m pip install -r requirements.txt
pytest -q tests/test_fase3.py
python -m src.fase3.ejecutar
jupyter nbconvert --to notebook --execute --inplace notebooks/fase3.ipynb
jupyter nbconvert --to html notebooks/fase3.ipynb --output-dir output/fase3 --output fase3.html
\`\`\`

**Requisito:** deben existir \`data/processed/ephc_salud_2024_limpio.csv\` y \`data/temporal/REG02_EPHC_2024_T1.csv\` a \`T4.csv\` (presentes en el repositorio). No se modifican los archivos fuente.

La ejecución produce archivos en \`output/fase3/\`:

| Archivo | Contenido |
|---|---|
| \`validacion_temporal.csv\` | Selección de hiperparámetros en T2 y T3 |
| \`metricas_clasificacion.csv\` | Baseline y 3 algoritmos en train y test |
| \`matriz_confusion.png\` | Errores en T4 |
| \`errores_subgrupos.csv\` | Errores desagregados |
| \`silhouette_k.csv\`, \`resumen_perfiles.csv\` | Validación y perfiles |
| \`enfermedad_por_perfil.png\` | Comparación exploratoria |
| \`cruce_departamental_nbi.csv\`, \`correlaciones_nbi.csv\` | Cruce departamental |
| \`nbi_vs_enfermedad_departamento.png\` | Dispersión territorial |
| \`clasificador.joblib\`, \`segmentador.joblib\` | Modelos serializados |
| \`resumen_ejecucion.json\` | Síntesis machine-readable |
| \`fase3.html\` | Exportación del notebook, si se ejecuta |

Inferencia con el modelo serializado:

\`\`\`bash
python -m src.fase3.predecir --entrada nuevos.csv --salida predicciones.csv
\`\`\`

## Limitaciones

- Correlaciones y diferencias entre perfiles son **asociaciones**, no efectos causales. Es posible que el modelo no supere suficientemente la línea base: reportarlo sin ocultarlo.
- Los datos de EPHC representan entrevistas repetidas por hogares y UPM; las métricas ponderadas no sustituyen una inferencia completa del diseño multietápico.
- En T4 cambia la composición de la muestra. Con un único año no existe validación de generalización interanual.
- La integración NBI usa 2022 (hogares), mientras la enfermedad es 2024 (personas); inferencia ecológica limitada a los departamentos cubiertos.
- Los algoritmos no deben usarse para decisiones clínicas ni para restringir servicios de salud o seguros.
- Para entrega final se debe redactar el informe PDF, revisar resultados reales y completar la PPT con las métricas de T4. Una prueba con datos sintéticos **no equivale** a validar conclusiones sobre Paraguay.
