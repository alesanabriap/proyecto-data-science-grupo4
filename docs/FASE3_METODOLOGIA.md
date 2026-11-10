# Fase 3 oficial: cobertura médica, perfiles socioeconómicos y NBI

**Objetivo:** contrastar si una reformulación de la pregunta de Fase 3 mejora la utilidad estadística y predictiva respecto de la clasificación de enfermedad reciente.

**Pregunta:** ¿Qué perfiles socioeconómicos y sociodemográficos se relacionan con la ausencia de seguro médico entre adultos de Paraguay durante 2024, y qué capacidad tienen los modelos de machine learning para clasificar esa condición?

**Fuente primaria:** EPHC 2024 REG02 anual original + trimestres originales. Se usa la población de 18+ años con indicador S01A válido (1-6: con seguro; 7: sin seguro) y factores de expansión positivos. Los casos excluidos por accidente/enfermedad en la investigación anterior vuelven a entrar porque no corresponden a la nueva pregunta.

**Predictores:** edad, años de estudio, ingreso per cápita mensual (log1p), sexo, área urbana/rural y departamento. No se usan quintil e ingreso simultáneamente. No se incluyen enfermedades, seguro médico ni identificadores de UPM como predictores.

**Clustering:** MiniBatchKMeans con k=3,4,5,6 y silhouette, sobre edad, años de estudio, ingreso, sexo y área (no incluye objetivo). Se calcula estabilidad por ARI de reinicializaciones. Se comparan modelos con y sin perfil para verificar si agrega señal predictiva.

**Clasificación:** regresión logística, Random Forest y HistGradientBoosting frente a línea base DummyClassifier. Entrenamiento ponderado. Validación origen expansivo T1→T2 y T1-T2→T3; T4 aislado para evaluación. Métricas: AUC, F1, precisión, exhaustividad, accuracy, balanced accuracy y Brier; errores con umbral 0,5.

**NBI:** 18 registros de la tabla departamental INE Censo 2022. Cruce por los 16 territorios con cobertura EPHC 2024, correlación ecológica Spearman. No se atribuyen privaciones departamentales a personas ni se afirma causalidad.

**Ejecución:**

```bash
pip install -r requirements.txt
python -m src.fase3.ejecutar
```

**Productos:** `output/fase3/resultados.json`, `comparacion_test.csv`, `ranking_validacion.csv`, `resumen_perfiles.csv`, `brechas_descriptivas.csv`, `correlaciones_nbi.csv`, gráficos y modelos joblib. GitHub Actions publica los resultados como artifact.

**Estado:** fase oficial, ejecutada sobre la base real. Los resultados están en `docs/FASE3_RESULTADOS.md`. Los análisis complementarios (búsqueda en grilla, brecha de sobreajuste, importancia de variables y casos peor predichos) se generan con `python -m src.fase3.complementos` en `output/fase3/complementos/`.


## Corrección de la pregunta tras devolución de la cátedra
Las fases 1 y 2 documentaron que la asociación ajustada entre seguro y enfermedad reciente es insuficiente para la pregunta original. Por decisión del grupo y aprobación del enfoque reformulado, la variable objetivo de la Fase 3 cambia a `sin_seguro`; **se mantienen las fases anteriores como antecedentes, no se reescribe su evidencia**. Se usó la base REG02 original sin excluir accidentados para el nuevo objetivo.

## Entregables oficiales
- Notebook: `notebooks/fase3.ipynb`.
- Scripts: `src/fase3/ejecutar.py`.
- Evidencia: `docs/FASE3_RESULTADOS.md`.
- GitHub Actions genera el CSV, métricas, PNG, modelos `.joblib` y notebook HTML.
