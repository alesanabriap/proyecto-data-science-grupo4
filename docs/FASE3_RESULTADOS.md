# Resultados validados — Fase 3 oficial: cobertura médica (EPHC 2024)

**Ejecución oficial:** [GitHub Actions — Fase 3 cobertura, ejecución satisfactoria](https://github.com/alesanabriap/proyecto-data-science-grupo4/actions/runs/42886654996). Los CSV, gráficos y modelos `.joblib` están disponibles como artifact `fase3-cobertura-2024`. Se trabaja sobre la base original, no sobre la base que excluía accidentes.

## Nueva pregunta propuesta
**¿Qué perfiles sociodemográficos y socioeconómicos se asocian a la falta de cobertura médica entre adultos paraguayos, y en qué medida esas características permiten clasificar dicha condición?**

## Muestra
- EPHC REG02 2024 original: **57.744** registros.
- Adultos válidos 18+: **40.266**.
- Desarrollo T1–T3: **28.816** personas; test T4: **11.450**.
- Falta de seguro en adultos (ponderada): **68,44 %**.
- Cobertura urbana: **36,90 %**; rural: **17,40 %**.
- Análisis de quintiles muestrales de ingreso per cápita: cobertura ~**5,93 %** (Q1) y ~**58,28 %** (Q5). Son quintiles calculados de la muestra, no necesariamente los quintiles poblacionales oficiales.

## Clustering
- MiniBatchKMeans sin etiqueta seguro, k en [3,4,5,6].
- Selección: **k = 3**, silhouette **0,224** (separación modesta).
- Estabilidad promedio ARI entre inicializaciones: **0,506** (inestable en algunas).
- Perfiles T4:
  - Perfil 0 (n=4089): sin seguro 49,20 %, edad media 38,9 años, ingreso mediano 2,97 millones Gs.
  - Perfil 1 (n=4274): sin seguro 79,21 %, edad media 58,4 años, ingreso mediano 1,15 millones Gs.
  - Perfil 2 (n=3087): sin seguro 85,04 %, edad media 28,1 años, ingreso mediano 1,02 millones Gs.
- Las diferencias descriptivas entre perfiles no prueban efectos causales ni que los clusters sean categorías naturales estables.

## Clasificación

Se entrenaron DummyClassifier, regresión logística, Random Forest e HistGradientBoosting, con y sin clusters como atributos. **Selección por AUC-ROC de validación temporal** T1→T2 y T1+T2→T3, sin utilizar T4. Los clusters se ajustan dentro de los datos de cada partición, sin utilizar la etiqueta de cobertura. Métricas con `factor_expansion` de la EPHC.

| Modelo (sin clusters) | AUC validación | AUC prueba T4 |
|---|---:|---:|
| Baseline | 0,500 | 0,500 |
| Regresión logística | 0,798 | 0,784 |
| Random Forest | 0,798 | 0,789 |
| **HistGradientBoosting** | **0,801** | **0,793** |

Modelo seleccionado **sin clustering**. En T4:
- Accuracy: **75,41 %** (baseline **67,95 %**).
- Balanced accuracy: **0,687**.
- Precision para ausencia de seguro: **78,77 %**.
- Recall para ausencia de seguro: **87,36 %**.
- F1: **0,828**; Brier: **0,167**.

El mejor AUC de validación **con clusters** fue **0,800494**, frente a **0,800670** **sin clusters**: cambio **−0,000175**. Por tanto, **clustering no demostró una mejora relevante de predicción**. Se conserva para describir los perfiles, no como parte indispensable de la tarea supervisada.

## NBI territorial
Cruce departamental Censo 2022 (hogares) con EPHC 2024 (personas adultas); **16** territorios con información común.

| Dimensión NBI | Spearman ρ | p exploratorio sin corrección |
|---|---:|---:|
| Al menos una NBI | 0,390 | 0,135 |
| Calidad de vivienda | 0,450 | 0,081 |
| Infraestructura sanitaria | 0,190 | 0,481 |
| Acceso a educación | 0,524 | 0,037 |
| Capacidad de subsistencia | 0,529 | 0,035 |

Aunque dos p-values nominales son menores a 0,05, **ninguno se mantiene bajo corrección de Benjamini–Hochberg para las cinco comparaciones** (q mínimo ≈ 0,094). Se trata de resultados exploratorios ecológicos, no evidencia causal ni individual.

## Conclusión y decisión

La reformulación hacia **falta de cobertura médica** ofrece un problema más directamente observable con las variables de la EPHC y mejora sustancialmente la discriminación estadística respecto del proyecto de enfermedad reciente (AUC anterior ≈ 0,624, aunque no es una comparación experimental entre el mismo objetivo/población). El modelo nuevo obtiene AUC ≈ 0,793 en T4 y mejora claramente la accuracy frente a baseline. La desigualdad socioeconómica se visualiza con diferencias amplias de cobertura por ingresos y residencia.

El clustering permite caracterizar grupos con fuerte contraste de cobertura, pero **sus agrupaciones son moderadamente separadas y presentan estabilidad irregular**, y **no mejoran el modelo supervisado**. NBI aporta contexto, pero sus correlaciones no se sostienen tras ajustes para múltiples comparaciones.

**Enfoque aprobado:** el grupo eligió cobertura médica como eje principal de la Fase 3, preservando clustering como análisis complementario y el modelo supervisado sin clusters como ganador. No fusionar el PR con `main` antes de la aprobación del grupo.

**Advertencias:** las métricas ponderadas no sustituyen una inferencia completa del diseño muestral; no se infiere relación causal ni riesgo médico. No se afirma que el algoritmo de clustering haya mejorado el desempeño.
