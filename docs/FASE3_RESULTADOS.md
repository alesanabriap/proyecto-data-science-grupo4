# Fase 3 — Resultados reales (EPHC 2024)

> **Estado:** ejecución verificada en GitHub Actions, [run exitoso #2](https://github.com/alesanabriap/proyecto-data-science-grupo4/actions/runs/42737981986). Los archivos CSV, PNG, notebook HTML y modelos serializados se incluyen en el artifact \`fase3-resultados-epch-2024\` de ese run. Aquí se consignan los valores procedentes de esa ejecución; **no son datos sintéticos**.

## Pregunta revisada

**¿Qué perfiles sociodemográficos y socioeconómicos presentan mayor prevalencia de enfermedad reciente en Paraguay y en qué medida sus características permiten clasificar esa condición según la EPHC 2024?**

La enfermedad es una **condición autorreportada en los últimos 90 días**, no una probabilidad futura de contagio ni un diagnóstico clínico.

## Base y partición temporal

- Total de registros analíticos: **57.109 personas**.
- Desarrollo T1–T3: **40.942 personas**.
- Prueba final T4: **16.167 personas**.
- Prevalencia ponderada T4: **27,53 %**.
- La clasificación no utiliza la variable objetivo ni identificadores como predictores; se audita ausencia de UPM presentes en ambas particiones.

## Resultado de clustering (exploratorio)

Se seleccionó **k = 6** por mejor coeficiente silhouette entre los valores k = 3, 4, 5 y 6 probados. La silhouette fue **0,199**, lo que sugiere separación débil y obliga a no exagerar la solidez de las agrupaciones.

| Perfil | Personas T4 | Edad media ponderada | Quintil medio | Enfermedad reciente ponderada |
|---|---:|---:|---:|---:|
| 0 | 3.802 | 27,9 años | 1,60 | 24,58 % |
| 1 | 4.575 | 33,9 años | 4,21 | 19,69 % |
| 2 | 1.367 | 59,4 años | 4,39 | 30,60 % |
| 3 | 1.044 | 2,1 años | 2,47 | 40,44 % |
| 4 | 2.835 | 10,2 años | 2,82 | 30,77 % |
| 5 | 2.544 | 62,0 años | 2,21 | 40,12 % |

**Lectura:** los perfiles 3 y 5 presentan prevalencias cercanas al 40 %, mientras el perfil 1 tiene alrededor de 20 %. La composición por edad explica parte de esta diferencia; no se hizo ajuste inferencial de prevalencia por cluster para concluir efectos independientes. No interpretar estos perfiles como grupos clínicos ni asignarles características personales.

## Comparación predictiva (T4)

Las métricas se calcularon con ponderación \`factor_expansion\`.

| Modelo | Accuracy | Recall (enfermo) | Precision | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Baseline (mayoría) | 72,47 % | 0,00 % | 0,00 % | 0,000 | 0,500 |
| Regresión logística | 72,50 % | 2,88 % | 50,88 % | 0,055 | 0,610 |
| Random Forest | **72,61 %** | **3,06 %** | 54,57 % | 0,058 | **0,624** |
| Gaussian Naive Bayes | 63,39 % | 34,14 % | 33,71 % | **0,339** | 0,582 |

Random Forest obtuvo el mejor AUC-ROC en validación (aprox. **0,602**) y fue seleccionado como ganador antes del conjunto T4.

**Conclusión clave:** Random Forest presenta un AUC-ROC superior al azar, pero el umbral de decisión 0,5 detecta **solo el 3,06 % de personas que declararon enfermedad**, y su accuracy supera a la línea base en **0,14 puntos porcentuales**. Por ello no sería útil como detector individual de enfermedad con la configuración actual. Naive Bayes sacrifica precisión/accuracy para lograr mayor recall; eso no lo convierte automáticamente en mejor modelo para todos los fines.

Una mejora futura sería optimizar un umbral de clasificación con datos de validación **sin consultar T4** y evaluarlo una sola vez en una partición final independiente.

## Cruce departamental con NBI (exploratorio)

Se intersectaron **16 unidades territoriales** de la EPHC 2024 con la tabla departamental del INE (NBI de hogares, Censo 2022). Correlaciones Spearman, sin ajuste por comparaciones múltiples ni pretensión causal:

| Dimensión NBI | Rho | p exploratorio |
|---|---:|---:|
| Al menos una NBI | −0,243 | 0,365 |
| Calidad vivienda | −0,184 | 0,495 |
| Infraestructura sanitaria | −0,380 | 0,147 |
| Acceso educación | −0,056 | 0,837 |
| Capacidad subsistencia | 0,294 | 0,269 |

Ninguno de los cinco indicadores muestra evidencia estadística suficiente de una relación monotónica departamental con la prevalencia de enfermedad declarada. **No implica** que las necesidades básicas no afecten la salud; las fuentes tienen unidades, fechas y mecanismos de medición diferentes, y solo se dispone de 16 observaciones territoriales.

## Respuesta consolidada a la pregunta

Se identificaron seis perfiles sociodemográficos con diferencias descriptivas de **aproximadamente 20 % a 40 %** en la prevalencia de enfermedad reciente declarada durante T4. No obstante, la baja separación entre clusters y la fuerte importancia de la edad limitan la interpretación de los perfiles como categorías independientes. Las variables disponibles permiten clasificar enfermedad reciente algo mejor que el azar en términos de AUC-ROC, pero no identificar de forma fiable los casos positivos con el umbral utilizado. El NBI aporta contexto territorial, aunque el cruce no confirmó una asociación ecológica robusta.

**Conclusión académica:** la investigación sí permite describir patrones y reconocer límites de predicción; no permite atribuir causas, formular diagnósticos ni estimar qué personas se enfermarán en el futuro.

## Archivos y fuentes

- \`output/fase3/metricas_clasificacion.csv\`
- \`output/fase3/resumen_perfiles.csv\`
- \`output/fase3/correlaciones_nbi.csv\`
- \`output/fase3/errores_subgrupos.csv\`
- \`output/fase3/resumen_ejecucion.json\`
- INE, Censo 2022, [Tríptico NBI](https://www.ine.gov.py/Publicaciones/Biblioteca/documento/262/NIB%20-%20TRIPTICO.pdf).
- INE, Encuesta Permanente de Hogares Continua 2024, [microdatos](https://www.ine.gov.py/microdatos/).

## Estado de entrega

**Implementados y ejecutados:** Python, pruebas unitarias, notebook HTML, validación temporal, comparaciones, serialización de modelos y resultados de NBI. **A revisar con el grupo:** informe final integrador PDF según rúbrica; discusión de estabilidad de clusters, confusión por edad y desempeño predictivo; presentación oral y declaración de apoyo de IA.
