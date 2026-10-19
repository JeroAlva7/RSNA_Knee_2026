# RSNA Knee Abnormality Detection

Proyecto para la competencia de Kaggle RSNA Knee Abnormality Detection.

## Estado actual

- Entorno virtual configurado.
- Kaggle API configurada.
- CSV de la competencia descargados localmente para análisis.
- Exploración inicial realizada con `inspect_data.py`.
- Pipeline de submissions de Kaggle configurado.
- Experimentos de metadata, pseudo-labeling y DICOM headers realizados.
- Public Score inicial: 0.500.
- Mejor Public Score actual: **0.762 (EXP-009)**.
- Última submission: **EXP-009 = 0.762**, Version 2, Succeeded.
- Mejor resultado anterior: 0.667 (EXP-007).
- Mejora obtenida: +0.095 de Macro AUC.
- Objetivo: 0.800.

## Estructura

- `src/`: código fuente de los experimentos.
- `notebooks/`: notebooks exactos utilizados en Kaggle.
- `experiments/`: historial de experimentos y scores.
- `inspect_data.py`: análisis inicial de los datos.
- `requirements.txt`: dependencias del proyecto.
- `REFERENCES.md`: referencias utilizadas.

Los datasets, imágenes DICOM, modelos, submissions y credenciales no se incluyen en el repositorio.

## Experimentos

El historial completo se encuentra en:

`experiments/EXPERIMENTS.md`


## Mejor experimento actual: EXP-009

El EXP-009 utiliza DINOv2 Small para extraer características
de imágenes de resonancia magnética de rodilla.

Se utilizaron 4349 estudios con etiquetas auxiliares derivadas
de reportes radiológicos (weak labels) y 58 estudios con
etiquetas gold para la evaluación local.

### Configuración

- Modelo base: Meta DINOv2 Small.
- Etiquetas auxiliares: llm_labels_v4_blend.csv.
- Clasificador weak seleccionado: Logistic Regression.
- Combinación: 90% weak model y 10% EXP-007.
- Macro AUC local: 0.7461.
- Public Score de Kaggle: **0.762**.
- Submission: EXP-009, Version 2, Succeeded.

### Notebook en Kaggle

https://www.kaggle.com/code/jerorojasalvarado/rsna-exp009-dinov2-weaklabels

### Reproducibilidad

El experimento se ejecutó en Kaggle con GPU T4 x2.

El código fuente se encuentra en
`src/exp009_dinov2_weaklabels.py`.

El notebook de Kaggle se conserva en
`notebooks/exp009_dinov2_weaklabels/`.

Los datos DICOM, pesos de modelos preentrenados,
credenciales y archivos de submission no se incluyen
en el repositorio.

## Estado del proyecto

Los experimentos realizados hasta EXP-009 están
documentados en `experiments/EXPERIMENTS.md`.

El mejor Public Score confirmado hasta ahora es **0.762**.