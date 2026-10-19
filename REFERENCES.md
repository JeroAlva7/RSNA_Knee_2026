# Referencias

Este proyecto implementa código propio.

Se consultan la documentación oficial de Kaggle/RSNA y discusiones técnicas de la competencia únicamente como referencias conceptuales. No se copian notebooks ni soluciones de otros competidores.

## Kaggle

- RSNA Knee Abnormality Detection - Overview
- RSNA Knee Abnormality Detection - Data
- RSNA Knee Abnormality Detection - Evaluation


## Recursos utilizados en EXP-009

### Competencia RSNA
RSNA Knee Abnormality Detection. Kaggle.
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

### Dataset de etiquetas auxiliares
RSNA Knee LLM Report Labels.
Autor de Kaggle: stevenleehans.
Archivo utilizado: llm_labels_v4_blend.csv.
https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels

### Modelo preentrenado
Meta AI. DINOv2 Small, PyTorch, Version 1.
https://www.kaggle.com/models/metaresearch/dinov2

### Publicación científica
Oquab et al. (2023).
DINOv2: Learning Robust Visual Features without Supervision.
https://arxiv.org/abs/2304.07193

## Desarrollo del proyecto

El procesamiento de imágenes DICOM, la extracción de
características, el entrenamiento de los clasificadores
y la combinación de predicciones están implementados
en los scripts originales del proyecto.

Se utilizaron recursos externos debidamente identificados,
incluyendo DINOv2 preentrenado y las etiquetas auxiliares
derivadas de reportes radiológicos.

