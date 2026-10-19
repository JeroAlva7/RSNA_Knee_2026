# RSNA Knee Abnormality Detection - Experiment Log

| ID | Fecha | Notebook | Descripción | Public Score | Git Commit |
|---|---|---|---|---:|---|
| EXP-001 | 2026-10-14 | RSNA_Knee_Baseline_01 | Probabilidad constante 0.5 | 0.500 | ad4e427 |
| EXP-002 | 2026-10-14 | RSNA_EXP002_Metadata_Gold | Logistic Regression con metadata de series MRI | 0.516 | 5866eb6 |
| EXP-003 | 2026-10-14 | RSNA_EXP003_Report_Teacher_Metadata_Student | TF-IDF report teacher + metadata student con soft pseudo-labels | 0.511 | 6064adf |
| EXP-004 | 2026-10-14 | RSNA_EXP004_Target_Hybrid | Target-wise hybrid de gold metadata y pseudo-labels seleccionados. Macro OOF 0.5788 | 0.518 | 2bcb5d7 |
| EXP-005 | 2026-10-14 | RSNA_EXP005_DICOM_Headers | 301 features de headers DICOM. Macro OOF 0.5358. No enviado a competencia | — | 368e90b |
| EXP-006 | 2026-10-14 | RSNA_EXP006_Hybrid_DICOM_Overlay | EXP-004 con overlay DICOM para MCL, meniscos y Baker's. OOF estimado 0.6164 | 0.516 | 1413fa9 |
| EXP-007 | 2026-10-14 | RSNA_EXP007_DINOv2_2D5_Gold | DINOv2 Small con MRI 2.5D multi-plane, pooling mean/max y PCA24 + Logistic Regression | 0.667 | 048b36d |
| EXP-008 | 2026-10-17 | RSNA_EXP008_DINOv2_Geometry | DINOv2 con 8 ventanas, recorte físico y resolución 336. Mejor Macro OOF 0.6528. No enviado a Kaggle | — | 4a391cd |
| EXP-009 | 2026-10-17 | RSNA_EXP009_DINOv2_WeakLabels (Version 2) | DINOv2 Small con 4349 weak labels v4, LOGIT y mezcla 90% weak / 10% gold. Macro AUC local 0.7461 | 0.762 | 173c42c |


## Mejor resultado actual

- Experimento: EXP-009
- Notebook: RSNA_EXP009_DINOv2_WeakLabels
- Versión de Kaggle: 2
- Estado: Succeeded
- Public Score: 0.762
- Macro AUC local: 0.7461
- Modelo: DINOv2 Small + weak labels v4 + LOGIT
- Entrenamiento weak: 4349 estudios
- Evaluación gold: 58 estudios
- Mezcla final: 90% weak / 10% EXP-007
- Mejor score anterior: 0.667 (EXP-007)
- Mejora pública: +0.095
- Objetivo: 0.800

Nota: el modelo y la mezcla final se seleccionaron usando
los 58 estudios gold. Por ello, el resultado local puede
ser optimista. El Public Score oficial de Kaggle es 0.762.