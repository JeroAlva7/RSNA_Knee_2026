# RSNA Knee Abnormality Detection - Experiment Log

| ID | Fecha | Notebook | Descripción | Public Score | Git Commit |
|---|---|---|---|---:|---|
| EXP-001 | 2026-10-14 | RSNA_Knee_Baseline_01 | Probabilidad constante 0.5 | 0.500 | ad4e427 |
| EXP-002 | 2026-10-14 | RSNA_EXP002_Metadata_Gold | Logistic Regression con metadata de series MRI | 0.516 | 5866eb6 |
| EXP-003 | 2026-10-14 | RSNA_EXP003_Report_Teacher_Metadata_Student | TF-IDF report teacher + metadata student con soft pseudo-labels | 0.511 | 6064adf |
| EXP-004 | 2026-10-14 | RSNA_EXP004_Target_Hybrid | Target-wise hybrid de gold metadata y pseudo-labels seleccionados. Macro OOF 0.5788 | 0.518 | 2bcb5d7 |
| EXP-005 | 2026-10-14 | RSNA_EXP005_DICOM_Headers | 301 features de headers DICOM. Macro OOF 0.5358. No enviado a competencia | — | 368e90b |
| EXP-006 | 2026-10-14 | RSNA_EXP006_Hybrid_DICOM_Overlay | EXP-004 con overlay DICOM para MCL, meniscos y Baker's. OOF estimado 0.6164 | 0.516 | 1413fa9 |