# Historial de experimentos — RSNA Knee Abnormality Detection

Registro de código, evaluaciones locales y resultados públicos. Fechas indicadas según las sesiones de desarrollo y las evidencias conservadas. Los scores son **Macro ROC-AUC**, salvo las columnas marcadas de otra forma.

| ID | Fecha | Notebook/implementación | Método y resultado local | Public Score | Referencia Git |
|---|---|---|---|---:|---|
| EXP-001 | 2026-10-14 | `RSNA_Knee_Baseline_01` | Probabilidad constante 0.5 | **0.500** | `ad4e427` |
| EXP-002 | 2026-10-14 | `RSNA_EXP002_Metadata_Gold` | Regresión logística con metadata de MRI | **0.516** | `5866eb6` |
| EXP-003 | 2026-10-14 | `RSNA_EXP003_Report_Teacher_Metadata_Student` | Teacher TF-IDF de reportes y student de metadatos; pseudoetiquetas | **0.511** | `6064adf` |
| EXP-004 | 2026-10-14 | `RSNA_EXP004_Target_Hybrid` | Híbrido por condición; macro OOF **0.5788** | **0.518** | `2bcb5d7` |
| EXP-005 | 2026-10-14 | `RSNA_EXP005_DICOM_Headers` | 301 features DICOM, macro OOF **0.5358**; no enviado | — | `368e90b` |
| EXP-006 | 2026-10-14 | `RSNA_EXP006_Hybrid_DICOM_Overlay` | Overlay de MCL, meniscos y Baker's; OOF **0.6164** estimado | **0.516** | `1413fa9` |
| EXP-007 | 2026-10-14 | `RSNA_EXP007_DINOv2_2D5_Gold` | DINOv2 Small + imágenes 2.5D + PCA24 + regresión logística; gold OOF **0.6370** | **0.667** | `048b36d` |
| EXP-008 | 2026-10-17 | `src/exp008_dinov2_geometry.py` | Ocho ventanas/serie, crop físico y 336 px; mejor OOF **0.6528**; no enviado | — | `4a391cd` |
| EXP-009 | 2026-10-17 | `RSNA_EXP009_DINOv2_WeakLabels`, **v2** | DINOv2 congelado, 4,349 weak v4 + mezcla 90% weak/10% gold; macro local **0.7461** | **0.762** | `173c42c` |
| EXP-010 | 2026-10-19 | `RSNA_EXP010_DINOv2_Multitask`, **v1** | Estadísticas por plano + MLP multitarea; gold **0.7640**, baseline 0.7463; filtro local **FAIL**; no enviado | — | `3c1d463` |
| **EXP-011** | **2026-10-19 a 20** | **`RSNA_EXP011_FineTune_DINOv2`, v1** | **Fine-tuning de los últimos 2 bloques + atención MIL; gold 0.7811; weak holdout 0.7778; filtro local FAIL** | **0.808** | **`571de8f`** |

## Estado final confirmado — 20 de octubre de 2026

- **Mejor experimento y submission oficial:** EXP-011 (`RSNA_EXP011_FineTune_DINOv2`, versión 1, `Succeeded`).
- **Public Score: `0.808`**. Objetivo público `>= 0.800` **alcanzado**.
- **Último Public Score previo:** EXP-009 `0.762`; diferencia **`+0.046`**.
- La competencia vuelve a ejecutar el notebook en un test oculto. La puntuación pública es diferente de la métrica local y no anticipa necesariamente el **Private Score**.
- Evidencia: [`docs/evidencia/exp011_public_score_0_808.png`](../docs/evidencia/exp011_public_score_0_808.png).
- URL del notebook: <https://www.kaggle.com/code/jerorojasalvarado/rsna-exp011-finetune-dinov2>.
- Diagnóstico reproducible: [`results/exp011_diagnostics.json`](results/exp011_diagnostics.json).

## Detalle EXP-010 — ejecutado, no enviado

- Entrenamiento weak: **4,349 estudios**; gold: **58**.
- Baseline reproducido EXP-009: **0.7463**.
- Mezclas MLP 0.25/0.50/0.75: **0.7569 / 0.7616 / 0.7640**.
- Mejor peso de MLP: **0.75**.
- Umbrales predefinidos: `>= 0.790` gold y ganancia `>= 0.035` sobre baseline; **no los alcanzó**.
- Archivos: [`results/exp010_diagnostics.json`](results/exp010_diagnostics.json) y [`results/exp010_gold_scores.csv`](results/exp010_gold_scores.csv).
- Tiempo visible: ~**2.56 horas**.

## Detalle EXP-011 — validación y submission oficial

- **4,349** estudios weak y **58** gold.
- **AUC weak holdout:** `0.7778226292` en **época 4**.
- **Macro AUC gold:** `0.7811077217`.
- **Tiempo de ejecución visible:** `2.2003` horas, aproximadamente.
- Filtro local predefinido: gold `>= 0.820` y weak holdout `>= 0.70`: **FAIL** porque el gold fue menor a 0.820. El filtro no impedía técnicamente presentar una submission: el EXP-011 se envió posteriormente por decisión del participante.
- **Resultado público de la submission de EXP-011: 0.808 (Succeeded).** Este score reemplaza a EXP-009 como mejor resultado público registrado.
- Resultados por condición: ACL `0.6507`, MCL `0.6542`, Medial Meniscus `0.7055`, Lateral Meniscus `0.7714`, Medial OA `0.8760`, Lateral OA `0.8182`, PF OA `0.7773`, Effusion `0.9006`, Synovitis `0.8315`, Baker's `0.8460`, Contusion `0.7598`, Fracture `0.7819`.

## Integridad y alcance de este registro

- **Exp-008** dispone de script fuente, pero no se encontró notebook versionado durante la revisión. No se debe inventar una versión de Kaggle no ejecutada.
- Los 58 gold son una evaluación reducida; usar sus resultados para seleccionar repetidamente modelos puede introducir optimismo o sobreajuste.
- Se excluyen los DICOM y los pesos binarios grandes del historial Git; los links oficiales y diagnósticos pequeños hacen trazable el trabajo.
- Los archivos `submission.csv` de los notebooks ejecutados manualmente pueden contener solo los tres estudios visibles. El leaderboard real se obtuvo mediante el **rerun privado de Kaggle** sobre el test de la competencia, no mediante un CSV local de tres filas.
