# EXP-010 — DINOv2 con estadísticas por plano y MLP multietiqueta

**Estado:** experimento ejecutado en Kaggle, **no enviado a competición**. El resultado histórico no debe confundirse con el mejor score público del proyecto, que posteriormente alcanzó **0.808 con EXP-011**.

## Idea del experimento

Se reutiliza la extracción visual de EXP-009 con DINOv2 Small y se agregan, por cada plano anatómico, estadísticas `mean`, `max`, `std` y `median` sin repetir los pases de DINOv2. El brazo adicional es una red MLP para las 12 condiciones, entrenada con 4,349 weak labels. Se compara con EXP-009 sobre 58 gold.

## Preparación local desde la raíz del repositorio (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python build_exp010_notebook.py
python -m py_compile .\src\exp010_dinov2_multitask.py .\build_exp010_notebook.py
```

El generador crea `notebooks/exp010_dinov2_multitask/rsna-exp010-dinov2-multitask.ipynb` y su `kernel-metadata.json`. **No consume GPU en Windows.**

## Ejecución de referencia en Kaggle (histórica)

```powershell
kaggle kernels push -p ".\notebooks\exp010_dinov2_multitask"
kaggle kernels status jerorojasalvarado/rsna-exp010-dinov2-multitask
```

**Atención:** `push` sube una nueva versión e inicia una ejecución de Kaggle. Este experimento ya se ejecutó; no repetir el comando solo para documentarlo. Se utilizó GPU T4 e Internet desactivado, con inputs de la competencia RSNA, DINOv2 Small y el dataset auxiliar de stevenleehans.

[Notebook EXP-010 en Kaggle](https://www.kaggle.com/code/jerorojasalvarado/rsna-exp010-dinov2-multitask)

## Resultado real de EXP-010 (19 de octubre de 2026)

| Medida | Valor |
|---|---:|
| Estudios weak / gold | 4,349 / 58 |
| Baseline EXP-009 reproducido | 0.7463 |
| Blend MLP 0.25 | 0.7569 |
| Blend MLP 0.50 | 0.7616 |
| **Blend MLP 0.75** | **0.7640** |
| Aumento sobre baseline | +0.0177 |
| Umbral gold exigido para recomendar envío | 0.790 |
| Umbral mínimo de ganancia | +0.035 |
| **RECOMENDACION SUBMIT** | **NO** |
| Tiempo de ejecución visible | ~2.56 h |
| Public Score | **No enviado / no disponible** |

Archivos de diagnóstico preservados en [`experiments/results/`](experiments/results/): `exp010_diagnostics.json` y `exp010_gold_scores.csv`.

El notebook generó un `submission.csv` para los tres estudios de test visibles, pero **no tuvo evaluación pública oficial**. Cuando el filtro local era `FAIL`, el código guardó el baseline para test; por ello ese archivo no constituye una submission mejorada de EXP-010.

## Limitaciones y decisión

El mejor gold AUC de EXP-010 fue `0.7640`, una ganancia inferior al filtro predefinido. **No se envió.** Más adelante, EXP-011 obtuvo una macro AUC gold de `0.7811` y, tras su submission oficial, un **Public Score de `0.808`**. Esta documentación se conserva para trazabilidad del intento intermedio.

Los recursos externos y créditos están documentados en [`REFERENCES.md`](REFERENCES.md).
