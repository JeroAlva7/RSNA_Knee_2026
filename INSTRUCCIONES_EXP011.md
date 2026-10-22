# EXP-011 — DINOv2 Small Fine-Tuning + atención multivista (MIL)

**Estado final: submission oficial enviada y evaluada exitosamente. Public Score: `0.808` (20/oct/2026).** Este archivo registra el método, los comandos empleados y los resultados, pero **no es necesario volver a ejecutar el notebook**.

## Método

- 4,349 estudios weak provenientes de `llm_labels_v4_blend.csv`, con separación de holdout ~13% para elegir época.
- 58 estudios gold para validación final local, sin entrenar la red ni seleccionar época con ellos.
- Hasta 2 series por plano (`sagittal`, `coronal`, `axial`) y 2 posiciones por serie; tres slices vecinos concatenados en canales RGB (12 vistas por rodilla).
- Recorte físico central de aproximadamente 140 mm (según `PixelSpacing`) y resolución 224 × 224.
- Meta DINOv2 Small: entrenamiento de **los últimos dos bloques transformer**, resto del backbone congelado.
- Atención MIL sobre las vistas + cabeza predictiva multietiqueta para 12 anomalías. Guardado del mejor checkpoint según weak holdout como `exp011_best.pt`.

## Preparación desde VS Code (PowerShell)

```powershell
cd "C:\Users\Admin\Documents\RSNA_Knee_2026"
.\.venv\Scripts\Activate.ps1
python build_exp011_notebook.py
python -m py_compile .\src\exp011_finetune_dinov2.py .\build_exp011_notebook.py
```

Esto genera o verifica `notebooks/exp011_finetune_dinov2/rsna-exp011-finetune-dinov2.ipynb` y `kernel-metadata.json`. Generar el notebook **no** inicia entrenamiento.

## Reproducción de la ejecución Kaggle — opcional, consume GPU

```powershell
kaggle kernels push -p ".\notebooks\exp011_finetune_dinov2"
kaggle kernels status jerorojasalvarado/rsna-exp011-finetune-dinov2
```

`kaggle kernels push` inicia una **versión nueva** y ejecución nueva; **no usarlo para el cierre documental**. El notebook requiere el input de la competencia, el modelo Meta DINOv2 Small y el dataset auxiliar de etiquetas, con GPU T4 e Internet desactivado.

[Notebook en Kaggle](https://www.kaggle.com/code/jerorojasalvarado/rsna-exp011-finetune-dinov2)

## Validación local real (primera ejecución, 19-20/oct/2026)

| Medida | Resultado |
|---|---:|
| Estudios weak | 4,349 |
| Estudios gold | 58 |
| Mejor época | **4** |
| **AUC holdout weak** | **0.7778** |
| **Macro AUC gold** | **0.7811** |
| Duración del entrenamiento/evaluación visible | ~**2.20 horas** |
| Filtro predefinido gold `>= 0.820` | **FAIL** |

El filtro era una heurística conservadora para decidir si recomendar el envío, **no una medida estadística de garantía**. EXP-011 no alcanzó `0.820` gold local, pero el participante decidió enviarlo posteriormente a competición para obtener una evaluación oficial.

Los detalles numéricos y la métrica por cada una de las 12 anomalías están en [`experiments/results/exp011_diagnostics.json`](experiments/results/exp011_diagnostics.json).

## Resultado oficial de Kaggle (20/oct/2026)

- Notebook: **`RSNA_EXP011_FineTune_DINOv2`**.
- Versión: **1**.
- Estado: **Succeeded**.
- **Public Score: `0.808`**, confirmado en **Submissions**.
- EXP-009 anterior: `0.762`. Ganancia: `+0.046`.
- Objetivo: Public Score `>= 0.800`, **alcanzado**.
- Evidencia visual: [`docs/evidencia/exp011_public_score_0_808.png`](docs/evidencia/exp011_public_score_0_808.png).

**Advertencia:** el resultado `0.808` es público y no prueba que el Private Score futuro sea igual. El `submission.csv` producido por la primera ejecución visible tenía tres estudios; la evaluación oficial hizo un rerun privado sobre test oculto. Por eso no se debe intentar subir manualmente el CSV de tres filas.

## Conservación del modelo y reproducibilidad

El notebook escribió `exp011_best.pt` (checkpoint ~89 MB). Se mantiene como archivo descargado o output de Kaggle; por su tamaño y porque el repositorio conserva el código original y sus inputs, **no se incluye por defecto en GitHub**. Tampoco se suben DICOM ni credenciales. Las referencias y licencias de terceros se consultan en [`REFERENCES.md`](REFERENCES.md).
