# RSNA Knee Abnormality Detection — Kaggle 2026

Proyecto de aprendizaje y experimentación en detección de anomalías de rodilla a partir de resonancias magnéticas (MRI) en formato DICOM. Se desarrollaron once experimentos que evolucionaron desde un baseline constante y clasificadores con metadatos hasta un modelo DINOv2 Small ajustado (*fine-tuning*) sobre etiquetas auxiliares.

## Resultado principal

| Métrica / dato | Resultado comprobado |
|---|---|
| **Mejor Public Score** | **0.808** |
| Experimento ganador | **EXP-011 — DINOv2 Small Fine-Tuning** |
| Kaggle submission | `RSNA_EXP011_FineTune_DINOv2`, **Version 1**, Succeeded |
| Fecha de confirmación en Kaggle | **20 de octubre de 2026** |
| Macro AUC local EXP-011 (58 estudios gold) | **0.7811** |
| AUC de validación auxiliar EXP-011 (weak holdout) | **0.7778** |
| Public Score anterior: EXP-009 | **0.762** |
| Mejora pública | **+0.046** |
| Objetivo planteado | **Public Score ≥ 0.800, alcanzado** |

**Importante:** `0.808` es el **Public Score** oficial mostrado por Kaggle, no una garantía del Private Score que se calcula sobre un conjunto oculto distinto. El `0.7811` se refiere a evaluación local y no debe confundirse con el leaderboard.

[Competencia Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · [Notebook EXP-011](https://www.kaggle.com/code/jerorojasalvarado/rsna-exp011-finetune-dinov2) · [Evidencia del Public Score](docs/evidencia/exp011_public_score_0_808.png)

## Objetivo y datos

El objetivo es predecir **12 condiciones**: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion y Fracture. La métrica de la competencia es el promedio de ROC-AUC por condición (*macro AUC*).

- **4,349 estudios de entrenamiento con weak labels**, procedentes del dataset auxiliar `RSNA Knee LLM Report Labels` (`llm_labels_v4_blend.csv`).
- **58 estudios con etiquetas gold**, utilizados para la auditoría local en EXP-011, sin entrenamiento del modelo ni selección de época con esas etiquetas.
- DICOM y tablas del dataset oficial de la competencia, obtenidos mediante los inputs de Kaggle.
- DINOv2 Small preentrenado de Meta AI como base visual; **no** se entrenó un transformer desde pesos aleatorios.

Las etiquetas auxiliares son estimaciones provenientes de reportes, no diagnósticos confirmados. El conjunto gold es pequeño, de modo que las estimaciones locales pueden ser variables. Este repositorio es un proyecto de experimentación y **no es un dispositivo médico**.

## Progresión de resultados

| Experimento | Método principal | AUC local documentado | Public Score |
|---|---|---:|---:|
| EXP-001 | Predicción constante | — | 0.500 |
| EXP-002 | Metadata y regresión logística | — | 0.516 |
| EXP-003 | Report teacher / metadata student | — | 0.511 |
| EXP-004 | Clasificadores híbridos por condición | 0.5788 OOF | 0.518 |
| EXP-005 | Headers DICOM | 0.5358 OOF | No enviado |
| EXP-006 | Overlay DICOM | 0.6164 OOF estimado | 0.516 |
| EXP-007 | DINOv2 2.5D + PCA + LR | 0.6370 | 0.667 |
| EXP-008 | DINOv2 geometría/crop/ventanas | 0.6528 OOF | No enviado |
| EXP-009 | DINOv2 congelado + weak labels | 0.7461 | 0.762 |
| EXP-010 | Estadísticas por plano + MLP | 0.7640 | No enviado |
| **EXP-011** | **Fine-tuning de DINOv2 + atención MIL** | **0.7811** | **0.808** |

**Notas:** las cifras OOF, gold y public corresponden a protocolos distintos y no se comparan como si fueran una misma muestra. EXP-010 tuvo validación `FAIL` bajo su umbral local predefinido y no se envió; EXP-011 también tuvo filtro local `FAIL` (`0.7811 < 0.820`), pero sí fue enviado posteriormente y alcanzó `0.808` público. El historial detallado y los commits están en [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md).

## ¿Cómo funciona EXP-011?

1. Selecciona series MRI por plano (`sagittal`, `coronal`, `axial`) y características del protocolo; extrae **12 vistas** por estudio, con cortes vecinos apilados en tres canales.
2. Aplica un recorte físico aproximado de **140 mm** cuando se dispone de `PixelSpacing` y redimensiona a 224 × 224.
3. Usa **DINOv2 Small**; mantiene congelada la mayor parte del backbone y ajusta **sus dos últimos bloques transformer**.
4. Agrega características mediante atención sobre vistas (**MIL, Multiple Instance Learning**) y predice conjuntamente las 12 etiquetas con una cabeza multietiqueta.
5. Entrena con 4,349 estudios weak y escoge la mejor época mediante un *holdout* separado de esos mismos estudios. Evalúa luego en los 58 gold.
6. Durante el envío oficial, Kaggle vuelve a ejecutar el notebook sobre su test oculto y extrae `/kaggle/working/submission.csv`.

El entrenamiento EXP-011 seleccionó la época **4**, con AUC de holdout `0.7778`. La ejecución visible de Kaggle tardó aproximadamente **2.20 horas**. [Detalles de implementación](INSTRUCCIONES_EXP011.md).

## Organización del repositorio

```text
RSNA_Knee_2026/
├── README.md
├── REFERENCES.md
├── INSTRUCCIONES_EXP010.md
├── INSTRUCCIONES_EXP011.md
├── requirements.txt
├── inspect_data.py
├── build_exp010_notebook.py
├── build_exp011_notebook.py
├── src/                            # Código fuente EXP-002 a EXP-011
├── notebooks/                      # Notebooks Kaggle + kernel-metadata.json
│   ├── exp001_baseline_01/
│   ├── exp002_metadata_gold/ ... exp007_dinov2_2d5_gold/
│   ├── exp009_dinov2_weaklabels/
│   ├── exp010_dinov2_multitask/
│   └── exp011_finetune_dinov2/
├── experiments/
│   ├── EXPERIMENTS.md              # Registro cronológico, scores y commits
│   └── results/                    # Diagnósticos pequeños de EXP-010/011
└── docs/
    ├── REPRODUCIBILIDAD.md
    └── evidencia/exp011_public_score_0_808.png
```

**EXP-008:** está registrado como script `src/exp008_dinov2_geometry.py`. A la fecha de esta revisión, no había un notebook correspondiente versionado en GitHub. Su ausencia no significa que su código fuente falte; si existe un `.ipynb` local, conviene conservarlo en `notebooks/exp008_dinov2_geometry/` tras verificarlo.

## Reproducción y entorno

Los scripts originales están en `src/`. EXP-010 y EXP-011 cuentan además con generadores de notebooks que incluyen los ajustes y entradas requeridos por Kaggle:

```powershell
# Ejecutar desde la raíz del proyecto, en PowerShell y con el entorno virtual activo
python build_exp010_notebook.py
python build_exp011_notebook.py
python -m py_compile .\src\exp010_dinov2_multitask.py .\src\exp011_finetune_dinov2.py
```

**Los comandos anteriores no entrenan el modelo.** `kaggle kernels push` sí crea una nueva versión y solicita una ejecución costosa: úsalo solo si se desea reproducir un experimento en Kaggle. Revisa primero [`docs/REPRODUCIBILIDAD.md`](docs/REPRODUCIBILIDAD.md).

`requirements.txt` documenta el entorno local de desarrollo en Windows, pero **no es una especificación completa del entorno GPU de Kaggle**: el notebook usa `torch` y `transformers`, disponibles en la imagen correspondiente de Kaggle y necesarios para ejecutar el modelo. Consultar dependencias al reproducir en otro entorno.

## Datos, resultados y atribución

Se versionan scripts, notebooks y diagnósticos pequeños. Se excluyen expresamente los **DICOM, credenciales, entornos virtuales, cachés, modelos preentrenados y pesos del checkpoint ajustado**. El mejor checkpoint (`exp011_best.pt`, ~89 MB) permanece como artefacto de la ejecución Kaggle y en el respaldo local descargado; no es necesario para consultar el código fuente.

Los datos pertenecen a sus respectivos proveedores y la implementación usa recursos públicos acreditados en [`REFERENCES.md`](REFERENCES.md). No se suben etiquetas de test privado ni archivos de submission como prueba de score: la puntuación oficial se confirma en la pestaña **Submissions** de Kaggle y mediante la [captura conservada](docs/evidencia/exp011_public_score_0_808.png).

## Conclusión

El proyecto pasó de un baseline de **0.500** a un **Public Score de 0.808** mediante experimentación progresiva. La mejora más relevante se consiguió al pasar de embeddings DINOv2 congelados y clasificadores separados (EXP-009: `0.762`) al ajuste fino de las últimas dos capas con atención multivista (EXP-011: `0.808`). **Se cumplió el objetivo público de 0.800**, sin confundirlo con la evaluación privada final.
