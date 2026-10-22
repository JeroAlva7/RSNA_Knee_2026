# Referencias y recursos externos

Este repositorio contiene la implementación del proyecto y **utiliza explícitamente** conjuntos de datos, librerías y modelos preentrenados de terceros. El origen de esos recursos se reconoce aquí; esto no implica que los datos, el modelo base o las etiquetas auxiliares hayan sido creados por el participante.

## Competencia y datos

**RSNA — Kaggle.** *RSNA Knee Abnormality Detection*.
- [Página oficial de la competencia](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [Datos y descripción](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
- [Evaluación y reglas](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

**stevenleehans (Kaggle).** *RSNA Knee LLM Report Labels*.
- [Dataset auxiliar](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)
- Archivo usado para EXP-009, EXP-010 y EXP-011: **`llm_labels_v4_blend.csv`**.
- Son **weak labels derivadas de reportes**, no etiquetas gold verificadas.

## Modelos preentrenados y literatura

**Meta AI.** *DINOv2 Small (PyTorch), versión 1*.
- [Modelo distribuido en Kaggle](https://www.kaggle.com/models/metaresearch/dinov2)
- Utilizado como extractor congelado en EXP-007 a EXP-010 y con ajuste fino de sus últimos dos bloques en EXP-011.

**Oquab, M., et al. (2023).** *DINOv2: Learning Robust Visual Features without Supervision*. arXiv:2304.07193.
- <https://arxiv.org/abs/2304.07193>

## Bibliotecas de implementación

- [pydicom](https://pydicom.github.io/pydicom/stable/): lectura de archivos DICOM.
- [PyTorch](https://pytorch.org/): entrenamiento del backbone y de las cabezas de predicción.
- [Hugging Face Transformers, documentación de DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2): carga del modelo preentrenado.
- [scikit-learn](https://scikit-learn.org/stable/): regresión logística, PCA y métricas de evaluación en experimentos anteriores.

## Alcance de la atribución

El código `src/`, los notebooks y el historial de experimentos documentan las modificaciones y decisiones de este proyecto. Los pesos iniciales de DINOv2 son externos; los pesos de `exp011_best.pt` son un **checkpoint derivado del ajuste fino** de DINOv2 y se conservan fuera de GitHub como resultado de Kaggle. Los modelos no se reivindican como desarrollados desde cero.

La evaluación oficial del mejor resultado se encuentra en la [página de submissions de la competencia](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions) y en la [captura de evidencia](docs/evidencia/exp011_public_score_0_808.png).
