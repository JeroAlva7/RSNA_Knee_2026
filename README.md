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
- Mejor Public Score actual: **0.667 (EXP-007)**.
- Última submission: **EXP-007 = 0.667**.
- 1/5 submissions utilizadas en la ventana actual.

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

## Próxima etapa

Desarrollar modelos basados en los píxeles de las imágenes MRI/DICOM en lugar de continuar optimizando únicamente metadata.