# EXP-011 | Fine-tuning DINOv2 Small — RSNA Knee

Objetivo: investigar si entrenar las últimas dos capas de DINOv2 sobre 4,349 estudios con weak labels mejora el resultado público de EXP-009 (0.762). **No existe garantía de 0.800.** No se copia el código de otros competidores; recursos externos se acreditan en `REFERENCES.md`.

## Instalación local

1. Extrae la carpeta ZIP desde la raíz `C:\Users\Admin\Documents\RSNA_Knee_2026`, conservando las rutas internas.
2. Activa el entorno: `.\.venv\Scripts\Activate.ps1`.
3. Genera el notebook de forma local, sin usar GPU: `python build_exp011_notebook.py`.
4. Verifica sintaxis: `python -m py_compile .\src\exp011_finetune_dinov2.py .\build_exp011_notebook.py`.
5. Comprueba la carpeta `notebooks/exp011_finetune_dinov2/` y los dos archivos.
6. Guarda el código en GitHub para trazabilidad con `git add src/exp011_finetune_dinov2.py build_exp011_notebook.py INSTRUCCIONES_EXP011.md notebooks/exp011_finetune_dinov2/`, `git commit -m "Prepare EXP-011 DINOv2 fine-tuning"`, `git push origin main`.

## Primera ejecución en Kaggle

**ATENCIÓN: este comando inicia inmediatamente el cálculo en Kaggle.**

`kaggle kernels push -p ".\notebooks\exp011_finetune_dinov2"`

Confirma que el notebook tiene GPU T4 y los tres inputs. El código construye un caché de ~7.9 GB de ventanas RGB de las imágenes, para no decodificar los DICOM en cada época. Si Kaggle carece de espacio temporal, fallará claramente; **no modifiques EXP-009**. El clasificador usa MIL con atención, ajuste fino de 2 bloques y weak holdout (13%).

El tiempo de extracción puede rondar 2–3 horas; entrenamiento 1–4 horas e inferencia escondida otros ~30–90 minutos. El rango real depende de Kaggle y **podría superar los límites**. El código incluye límites de tiempo y aborta si no puede evaluar todas las imágenes; nunca pone 0.5 silenciosamente para pacientes no procesados.

Revisa los logs finales:

- `BEST WEAK AUC ... epoch ...`
- `EXP011 GOLD MACRO AUC: ...`
- `EXP011 LOCAL GATE (>=0.820 gold, >=0.70 weak holdout): PASS/FAIL`
- `EXP011 COMPLETE ...`

Si la ejecución falla, comparte la primera línea de `Traceback` más el error. **No envíes a la competencia aunque diga PASS sin verificar en conjunto**. Si FAIL, no gastar submission.

EXP-011 crea `submission.csv` y `exp011_diagnostics.json` en `/kaggle/working/`. No es necesario conservar ni subir los datos temporales (los archivos de caché van a `/kaggle/temp`). El score público anterior 0.762 de EXP-009 queda como referencia.

**Plazo oficial:** 22 octubre 2026, 23:59 UTC (17:59 hora Guadalajara). Kaggle limita GPU submissions a nueve horas y requiere Internet OFF.
