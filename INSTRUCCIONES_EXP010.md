# EXP-010 — RSNA Knee (objetivo Public Macro AUC 0.800)

**Punto de partida:** EXP-009 logró `0.762` Public Score en Kaggle y `0.7461` Macro AUC local sobre 58 estudios gold.

**Limitación:** Ningún código puede garantizar un Public Score de 0.800 o un score final privado de 0.800. El método usa una pequeña validación gold para decidir si vale la pena enviar y no asegura que el test oculto mejore.

## Idea y diferencias frente a EXP-009

Se utiliza el **mismo recorrido de resonancias y el mismo DINOv2 Small** de EXP-009: 4 ventanas por serie y 2 series por plano. Por cada plano se guardan `mean`, `max`, `std` y `mediana` de los embeddings. Los dos primeros quedan exactamente en el mismo orden que EXP-009: eso permite reconstruir su modelo baseline 0.7461 local sin una segunda pasada de GPU. Las otras estadísticas se usan para entrenar una red neuronal propia multietiqueta sobre 4,349 weak labels. Se elige la época en un 15% de los estudios weak, nunca en las 58 etiquetas gold. La validación gold se utiliza para comparar el candidato y aplicar un umbral conservador.

## Preparación local en VS Code (PowerShell)

1. Descarga este paquete ZIP y colócalo en `C:\Users\Admin\Documents\RSNA_Knee_2026`.
2. Abre VS Code en esa carpeta y activa el entorno: `\.venv\Scripts\Activate.ps1` (en PowerShell debe usarse `.\.venv\Scripts\Activate.ps1`).
3. Extrae el ZIP **en la raíz del repositorio**, conservando `src/exp010_dinov2_multitask.py` y `build_exp010_notebook.py`.
4. Genera el notebook Kaggle ejecutando `python build_exp010_notebook.py`.
5. Debes obtener la carpeta `notebooks/exp010_dinov2_multitask` con el `.ipynb` y `kernel-metadata.json`.
6. Comprueba la sintaxis: `python -m py_compile .\src\exp010_dinov2_multitask.py .\build_exp010_notebook.py`. No requiere descargar resonancias ni ejecutar GPU en Windows.
7. Guarda el trabajo original: `git add src/exp010_dinov2_multitask.py build_exp010_notebook.py notebooks/exp010_dinov2_multitask/` y `git commit -m "Add original EXP-010 DINOv2 multitask experiment"`, `git push origin main`.

## Ejecución en Kaggle

La ruta más reproducible para crear el nuevo notebook es la API de Kaggle ya configurada:

```powershell
kaggle kernels push -p ".\notebooks\exp010_dinov2_multitask"
```

ATENCIÓN: **este comando inicia una ejecución Kaggle** que puede durar varias horas. No lo pulses sin aceptar el tiempo de cómputo. El `kernel-metadata.json` heredó los 3 inputs y GPU de tu EXP-009: competencia RSNA, modelo DINOv2 Small y LLM Report Labels v4, con Internet desactivado. Confirma visualmente esos inputs y la GPU T4 x2 desde Kaggle.

Notebook esperado: `https://www.kaggle.com/code/jerorojasalvarado/rsna-exp010-dinov2-multitask`

Al terminar, revisa el output:

```
EXP-009 BASELINE GOLD MACRO AUC: 0.7461   # aprox.
EXP010 BLEND MLP=0.25 GOLD MACRO AUC: ...
EXP010 BLEND MLP=0.50 GOLD MACRO AUC: ...
EXP010 BLEND MLP=0.75 GOLD MACRO AUC: ...
EXP010 BEST GOLD: ...
RECOMENDACION SUBMIT: SI / NO
```

Antes de enviar, confirma que se generó `submission.csv`, `exp010_diagnostics.json` y `exp010_gold_scores.csv`, y que el notebook terminó sin errores.

**Si dice `RECOMENDACION SUBMIT: NO`, NO ENVÍES.** Tu `0.762` de EXP-009 sigue siendo el mejor score público confirmado. Si dice `SI`, es un candidato y puedes evaluar si deseas invertir las ~3–4 horas adicionales de re-ejecución con el test oculto. **Tampoco garantiza superar 0.762 ni llegar a 0.800.**

## Competencia y tiempo límite

- Kaggle: `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection`
- Notebook con GPU: máximo **9 horas por ejecución**.
- Internet de notebook: desactivado.
- `submission.csv` en `/kaggle/working/`.
- Fecha límite: **22 de octubre de 2026 a las 23:59 UTC** (aprox. **17:59 horas de Guadalajara**). No esperes al último minuto.
- Conserva EXP-009 y selecciona su submission si EXP-010 resulta inferior y las reglas de selección final lo requieren.

## Originalidad y atribución

Este programa es un diseño propio que reutiliza las funciones que ya desarrollaste en EXP-009. Sus entradas utilizan el modelo preentrenado Meta DINOv2 Small y el dataset auxiliar `stevenleehans/rsna-knee-llm-report-labels`, que deben seguir acreditados en `REFERENCES.md`. No copia una solución de otro competidor.
