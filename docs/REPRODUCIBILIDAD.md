# Reproducibilidad y verificación del proyecto

## 1. Código fuente y versiones

Repositorio: <https://github.com/JeroAlva7/RSNA_Knee_2026> (rama `main`). Los scripts `src/exp002_*.py` a `src/exp011_*.py` documentan cada fase. Los notebooks versionados viven en `notebooks/` junto a su `kernel-metadata.json`. El script EXP-008 está presente, pero la revisión del 22/oct/2026 no encontró un notebook EXP-008 versionado.

## 2. Entorno local de desarrollo

Entorno reportado: Windows, VS Code, PowerShell, Python en `.venv`. Los archivos `requirements.txt` describen un conjunto de dependencias locales, no la versión congelada de la imagen GPU de Kaggle. Para DINOv2 se requieren `torch` y `transformers`, además de `pydicom`, `numpy`, `pandas`, `Pillow`, `scikit-learn` y decodificadores DICOM adecuados.

**No es necesario ejecutar DINOv2 con todos los DICOM en Windows**: los experimentos GPU se ejecutaron en Kaggle sobre los inputs montados y con Internet OFF.

## 3. Inputs externos y privacidad

- Competencia: `rsna-knee-abnormality-detection`.
- Modelo: `metaresearch/dinov2/PyTorch/small/1`.
- Dataset adicional: `stevenleehans/rsna-knee-llm-report-labels`, archivo `llm_labels_v4_blend.csv`.

Los datos originales, credenciales, caches y los pesos preentrenados no se versionan. Los archivos `kernel-metadata.json` declaran los inputs de cada notebook.

## 4. Generar notebooks (sin ejecutar Kaggle)

```powershell
cd "C:\Users\Admin\Documents\RSNA_Knee_2026"
.\.venv\Scripts\Activate.ps1
python build_exp010_notebook.py
python build_exp011_notebook.py
python -m py_compile .\src\exp010_dinov2_multitask.py .\src\exp011_finetune_dinov2.py
```

Estos comandos solo generan/validan archivos. Un `kaggle kernels push -p ...` **sí** desencadena una nueva ejecución y no es necesario para archivar el resultado de EXP-011.

## 5. Evaluaciones y artefactos

- EXP-010: gold `0.7640`, gate `FAIL`, sin submission pública. Datos en `experiments/results/exp010_*`.
- EXP-011: weak holdout `0.7778`, gold `0.7811`, gate `FAIL`, pero enviado a competición posteriormente por decisión del usuario.
- **EXP-011 Public Score `0.808`, Succeeded, versión 1**, confirmado en Kaggle el 20/oct/2026. Captura `docs/evidencia/exp011_public_score_0_808.png`.
- La puntuación pública **no** se puede reconstruir solo con los tres estudios presentes en el `submission.csv` generado en modo desarrollo. El servidor Kaggle ejecuta su test oculto cuando se registra una submission de notebook.
- `exp011_best.pt` es un artefacto derivado de ~89 MB. Mantenerlo en outputs Kaggle o en almacenamiento externo; no es obligatorio subirlo a GitHub.

## 6. Comprobar si todos los cambios llegaron a GitHub

```powershell
git status --short --untracked-files=all
git fetch origin
git status -sb
git log -1 --oneline
git ls-files
```

No ejecutar `git add -A` sin revisar archivos no deseados. Preferir `git add` por directorio y archivo; revisar siempre `git diff --cached --stat` y `git diff --cached` antes de hacer commit. No subir jamás `.env`, `kaggle.json`, `.venv/`, `data/`, `models/` ni secretos.

## 7. Límite metodológico

Los resultados locales de EXP-007 a EXP-011 tienen diferencias de datos/protocolo. La validación gold consta de solo 58 muestras y no sustituye una evaluación independiente. `0.808` es el score público mostrado en Kaggle y puede diferir del Private Score final. El proyecto es educativo, no clínico.
