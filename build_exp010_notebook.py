"""Genera notebook independiente EXP010 reutilizando tu script EXP009.

Uso desde la raiz de RSNA_Knee_2026:
  python build_exp010_notebook.py
El notebook resultante se importa a Kaggle SIN entrenar nada localmente.
"""
from pathlib import Path
import json
import nbformat

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'src' / 'exp009_dinov2_weaklabels.py'
NEW = ROOT / 'src' / 'exp010_dinov2_multitask.py'
OLD_METADATA = ROOT / 'notebooks' / 'exp009_dinov2_weaklabels' / 'kernel-metadata.json'
FOLDER = ROOT / 'notebooks' / 'exp010_dinov2_multitask'
NAME = 'rsna-exp010-dinov2-multitask.ipynb'

for path in (OLD, NEW, OLD_METADATA):
    if not path.is_file():
        raise FileNotFoundError(f'Falta archivo requerido: {path}')

base = OLD.read_text(encoding='utf-8')
marker = 'if __name__ == "__main__":'
if base.count(marker) != 1:
    raise RuntimeError('No encuentro un unico bloque de ejecución EXP009 al final del archivo')
base = base[:base.rindex(marker)].rstrip() + '\n'
extra = NEW.read_text(encoding='utf-8')
# Segunda celda no debe disparar main010() dos veces.
if extra.count("if __name__ == '__main__':") != 1:
    raise RuntimeError('EXP010 no tiene marcador de ejecucion valido')

nb = nbformat.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
nb.cells = [
    nbformat.v4.new_markdown_cell('''# EXP-010 | DINOv2 + estadísticas por plano + MLP multietiqueta

**Desarrollo propio:** conserva el brazo EXP-009 y prueba una rama no lineal sobre las mismas inferencias de DINOv2. Usa 4,349 weak y 58 gold para auditoria. Antes de enviar mira **RECOMENDACION SUBMIT**.

El notebook se generó de `src/exp009_dinov2_weaklabels.py` y `src/exp010_dinov2_multitask.py`. Requiere **GPU T4**, Internet OFF y los mismos tres inputs de EXP-009. No garantiza 0.80 en Kaggle.'''),
    nbformat.v4.new_code_cell(base),
    nbformat.v4.new_code_cell(extra),
]
FOLDER.mkdir(parents=True, exist_ok=True)
nbformat.validate(nb)
nbformat.write(nb, FOLDER / NAME)

old_meta = json.loads(OLD_METADATA.read_text(encoding='utf-8'))
keep = [
    'language', 'kernel_type', 'is_private', 'enable_gpu', 'enable_tpu',
    'enable_internet', 'keywords', 'dataset_sources', 'kernel_sources',
    'competition_sources', 'model_sources', 'machine_shape',
]
metadata = {k: old_meta[k] for k in keep if k in old_meta}
metadata.update({
    'id': 'jerorojasalvarado/rsna-exp010-dinov2-multitask',
    'title': 'RSNA_EXP010_DINOv2_Multitask',
    'code_file': NAME,
    'enable_gpu': True,
    'enable_internet': False,
})
(FOLDER / 'kernel-metadata.json').write_text(
    json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
)
print('Notebook:', FOLDER / NAME)
print('Metadata:', FOLDER / 'kernel-metadata.json')
print('Celdas:', len(nb.cells))
print('No se ha ejecutado el modelo ni usado GPU.')
