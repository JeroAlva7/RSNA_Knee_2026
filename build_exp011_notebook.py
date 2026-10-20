"""Run locally from project root: python build_exp011_notebook.py"""
from pathlib import Path
import json
import nbformat
root=Path(__file__).resolve().parent
source=root/'src'/'exp011_finetune_dinov2.py'
if not source.is_file():
    raise FileNotFoundError(source)
folder=root/'notebooks'/'exp011_finetune_dinov2'
folder.mkdir(parents=True,exist_ok=True)
name='rsna-exp011-finetune-dinov2.ipynb'
nb=nbformat.v4.new_notebook()
nb.cells=[nbformat.v4.new_markdown_cell(
    '# EXP-011 — Fine-tune DINOv2 Small (original)\n\n'
    'Adjusts the final 2 transformer blocks using 4,349 report-derived weak-label studies. '
    'The 58 gold studies are reserved for the final evaluation and not used in training/early stop. '
    '**Local AUC does not guarantee a Kaggle Public or Private Score of 0.80.**\n\n'
    'GPU T4, internet OFF, 3 inputs: RSNA competition, DINOv2 Small, report-labels.'
),nbformat.v4.new_code_cell(source.read_text(encoding='utf-8'))]
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python'}}
nbformat.validate(nb)
nbformat.write(nb,folder/name)
meta={'id':'jerorojasalvarado/rsna-exp011-finetune-dinov2',
      'title':'RSNA_EXP011_FineTune_DINOv2',
      'code_file':name,'language':'python','kernel_type':'notebook',
      'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
      'keywords':['gpu'],
      'competition_sources':['rsna-knee-abnormality-detection'],
      'dataset_sources':['stevenleehans/rsna-knee-llm-report-labels'],
      'model_sources':['metaresearch/dinov2/PyTorch/small/1'],
      'kernel_sources':[],'machine_shape':'NvidiaTeslaT4'}
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
print('Created:',folder/name)
print('Cells:',len(nb.cells))
print('No model executed.')
