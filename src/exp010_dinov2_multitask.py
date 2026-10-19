"""EXP-010: mejora original, economica y verificable de EXP-009.

Se ejecuta DESPUES de cargar las funciones del codigo EXP-009, SIN llamar
EXP-009.main(). No usa reportes de test ni etiquetas gold para entrenar
el brazo weak o seleccionar epocas.

Genera: /kaggle/working/submission.csv, exp010_diagnostics.json,
exp010_gold_scores.csv. No garantiza alcanzar 0.80 Public AUC.
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Variables, funciones y librerias reutilizadas desde EXP-009:
# UID, SID, TARGETS, PLANES, SEED, find_data, find_dino,
# find_weak_labels, find_uid_column, find_score_columns, choose_series,
# series_windows, encode, build_matrix, gold_oof_and_test, rank01,
# macro_auc, AutoImageProcessor, AutoModel.

OUTDIR = Path('/kaggle/working')
N_EPOCHS = 75
MLP_SEEDS = (17, 42)
BASELINE_WEAK_WEIGHT = 0.90  # Combinacion validada en EXP-009
BLEND_WEIGHTS = (0.25, 0.50, 0.75)
# Predefinimos un filtro conservador antes de observar resultados:
MIN_GOLD_MACRO = 0.790
MIN_GAIN_OVER_BASELINE = 0.035
MAX_FEATURE_HOURS = 6.4   # margen para procesamiento posterior dentro de 9h


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_study_exp010(study_uid, chosen, dicom_root, processor, model, device):
    """Mismo recorrido/encoder EXP009; mayor estadistica sin nuevos forward-passes.

    Primer bloque de 6*hidden es IDENTICO al EXP009 (mean,max por plano).
    Segundo bloque agrega std y mediana por plano, sin costo GPU adicional.
    """
    hidden = int(model.config.hidden_size)
    base, extras = [], []
    for plane in PLANES:
        images = []
        for series_uid in chosen.get(str(study_uid), {}).get(plane, []):
            folder = dicom_root / str(study_uid) / str(series_uid)
            if folder.exists():
                images.extend(series_windows(folder))
        features = encode(images, processor, model, device)
        if len(features):
            # Orden exactamente igual al EXP-009.
            base.extend([features.mean(axis=0), features.max(axis=0)])
            extras.extend([features.std(axis=0), np.median(features, axis=0)])
        else:
            zero = np.zeros(hidden, np.float32)
            base.extend([zero, zero.copy()])
            extras.extend([zero.copy(), zero.copy()])
    return np.concatenate(base + extras).astype(np.float32)


class MultiTaskHead(nn.Module):
    """No modifica ni copia pesos de otro competidor; aprende sobre DINOv2."""
    def __init__(self, n_features, n_targets):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 384),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, n_targets),
        )

    def forward(self, X):
        return self.layers(X)


def train_multitask_weak(Z, Y, Z_gold, Z_test, seed, device):
    """Selecciona epoch SOLO en 15% de weak reports; nunca en gold."""
    set_seed(seed)
    all_indices = np.arange(len(Z))
    tr_idx, va_idx = train_test_split(
        all_indices, test_size=0.15, random_state=seed, shuffle=True)

    X_tensor = torch.tensor(Z, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    net = MultiTaskHead(Z.shape[1], Y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.00065, weight_decay=0.035)
    criterion = nn.BCEWithLogitsLoss()

    X_val = X_tensor[va_idx].to(device)
    Y_val = Y_tensor[va_idx].to(device)
    best_state = None
    best_loss = float('inf')
    best_epoch = 0
    patience = 14

    for epoch in range(1, N_EPOCHS + 1):
        net.train()
        rng = np.random.default_rng(seed * 1000 + epoch)
        perm = rng.permutation(tr_idx)
        for start in range(0, len(perm), 128):
            batch_idx = perm[start:start + 128]
            X_batch = X_tensor[batch_idx].to(device)
            Y_batch = Y_tensor[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = float(criterion(net(X_val), Y_val).item())
        if val_loss < best_loss - 0.00004:
            best_state = copy.deepcopy(net.state_dict())
            best_loss = val_loss
            best_epoch = epoch
        if (epoch % 10 == 0 or epoch == 1):
            print(f'EXP010 seed={seed}, epoch={epoch}, weak-val BCE={val_loss:.5f}', flush=True)
        if epoch - best_epoch >= patience:
            break

    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), 512):
                tensor = torch.tensor(X[start:start + 512], dtype=torch.float32, device=device)
                outputs.append(torch.sigmoid(net(tensor)).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    print(f'EXP010 seed={seed}: mejor epoch={best_epoch}, weak-val BCE={best_loss:.5f}', flush=True)
    return predict(Z_gold), predict(Z_test), best_loss, best_epoch


def soft_matrix(weak_train, score_columns):
    columns = []
    for target in TARGETS:
        values = pd.to_numeric(weak_train[score_columns[target]], errors='coerce').to_numpy(float)
        if np.isfinite(values).sum() < 100:
            raise ValueError(f'Sin etiquetas weak validas para {target}')
        midpoint = float(np.nanmedian(values))
        values = np.where(np.isfinite(values), values, midpoint)
        if float(np.min(values)) < 0 or float(np.max(values)) > 1:
            raise ValueError(f'Etiquetas weak fuera de [0,1]: {target}')
        columns.append(values)
    return np.column_stack(columns).astype(np.float32)


def baseline_exp009(Z_weak, Z_gold, Z_test, gold, X_gold_base, X_test_base, Y):
    """Reconstruye exactamente la configuracion publica de EXP-009."""
    weak_gold = {}
    weak_test = {}
    for col, target in enumerate(TARGETS):
        soft = Y[:, col]
        confidence = 2.0 * np.abs(soft - 0.5)
        keep = confidence >= 0.10
        labels = (soft[keep] > 0.5).astype(int)
        if len(np.unique(labels)) < 2:
            raise ValueError(f'Solo una clase weak: {target}')
        clf = LogisticRegression(
            C=0.1, class_weight='balanced', solver='liblinear',
            max_iter=3000, random_state=SEED,
        )
        clf.fit(Z_weak[keep], labels, sample_weight=confidence[keep])
        weak_gold[target] = clf.predict_proba(Z_gold)[:, 1]
        weak_test[target] = clf.predict_proba(Z_test)[:, 1]

    gold_oof, gold_test = gold_oof_and_test(X_gold_base, X_test_base, gold)
    baseline_gold = {}
    baseline_test = {}
    for target in TARGETS:
        baseline_gold[target] = (
            BASELINE_WEAK_WEIGHT * rank01(weak_gold[target])
            + (1 - BASELINE_WEAK_WEIGHT) * rank01(gold_oof[target])
        )
        baseline_test[target] = (
            BASELINE_WEAK_WEIGHT * rank01(weak_test[target])
            + (1 - BASELINE_WEAK_WEIGHT) * rank01(gold_test[target])
        )
    return baseline_gold, baseline_test


def validate_and_submit(gold, test, base_gold, base_test, mlp_gold, mlp_test, diagnostics):
    base_auc, base_scores = macro_auc(gold, base_gold)
    print('\nEXP-009 BASELINE GOLD MACRO AUC:', f'{base_auc:.4f}', flush=True)
    # Este chequeo evita que un cambio silencioso rompa la reproduccion de EXP009.
    if abs(base_auc - 0.7461) > 0.012:
        raise RuntimeError(
            f'Baseline EXP009 alterado: {base_auc:.4f} vs 0.7461. '
            'NO enviar. Revisar pixel pipeline y labels.'
        )

    if not all(np.isfinite(v).all() for v in mlp_gold.values()):
        raise RuntimeError('Predicciones MLP no finitas')

    candidate_scores = {}
    for w in BLEND_WEIGHTS:
        combined_gold = {
            t: (1.0 - w) * base_gold[t] + w * rank01(mlp_gold[t])
            for t in TARGETS
        }
        candidate_scores[w] = macro_auc(gold, combined_gold)
        print(f'EXP010 BLEND MLP={w:.2f} GOLD MACRO AUC: {candidate_scores[w][0]:.4f}', flush=True)

    winner = max(candidate_scores, key=lambda w: candidate_scores[w][0])
    improved_auc, improved_targets = candidate_scores[winner]
    gate = improved_auc >= MIN_GOLD_MACRO and improved_auc - base_auc >= MIN_GAIN_OVER_BASELINE

    diagnostics.update({
        'baseline_gold_auc': float(base_auc),
        'candidate_gold_auc': float(improved_auc),
        'candidate_mlp_weight': float(winner),
        'candidate_gold_gain': float(improved_auc - base_auc),
        'min_gold_macro_to_submit': MIN_GOLD_MACRO,
        'min_gain_to_submit': MIN_GAIN_OVER_BASELINE,
        'recommended_to_submit': bool(gate),
    })

    print('\n' + '=' * 70)
    print('EXP010 GOLD VALIDATION FINAL')
    print('EXP009 GOLD BASELINE:', f'{base_auc:.4f}')
    print('EXP010 BEST GOLD:', f'{improved_auc:.4f}')
    print('BEST MLP WEIGHT:', winner)
    print('RECOMENDACION SUBMIT:', 'SI' if gate else 'NO')
    print('Ningun score local garantiza 0.800 publico.')
    print('=' * 70)

    score_rows = []
    for t in TARGETS:
        score_rows.append({
            'target': t,
            'exp009_gold_auc': float(base_scores[t]),
            'exp010_gold_auc': float(improved_targets[t]),
            'delta': float(improved_targets[t] - base_scores[t]),
        })
    pd.DataFrame(score_rows).to_csv(OUTDIR / 'exp010_gold_scores.csv', index=False)

    result = test[[UID]].copy()
    for t in TARGETS:
        if gate:
            result[t] = (1 - winner) * base_test[t] + winner * rank01(mlp_test[t])
        else:
            # Fallback para que incluso una version de prueba tenga CSV valido.
            # NO ENVIAR si el gate indica NO; ya tienes EXP009=0.762.
            result[t] = base_test[t]

    assert len(result) == len(test), 'Faltan estudios test'
    assert result[UID].is_unique
    assert result[TARGETS].notna().all().all()
    assert np.isfinite(result[TARGETS].to_numpy(float)).all()
    assert (result[TARGETS].to_numpy(float) >= 0).all()
    assert (result[TARGETS].to_numpy(float) <= 1).all()
    result.to_csv(OUTDIR / 'submission.csv', index=False)
    (OUTDIR / 'exp010_diagnostics.json').write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('Submission guardada:', OUTDIR / 'submission.csv', 'Filas:', len(result))
    return gate


def main010():
    start = time.time()
    set_seed(SEED)
    print('=' * 70)
    print('EXP-010 | DINOv2 features enriquecidos + MLP multietiqueta')
    print('Baseline protegido: EXP-009 0.762 Public')
    print('=' * 70)
    if not torch.cuda.is_available():
        raise RuntimeError('Activa GPU T4 en Kaggle antes de ejecutar EXP010.')

    data_dir = find_data()
    dino_dir = find_dino()
    weak_path = find_weak_labels()
    device = torch.device('cuda')
    print('Dataset:', data_dir, 'DINO:', dino_dir, 'Labels:', weak_path)
    processor = AutoImageProcessor.from_pretrained(dino_dir, local_files_only=True)
    model = AutoModel.from_pretrained(dino_dir, local_files_only=True).eval().to(device)

    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    train_series = pd.read_csv(data_dir / 'train_series.csv')
    test_series = pd.read_csv(data_dir / 'test_series.csv')
    weak = pd.read_csv(weak_path)
    weak = weak.rename(columns={find_uid_column(weak): UID})
    for d in [train, test, train_series, test_series, weak]:
        d[UID] = d[UID].astype(str)
    gold = train.loc[train[TARGETS].notna().all(axis=1), [UID] + TARGETS].reset_index(drop=True)
    gold_uids = set(gold[UID])
    weak_train = train[[UID]].merge(weak, on=UID, how='inner')
    weak_train = weak_train.loc[~weak_train[UID].isin(gold_uids)].reset_index(drop=True)
    score_columns = find_score_columns(weak)
    Y = soft_matrix(weak_train, score_columns)
    print('GOLD:', len(gold), 'WEAK:', len(weak_train), 'TEST:', len(test), flush=True)
    if len(gold) != 58 or len(weak_train) < 4000:
        raise RuntimeError('Dataset TRAIN inesperado; no iniciar procesamiento largo')

    selected_train = choose_series(train_series)
    selected_test = choose_series(test_series)
    # build_matrix usa la funcion global one_study: la sustituimos temporalmente
    # para NO ejecutar DINO dos veces por estudio.
    global one_study
    old_one_study = one_study
    one_study = one_study_exp010
    try:
        Xw, times_w = build_matrix(
            weak_train[UID].tolist(), selected_train, data_dir / 'train_series',
            processor, model, device, 'WEAK010'
        )
        Xg, _ = build_matrix(
            gold[UID].tolist(), selected_train, data_dir / 'train_series',
            processor, model, device, 'GOLD010'
        )
        Xt, times_t = build_matrix(
            test[UID].tolist(), selected_test, data_dir / 'test_series',
            processor, model, device, 'TEST010'
        )
    finally:
        one_study = old_one_study

    # Liberar GPU usada para extraccion; luego MLP usa GPU con poca memoria.
    del model
    torch.cuda.empty_cache()
    print('EXTENDED SHAPES:', Xw.shape, Xg.shape, Xt.shape)
    base_dim = len(PLANES) * 2 * 384  # DINOv2-Small hidden=384
    if Xw.shape[1] != base_dim * 2:
        raise RuntimeError(f'Tamaño de DINOv2 inesperado: {Xw.shape[1]}')
    estimated_h = np.mean(times_w) * len(weak_train) / 3600 + np.mean(times_t) * len(test) / 3600
    hidden_h = np.mean(times_w) * len(weak_train) / 3600 + np.mean(times_t) * 1322 / 3600
    print(f'FEATURES draft={estimated_h:.2f} h; hidden approx={hidden_h:.2f} h')
    if hidden_h > MAX_FEATURE_HOURS:
        raise RuntimeError('Tiempo estimado de features muy largo para límite 9h. No enviar.')

    # --- Reproducir exactamente la representacion de EXP009.
    Xw_base, Xg_base, Xt_base = Xw[:, :base_dim], Xg[:, :base_dim], Xt[:, :base_dim]
    scaler = StandardScaler()
    Zw = scaler.fit_transform(Xw_base)
    Zg = scaler.transform(Xg_base)
    Zt = scaler.transform(Xt_base)
    pca_base = PCA(n_components=128, whiten=True, svd_solver='randomized', random_state=SEED)
    Zw_base = pca_base.fit_transform(Zw)
    Zg_base, Zt_base = pca_base.transform(Zg), pca_base.transform(Zt)

    base_gold, base_test = baseline_exp009(
        Zw_base, Zg_base, Zt_base, gold, Xg_base, Xt_base, Y
    )

    # --- Nuevo brazo: medianas/desviaciones de cada plano + red multi-output.
    ext_scaler = StandardScaler()
    Zw_ext = ext_scaler.fit_transform(Xw)
    Zg_ext = ext_scaler.transform(Xg)
    Zt_ext = ext_scaler.transform(Xt)
    pca_ext = PCA(n_components=256, whiten=True, svd_solver='randomized', random_state=SEED)
    Zw_ext = pca_ext.fit_transform(Zw_ext).astype(np.float32)
    Zg_ext = pca_ext.transform(Zg_ext).astype(np.float32)
    Zt_ext = pca_ext.transform(Zt_ext).astype(np.float32)

    predictions_g = []
    predictions_t = []
    training_log = []
    for seed in MLP_SEEDS:
        out_g, out_t, val_bce, best_epoch = train_multitask_weak(
            Zw_ext, Y, Zg_ext, Zt_ext, seed, device
        )
        predictions_g.append(out_g)
        predictions_t.append(out_t)
        training_log.append({'seed': seed, 'weak_val_bce': val_bce, 'best_epoch': best_epoch})
    pg = np.mean(predictions_g, axis=0)
    pt = np.mean(predictions_t, axis=0)
    mlp_gold = {t: pg[:, i] for i, t in enumerate(TARGETS)}
    mlp_test = {t: pt[:, i] for i, t in enumerate(TARGETS)}
    diagnostics = {
        'model': 'EXP010 original multitask-MLP on DINOv2 stats',
        'weak_train_studies': len(weak_train), 'gold_studies': len(gold),
        'test_studies': len(test), 'weak_labels': weak_path.name,
        'feature_hours_estimate_hidden': float(hidden_h),
        'multitask_weak_validation': training_log,
    }
    gate = validate_and_submit(gold, test, base_gold, base_test, mlp_gold, mlp_test, diagnostics)
    print(f'EXP010 COMPLETO en {(time.time()-start)/3600:.2f} horas', flush=True)
    print('SI SUBMIT:', bool(gate), flush=True)


if __name__ == '__main__':
    main010()
