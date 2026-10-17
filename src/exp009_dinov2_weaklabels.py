from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

UID = "StudyInstanceUID"
SID = "SeriesInstanceUID"

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PLANES = [
    "sagittal",
    "coronal",
    "axial",
]

SERIES_PER_PLANE = 2
WINDOWS_PER_SERIES = 4
SEED = 42


def find_data():
    candidates = [
        Path(
            "/kaggle/input/competitions/"
            "rsna-knee-abnormality-detection"
        ),
        Path(
            "/kaggle/input/"
            "rsna-knee-abnormality-detection"
        ),
    ]

    for folder in candidates:
        if (
            (folder / "train.csv").exists()
            and
            (folder / "train_series").exists()
        ):
            return folder

    raise FileNotFoundError(
        "No encuentro los datos de la competencia."
    )


def find_dino():
    candidates = [
        Path(
            "/kaggle/input/dinov2/"
            "pytorch/small/1"
        ),
        Path(
            "/kaggle/input/models/metaresearch/"
            "dinov2/pytorch/small/1"
        ),
    ]

    for folder in candidates:
        if (
            folder / "config.json"
        ).exists():
            return folder

    for config in Path(
        "/kaggle/input"
    ).rglob("config.json"):

        text = str(config).lower()

        if (
            "dinov2" in text
            and "small" in text
        ):
            return config.parent

    raise FileNotFoundError(
        "Agrega Meta DINO v2 > "
        "PyTorch > small como Input."
    )


def boolish(value):
    if pd.isna(value):
        return 0

    return int(
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def normalize_plane(value):
    text = str(value).lower()

    for plane in PLANES:
        if plane in text:
            return plane

    return "other"


def choose_series(series_df):
    df = series_df.copy()

    df["_plane"] = (
        df["Anatomical_Plane"]
        .map(normalize_plane)
    )

    df["_fluid"] = (
        df["Fluid_Sensitive"]
        .map(boolish)
    )

    df["_fat"] = (
        df["Fat_Suppression"]
        .map(boolish)
    )

    df["_score"] = (
        2 * df["_fluid"]
        + df["_fat"]
    )

    selected = {}

    for study_uid, group in df.groupby(
        UID,
        sort=False,
    ):
        selected[
            str(study_uid)
        ] = {}

        for plane in PLANES:
            gp = (
                group[
                    group["_plane"]
                    == plane
                ]
                .sort_values(
                    "_score",
                    ascending=False,
                    kind="stable",
                )
            )

            chosen = []
            protocols = set()

            for _, row in gp.iterrows():

                protocol = (
                    int(row["_fluid"]),
                    int(row["_fat"]),
                )

                series_uid = str(
                    row[SID]
                )

                if (
                    protocol
                    not in protocols
                ):
                    chosen.append(
                        series_uid
                    )

                    protocols.add(
                        protocol
                    )

                if (
                    len(chosen)
                    == SERIES_PER_PLANE
                ):
                    break

            if (
                len(chosen)
                < SERIES_PER_PLANE
            ):
                for series_uid in (
                    gp[SID]
                    .astype(str)
                ):
                    if (
                        series_uid
                        not in chosen
                    ):
                        chosen.append(
                            series_uid
                        )

                    if (
                        len(chosen)
                        == SERIES_PER_PLANE
                    ):
                        break

            selected[
                str(study_uid)
            ][plane] = chosen

    return selected


def sorted_files(series_dir):
    records = []

    for order, path in enumerate(
        series_dir.glob("*.dcm")
    ):

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "ImagePositionPatient",
                    "ImageOrientationPatient",
                    "InstanceNumber",
                ],
            )

            ipp = getattr(
                ds,
                "ImagePositionPatient",
                None,
            )

            iop = getattr(
                ds,
                "ImageOrientationPatient",
                None,
            )

            if (
                ipp is not None
                and iop is not None
                and len(iop) >= 6
            ):
                row = np.asarray(
                    iop[:3],
                    float,
                )

                col = np.asarray(
                    iop[3:6],
                    float,
                )

                normal = np.cross(
                    row,
                    col,
                )

                position = np.asarray(
                    ipp[:3],
                    float,
                )

                key = float(
                    np.dot(
                        position,
                        normal,
                    )
                )

            else:
                key = float(
                    getattr(
                        ds,
                        "InstanceNumber",
                        order,
                    )
                )

        except Exception:
            key = float(order)

        records.append(
            (
                key,
                path,
            )
        )

    records.sort(
        key=lambda value: value[0]
    )

    return [
        path
        for _, path
        in records
    ]


def read_u8(path):
    ds = pydicom.dcmread(
        path,
        force=True,
    )

    array = (
        ds.pixel_array
        .astype(np.float32)
    )

    array = (
        array
        * float(
            getattr(
                ds,
                "RescaleSlope",
                1.0,
            )
        )
        + float(
            getattr(
                ds,
                "RescaleIntercept",
                0.0,
            )
        )
    )

    if (
        str(
            getattr(
                ds,
                "PhotometricInterpretation",
                "",
            )
        ).upper()
        == "MONOCHROME1"
    ):
        array = (
            array.max()
            - array
        )

    valid = np.isfinite(
        array
    )

    if not valid.any():
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    low, high = np.percentile(
        array[valid],
        [
            1,
            99,
        ],
    )

    if high <= low:
        high = low + 1

    array = np.clip(
        (
            array
            - low
        )
        / (
            high
            - low
        ),
        0,
        1,
    )

    return (
        255 * array
    ).round().astype(
        np.uint8
    )


def make_triplet(
    files,
    center,
):
    indices = [
        max(
            0,
            center - 1,
        ),
        center,
        min(
            len(files) - 1,
            center + 1,
        ),
    ]

    channels = [
        read_u8(
            files[index]
        )
        for index
        in indices
    ]

    height = min(
        image.shape[0]
        for image in channels
    )

    width = min(
        image.shape[1]
        for image in channels
    )

    rgb = np.stack(
        [
            image[
                :height,
                :width,
            ]
            for image
            in channels
        ],
        axis=-1,
    )

    return Image.fromarray(
        rgb
    )


def series_windows(
    series_dir,
):
    files = sorted_files(
        series_dir
    )

    if len(files) < 2:
        return []

    centers = np.rint(
        np.linspace(
            0.15,
            0.85,
            WINDOWS_PER_SERIES,
        )
        * (
            len(files)
            - 1
        )
    ).astype(int)

    images = []

    for center in centers:
        try:
            images.append(
                make_triplet(
                    files,
                    int(center),
                )
            )

        except Exception:
            pass

    return images


@torch.inference_mode()
def encode(
    images,
    processor,
    model,
    device,
):
    if not images:
        return np.empty(
            (
                0,
                model.config.hidden_size,
            ),
            dtype=np.float32,
        )

    features = []

    for start in range(
        0,
        len(images),
        24,
    ):
        batch = images[
            start:
            start + 24
        ]

        pixel_values = (
            processor(
                images=batch,
                return_tensors="pt",
            )["pixel_values"]
            .to(device)
        )

        if (
            device.type
            == "cuda"
        ):
            with torch.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                output = (
                    model(
                        pixel_values=
                        pixel_values
                    )
                    .last_hidden_state[
                        :,
                        0,
                    ]
                )

        else:
            output = (
                model(
                    pixel_values=
                    pixel_values
                )
                .last_hidden_state[
                    :,
                    0,
                ]
            )

        features.append(
            output
            .float()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        features
    )


def one_study(
    study_uid,
    chosen,
    dicom_root,
    processor,
    model,
    device,
):
    hidden = int(
        model.config.hidden_size
    )

    blocks = []

    for plane in PLANES:

        images = []

        for series_uid in (
            chosen
            .get(
                str(study_uid),
                {},
            )
            .get(
                plane,
                [],
            )
        ):

            series_dir = (
                dicom_root
                / str(study_uid)
                / str(series_uid)
            )

            if (
                series_dir.exists()
            ):
                images += (
                    series_windows(
                        series_dir
                    )
                )

        features = encode(
            images,
            processor,
            model,
            device,
        )

        if len(features):

            blocks.append(
                features.mean(
                    axis=0
                )
            )

            blocks.append(
                features.max(
                    axis=0
                )
            )

        else:

            blocks.append(
                np.zeros(
                    hidden,
                    np.float32,
                )
            )

            blocks.append(
                np.zeros(
                    hidden,
                    np.float32,
                )
            )

    return np.concatenate(
        blocks
    ).astype(
        np.float32
    )


def build_matrix(
    study_uids,
    chosen,
    root,
    processor,
    model,
    device,
    label,
):
    output = []
    times = []

    total = len(
        study_uids
    )

    for index, study_uid in enumerate(
        study_uids,
        start=1,
    ):

        start = time.time()

        output.append(
            one_study(
                study_uid,
                chosen,
                root,
                processor,
                model,
                device,
            )
        )

        times.append(
            time.time()
            - start
        )

        if (
            index % 10 == 0
            or index == total
        ):
            average = np.mean(
                times[
                    -min(
                        20,
                        len(times),
                    ):
                ]
            )

            print(
                f"{label}: "
                f"{index}/{total} | "
                f"{average:.2f} "
                "s/estudio"
            )

    return (
        np.vstack(
            output
        ),
        times,
    )


def candidates():
    return {

        "PCA16_C0.1":
        Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "pca",
                    PCA(
                        16,
                        whiten=True,
                        random_state=SEED,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=0.1,
                        class_weight=
                        "balanced",
                        solver=
                        "liblinear",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        "PCA24_C0.1":
        Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "pca",
                    PCA(
                        24,
                        whiten=True,
                        random_state=SEED,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=0.1,
                        class_weight=
                        "balanced",
                        solver=
                        "liblinear",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        "RAW_C0.002":
        Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=0.002,
                        class_weight=
                        "balanced",
                        solver=
                        "liblinear",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def cv_score(
    pipeline,
    X,
    gold,
):
    scores = {}

    for target in TARGETS:

        y = (
            gold[target]
            .astype(int)
            .to_numpy()
        )

        minimum_class = int(
            pd.Series(y)
            .value_counts()
            .min()
        )

        cv = StratifiedKFold(
            n_splits=min(
                5,
                minimum_class,
            ),
            shuffle=True,
            random_state=SEED,
        )

        prediction = (
            cross_val_predict(
                pipeline,
                X,
                y,
                cv=cv,
                method=
                "predict_proba",
            )[
                :,
                1,
            ]
        )

        scores[target] = (
            roc_auc_score(
                y,
                prediction,
            )
        )

    macro = float(
        np.mean(
            list(
                scores.values()
            )
        )
    )

    return (
        macro,
        scores,
    )

def normalized_name(value):
    return "".join(
        character.lower()
        for character in str(value)
        if character.isalnum()
    )


def find_weak_labels():
    candidates = [
        Path(
            "/kaggle/input/datasets/"
            "stevenleehans/"
            "rsna-knee-llm-report-labels"
        ),
        Path(
            "/kaggle/input/datasets/"
            "pilkwang/"
            "rsna-knee-llm-labels"
        ),
    ]

    for folder in candidates:
        if not folder.exists():
            continue

        csv_files = list(
            folder.glob("*.csv")
        )

        if csv_files:
            # Usar el CSV más grande.
            return max(
                csv_files,
                key=lambda p:
                p.stat().st_size,
            )

    raise FileNotFoundError(
        "No encuentro el dataset "
        "RSNA Knee LLM Report Labels."
    )


def find_uid_column(df):
    for column in df.columns:
        if (
            normalized_name(column)
            == "studyinstanceuid"
        ):
            return column

    raise KeyError(
        "No encuentro StudyInstanceUID "
        "en weak labels."
    )


def find_score_columns(df):
    mapping = {}

    for target in TARGETS:
        target_normalized = (
            normalized_name(target)
        )

        candidates = []

        for column in df.columns:
            column_normalized = (
                normalized_name(column)
            )

            if (
                target_normalized
                in column_normalized
                and "score"
                in column_normalized
            ):
                candidates.append(
                    column
                )

        # Fallback:
        # columna exactamente llamada como target.
        if not candidates:
            for column in df.columns:
                if (
                    normalized_name(
                        column
                    )
                    == target_normalized
                ):
                    candidates.append(
                        column
                    )

        if not candidates:
            raise KeyError(
                f"No encuentro score para {target}"
            )

        mapping[target] = (
            candidates[0]
        )

    return mapping


def rank01(values):
    return (
        pd.Series(
            np.asarray(values)
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy(
            dtype=float
        )
    )


def macro_auc(
    gold,
    predictions,
):
    scores = {}

    for target in TARGETS:
        scores[target] = (
            roc_auc_score(
                gold[target]
                .astype(int),
                predictions[target],
            )
        )

    return (
        float(
            np.mean(
                list(
                    scores.values()
                )
            )
        ),
        scores,
    )


def gold_oof_and_test(
    X_gold,
    X_test,
    gold,
):
    oof = {}
    test_predictions = {}

    template = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    24,
                    whiten=True,
                    random_state=SEED,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    for target in TARGETS:
        y = (
            gold[target]
            .astype(int)
            .to_numpy()
        )

        minimum_class = int(
            pd.Series(y)
            .value_counts()
            .min()
        )

        cv = StratifiedKFold(
            n_splits=min(
                5,
                minimum_class,
            ),
            shuffle=True,
            random_state=SEED,
        )

        oof[target] = (
            cross_val_predict(
                template,
                X_gold,
                y,
                cv=cv,
                method="predict_proba",
            )[:, 1]
        )

        fitted = (
            clone(template)
            .fit(
                X_gold,
                y,
            )
        )

        test_predictions[target] = (
            fitted
            .predict_proba(
                X_test
            )[:, 1]
        )

    return (
        oof,
        test_predictions,
    )

def main():
    print("=" * 70)
    print(
        "EXP-009 "
        "DINOv2 + Report Weak Labels"
    )
    print("=" * 70)

    data_dir = find_data()
    dino_dir = find_dino()
    weak_path = find_weak_labels()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Datos:", data_dir)
    print("DINO:", dino_dir)
    print("Weak labels:", weak_path)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda
            .get_device_name(0),
        )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            dino_dir,
            local_files_only=True,
        )
    )

    model = (
        AutoModel
        .from_pretrained(
            dino_dir,
            local_files_only=True,
        )
        .eval()
        .to(device)
    )

    train = pd.read_csv(
        data_dir / "train.csv"
    )

    train_series = pd.read_csv(
        data_dir / "train_series.csv"
    )

    test = pd.read_csv(
        data_dir / "test.csv"
    )

    test_series = pd.read_csv(
        data_dir / "test_series.csv"
    )

    weak = pd.read_csv(
        weak_path
    )

    weak_uid_column = (
        find_uid_column(
            weak
        )
    )

    weak = weak.rename(
        columns={
            weak_uid_column: UID
        }
    )

    weak[UID] = (
        weak[UID]
        .astype(str)
    )

    train[UID] = (
        train[UID]
        .astype(str)
    )

    train_series[UID] = (
        train_series[UID]
        .astype(str)
    )

    test[UID] = (
        test[UID]
        .astype(str)
    )

    test_series[UID] = (
        test_series[UID]
        .astype(str)
    )

    score_columns = (
        find_score_columns(
            weak
        )
    )

    print("\nWeak-label columns:")

    for target in TARGETS:
        print(
            f"{target:20s} "
            f"-> "
            f"{score_columns[target]}"
        )

    gold = (
        train.loc[
            train[
                TARGETS
            ]
            .notna()
            .all(axis=1),
            [UID] + TARGETS,
        ]
        .reset_index(
            drop=True
        )
    )

    gold_uids = set(
        gold[UID]
        .astype(str)
    )

    # =======================================================
    # Medir calidad del weak-label dataset
    # SOLO como auditoría.
    # =======================================================

    weak_gold = (
        gold[[UID] + TARGETS]
        .merge(
            weak,
            on=UID,
            how="inner",
            suffixes=(
                "_gold",
                "_weak",
            ),
        )
    )

    weak_label_auc = {}

    for target in TARGETS:
        weak_label_auc[target] = (
            roc_auc_score(
                weak_gold[target]
                .astype(int),
                pd.to_numeric(
                    weak_gold[
                        score_columns[target]
                    ],
                    errors="coerce",
                )
                .fillna(0.5),
            )
        )

    weak_label_macro = float(
        np.mean(
            list(
                weak_label_auc.values()
            )
        )
    )

    print("\n" + "=" * 70)
    print(
        "WEAK LABEL QUALITY "
        "VS 58 GOLD"
    )
    print("=" * 70)

    print(
        "Macro weak-label AUC:",
        f"{weak_label_macro:.4f}",
    )

    for target in TARGETS:
        print(
            f"{target:20s} "
            f"{weak_label_auc[target]:.4f}"
        )

    if weak_label_macro < 0.75:
        raise RuntimeError(
            "Weak labels demasiado débiles. "
            "No continuar."
        )

    # =======================================================
    # Excluir por completo los 58 GOLD
    # del weak training.
    # =======================================================

    weak_train = (
        train[[UID]]
        .merge(
            weak,
            on=UID,
            how="inner",
        )
    )

    weak_train = (
        weak_train[
            ~weak_train[UID]
            .isin(
                gold_uids
            )
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "\nGold studies:",
        len(gold),
    )

    print(
        "Weak train studies:",
        len(weak_train),
    )

    print(
        "Visible test:",
        len(test),
    )

    # =======================================================
    # DINO features
    # =======================================================

    needed_train_uids = set(
        weak_train[UID]
    ) | set(
        gold[UID]
    )

    selected_train = (
        choose_series(
            train_series[
                train_series[UID]
                .isin(
                    needed_train_uids
                )
            ]
        )
    )

    selected_test = (
        choose_series(
            test_series
        )
    )

    X_weak, weak_times = (
        build_matrix(
            weak_train[UID]
            .tolist(),
            selected_train,
            data_dir
            / "train_series",
            processor,
            model,
            device,
            "WEAK",
        )
    )

    X_gold, _ = (
        build_matrix(
            gold[UID]
            .tolist(),
            selected_train,
            data_dir
            / "train_series",
            processor,
            model,
            device,
            "GOLD",
        )
    )

    X_test, test_times = (
        build_matrix(
            test[UID]
            .tolist(),
            selected_test,
            data_dir
            / "test_series",
            processor,
            model,
            device,
            "TEST",
        )
    )

    print(
        "\nShapes:",
        X_weak.shape,
        X_gold.shape,
        X_test.shape,
    )

    if weak_times:
        train_hours = (
            np.mean(weak_times)
            * len(weak_train)
            / 3600
        )

        print(
            "Weak feature time:",
            f"{train_hours:.2f} h",
        )

    if test_times:
        projected = (
            np.mean(test_times)
            * 1322
            / 3600
        )

        print(
            "Hidden test projection:",
            f"{projected:.2f} h",
        )

    # =======================================================
    # PCA fitted ONLY on weak training.
    # Gold remains genuine holdout.
    # =======================================================

    scaler = StandardScaler()

    X_weak_scaled = (
        scaler.fit_transform(
            X_weak
        )
    )

    X_gold_scaled = (
        scaler.transform(
            X_gold
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    pca = PCA(
        n_components=128,
        whiten=True,
        svd_solver="randomized",
        random_state=SEED,
    )

    Z_weak = (
        pca.fit_transform(
            X_weak_scaled
        )
    )

    Z_gold = (
        pca.transform(
            X_gold_scaled
        )
    )

    Z_test = (
        pca.transform(
            X_test_scaled
        )
    )

    print(
        "PCA shapes:",
        Z_weak.shape,
        Z_gold.shape,
        Z_test.shape,
    )

    # =======================================================
    # Candidate weak-label model families
    # =======================================================

    family_gold = {}
    family_test = {}
    family_scores = {}

    families = [
        "RIDGE20",
        "RIDGE50",
        "LOGIT",
    ]

    for family in families:
        gold_predictions = {}
        test_predictions = {}

        for target in TARGETS:
            soft_y = (
                pd.to_numeric(
                    weak_train[
                        score_columns[
                            target
                        ]
                    ],
                    errors="coerce",
                )
                .fillna(0.5)
                .to_numpy(
                    dtype=float
                )
            )

            if family == "RIDGE20":
                fitted = Ridge(
                    alpha=20.0
                )

                fitted.fit(
                    Z_weak,
                    soft_y,
                )

                gold_pred = (
                    fitted.predict(
                        Z_gold
                    )
                )

                test_pred = (
                    fitted.predict(
                        Z_test
                    )
                )

            elif family == "RIDGE50":
                fitted = Ridge(
                    alpha=50.0
                )

                fitted.fit(
                    Z_weak,
                    soft_y,
                )

                gold_pred = (
                    fitted.predict(
                        Z_gold
                    )
                )

                test_pred = (
                    fitted.predict(
                        Z_test
                    )
                )

            else:
                confidence = (
                    2.0
                    * np.abs(
                        soft_y
                        - 0.5
                    )
                )

                keep = (
                    confidence
                    >= 0.10
                )

                hard_y = (
                    soft_y[
                        keep
                    ]
                    > 0.5
                ).astype(int)

                fitted = (
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=3000,
                        random_state=SEED,
                    )
                )

                fitted.fit(
                    Z_weak[
                        keep
                    ],
                    hard_y,
                    sample_weight=(
                        confidence[
                            keep
                        ]
                    ),
                )

                gold_pred = (
                    fitted
                    .predict_proba(
                        Z_gold
                    )[:, 1]
                )

                test_pred = (
                    fitted
                    .predict_proba(
                        Z_test
                    )[:, 1]
                )

            gold_predictions[
                target
            ] = gold_pred

            test_predictions[
                target
            ] = test_pred

        macro, scores = (
            macro_auc(
                gold,
                gold_predictions,
            )
        )

        family_gold[
            family
        ] = gold_predictions

        family_test[
            family
        ] = test_predictions

        family_scores[
            family
        ] = (
            macro,
            scores,
        )

        print(
            f"\n{family} "
            f"Gold Macro AUC: "
            f"{macro:.4f}"
        )

    best_family = max(
        family_scores,
        key=lambda name:
        family_scores[name][0],
    )

    print(
        "\nBEST WEAK FAMILY:",
        best_family,
    )

    print(
        "Weak model Gold AUC:",
        f"{family_scores[best_family][0]:.4f}",
    )

    # =======================================================
    # EXP-007 gold-only arm.
    # Gold OOF avoids leakage.
    # =======================================================

    (
        gold_oof,
        gold_test_predictions,
    ) = gold_oof_and_test(
        X_gold,
        X_test,
        gold,
    )

    gold_macro, _ = macro_auc(
        gold,
        gold_oof,
    )

    print(
        "\nEXP-007 OOF arm:",
        f"{gold_macro:.4f}",
    )

    # =======================================================
    # Global rank blend:
    # weak holdout predictions
    # +
    # EXP007 gold OOF predictions.
    #
    # ONE global weight only.
    # No target-by-target overfitting.
    # =======================================================

    weak_gold = (
        family_gold[
            best_family
        ]
    )

    weak_test = (
        family_test[
            best_family
        ]
    )

    weights = [
        0.50,
        0.65,
        0.80,
        0.90,
        1.00,
    ]

    blend_results = {}

    for weak_weight in weights:
        blended_gold = {}

        for target in TARGETS:
            blended_gold[
                target
            ] = (
                weak_weight
                * rank01(
                    weak_gold[
                        target
                    ]
                )
                +
                (
                    1.0
                    - weak_weight
                )
                * rank01(
                    gold_oof[
                        target
                    ]
                )
            )

        macro, scores = (
            macro_auc(
                gold,
                blended_gold,
            )
        )

        blend_results[
            weak_weight
        ] = (
            macro,
            scores,
        )

        print(
            "Blend weak weight",
            weak_weight,
            "Macro:",
            f"{macro:.4f}",
        )

    best_weight = max(
        blend_results,
        key=lambda weight:
        blend_results[
            weight
        ][0],
    )

    final_macro = (
        blend_results[
            best_weight
        ][0]
    )

    final_scores = (
        blend_results[
            best_weight
        ][1]
    )

    print("\n" + "=" * 70)

    print(
        "BEST WEAK FAMILY:",
        best_family,
    )

    print(
        "BEST WEAK WEIGHT:",
        best_weight,
    )

    print(
        "FINAL GOLD MACRO AUC:",
        f"{final_macro:.4f}",
    )

    print("=" * 70)

    for target in TARGETS:
        print(
            f"{target:20s} "
            f"{final_scores[target]:.4f}"
        )

    # =======================================================
    # Final hidden-test prediction
    # =======================================================

    submission = (
        test[
            [UID]
        ]
        .copy()
    )

    for target in TARGETS:
        submission[target] = (
            best_weight
            * rank01(
                weak_test[
                    target
                ]
            )
            +
            (
                1.0
                - best_weight
            )
            * rank01(
                gold_test_predictions[
                    target
                ]
            )
        )

    assert (
        submission[
            TARGETS
        ]
        .notna()
        .all()
        .all()
    )

    assert (
        submission[UID]
        .is_unique
    )

    submission.to_csv(
        "/kaggle/working/"
        "submission.csv",
        index=False,
    )

    print(
        "\nSubmission creada"
    )

    print(
        submission.head()
    )


if __name__ == "__main__":
    main()