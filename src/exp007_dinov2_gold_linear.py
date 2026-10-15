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
from sklearn.linear_model import LogisticRegression
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


def main():

    print(
        "=" * 70
    )

    print(
        "EXP-007 "
        "DINOv2 2.5D MRI"
    )

    print(
        "=" * 70
    )

    data_dir = find_data()
    dino_dir = find_dino()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Datos:",
        data_dir,
    )

    print(
        "DINO:",
        dino_dir,
    )

    print(
        "Device:",
        device,
    )

    if (
        device.type
        == "cuda"
    ):
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
        data_dir
        / "train.csv"
    )

    train_series = (
        pd.read_csv(
            data_dir
            / "train_series.csv"
        )
    )

    test = pd.read_csv(
        data_dir
        / "test.csv"
    )

    test_series = (
        pd.read_csv(
            data_dir
            / "test_series.csv"
        )
    )

    gold = (
        train.loc[
            train[
                TARGETS
            ]
            .notna()
            .all(axis=1),
            [
                UID
            ]
            + TARGETS,
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "Gold:",
        len(gold),
    )

    print(
        "Visible test:",
        len(test),
    )

    train_selected = (
        choose_series(
            train_series[
                train_series[
                    UID
                ].isin(
                    gold[
                        UID
                    ]
                )
            ]
        )
    )

    test_selected = (
        choose_series(
            test_series
        )
    )

    X_gold, _ = (
        build_matrix(
            gold[
                UID
            ]
            .astype(str)
            .tolist(),
            train_selected,
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
            test[
                UID
            ]
            .astype(str)
            .tolist(),
            test_selected,
            data_dir
            / "test_series",
            processor,
            model,
            device,
            "TEST",
        )
    )

    print(
        "Shapes:",
        X_gold.shape,
        X_test.shape,
    )

    if test_times:

        projected = (
            np.mean(
                test_times
            )
            * 1322
            / 3600
        )

        print(
            "Proyección "
            "1322 estudios:",
            f"{projected:.2f} h",
        )

    candidate_models = (
        candidates()
    )

    results = {}

    for (
        name,
        pipeline,
    ) in (
        candidate_models.items()
    ):

        macro, scores = (
            cv_score(
                pipeline,
                X_gold,
                gold,
            )
        )

        results[name] = (
            macro,
            scores,
        )

        print(
            f"{name}: "
            f"Macro OOF "
            f"= {macro:.4f}"
        )

    best = max(
        results,
        key=lambda name:
        results[name][0],
    )

    macro, scores = (
        results[best]
    )

    print(
        "\nMEJOR:",
        best,
    )

    print(
        "Macro OOF:",
        f"{macro:.4f}",
    )

    for target in TARGETS:
        print(
            f"{target:20s} "
            f"{scores[target]:.4f}"
        )

    submission = (
        test[
            [UID]
        ]
        .copy()
    )

    template = (
        candidate_models[
            best
        ]
    )

    for target in TARGETS:

        y = (
            gold[target]
            .astype(int)
            .to_numpy()
        )

        fitted = (
            clone(template)
            .fit(
                X_gold,
                y,
            )
        )

        submission[target] = (
            fitted
            .predict_proba(
                X_test
            )[
                :,
                1,
            ]
        )

    submission[
        TARGETS
    ] = (
        submission[
            TARGETS
        ]
        .clip(
            0.001,
            0.999,
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
        submission[
            UID
        ]
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