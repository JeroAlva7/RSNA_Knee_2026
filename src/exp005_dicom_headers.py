from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


UID = "StudyInstanceUID"
SERIES_UID = "SeriesInstanceUID"

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


NUMERIC_TAGS = [
    "MagneticFieldStrength",
    "SliceThickness",
    "SpacingBetweenSlices",
    "RepetitionTime",
    "EchoTime",
    "EchoTrainLength",
    "FlipAngle",
    "Rows",
    "Columns",
    "NumberOfAverages",
    "ImagingFrequency",
    "PixelBandwidth",
    "PercentPhaseFieldOfView",
    "SAR",
    "ReconstructionDiameter",
]


CATEGORICAL_TAGS = [
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "SequenceName",
    "InPlanePhaseEncodingDirection",
    "PatientPosition",
    "PatientSex",
    "ImageType",
]


def find_data_dir():
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
            and (folder / "train_series").exists()
        ):
            return folder

    raise FileNotFoundError(
        "EXP-005 necesita ejecutarse en Kaggle "
        "porque requiere los archivos DICOM."
    )


def safe_float(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def safe_string(value):
    if value is None:
        return "__MISSING__"

    text = str(value).strip()

    if not text:
        return "__MISSING__"

    return text


def patient_age_to_years(value):
    if value is None:
        return np.nan

    text = str(value).strip().upper()

    if len(text) < 2:
        return np.nan

    try:
        number = float(text[:-1])
        unit = text[-1]

        if unit == "Y":
            return number
        if unit == "M":
            return number / 12.0
        if unit == "W":
            return number / 52.0
        if unit == "D":
            return number / 365.25

    except Exception:
        pass

    return np.nan


def extract_series_headers(series_df, dicom_root):
    records = []

    total = len(series_df)

    for i, row in enumerate(
        series_df.itertuples(index=False),
        start=1,
    ):
        study_uid = getattr(
            row,
            UID,
        )

        series_uid = getattr(
            row,
            SERIES_UID,
        )

        series_dir = (
            dicom_root
            / str(study_uid)
            / str(series_uid)
        )

        files = list(
            series_dir.glob("*.dcm")
        )

        record = {
            UID: study_uid,
            SERIES_UID: series_uid,
            "slice_count": len(files),
            "Fluid_Sensitive": getattr(
                row,
                "Fluid_Sensitive",
            ),
            "Fat_Suppression": getattr(
                row,
                "Fat_Suppression",
            ),
            "Anatomical_Plane": safe_string(
                getattr(
                    row,
                    "Anatomical_Plane",
                )
            ),
        }

        for tag in NUMERIC_TAGS:
            record[tag] = np.nan

        for tag in CATEGORICAL_TAGS:
            record[tag] = "__MISSING__"

        record["PixelSpacing0"] = np.nan
        record["PixelSpacing1"] = np.nan
        record["PatientAgeYears"] = np.nan

        if files:
            # Header central de la serie.
            # No se cargan pixels.
            dicom_path = files[
                len(files) // 2
            ]

            try:
                ds = pydicom.dcmread(
                    dicom_path,
                    stop_before_pixels=True,
                    force=True,
                )

                for tag in NUMERIC_TAGS:
                    record[tag] = safe_float(
                        getattr(
                            ds,
                            tag,
                            None,
                        )
                    )

                for tag in CATEGORICAL_TAGS:
                    record[tag] = safe_string(
                        getattr(
                            ds,
                            tag,
                            None,
                        )
                    )

                spacing = getattr(
                    ds,
                    "PixelSpacing",
                    None,
                )

                if spacing is not None:
                    try:
                        if len(spacing) >= 1:
                            record[
                                "PixelSpacing0"
                            ] = safe_float(
                                spacing[0]
                            )

                        if len(spacing) >= 2:
                            record[
                                "PixelSpacing1"
                            ] = safe_float(
                                spacing[1]
                            )

                    except Exception:
                        pass

                record[
                    "PatientAgeYears"
                ] = patient_age_to_years(
                    getattr(
                        ds,
                        "PatientAge",
                        None,
                    )
                )

            except Exception:
                pass

        records.append(
            record
        )

        if (
            i % 500 == 0
            or i == total
        ):
            print(
                f"Headers procesados: "
                f"{i}/{total}"
            )

    return pd.DataFrame(
        records
    )


def aggregate_features(
    train_series,
    test_series,
):
    numeric_columns = [
        "slice_count",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "PixelSpacing0",
        "PixelSpacing1",
        "PatientAgeYears",
    ] + NUMERIC_TAGS

    categorical_columns = [
        "Anatomical_Plane",
    ] + CATEGORICAL_TAGS

    def numeric_features(df):
        data = df[
            [UID] + numeric_columns
        ].copy()

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        grouped = (
            data.groupby(UID)[
                numeric_columns
            ]
            .agg(
                [
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
            )
        )

        grouped.columns = [
            f"{a}_{b}"
            for a, b
            in grouped.columns
        ]

        return grouped

    train_features = numeric_features(
        train_series
    )

    test_features = numeric_features(
        test_series
    )

    train_n_series = (
        train_series
        .groupby(UID)[SERIES_UID]
        .nunique()
        .rename("n_series")
    )

    test_n_series = (
        test_series
        .groupby(UID)[SERIES_UID]
        .nunique()
        .rename("n_series")
    )

    train_features = (
        train_features
        .join(train_n_series)
    )

    test_features = (
        test_features
        .join(test_n_series)
    )

    # Categorical features.
    # El vocabulario se define únicamente con TRAIN.
    for column in categorical_columns:

        train_values = (
            train_series[column]
            .fillna("__MISSING__")
            .astype(str)
        )

        test_values = (
            test_series[column]
            .fillna("__MISSING__")
            .astype(str)
        )

        top_categories = (
            train_values
            .value_counts()
            .head(12)
            .index
        )

        train_mapped = (
            train_values
            .where(
                train_values.isin(
                    top_categories
                ),
                "__OTHER__",
            )
        )

        test_mapped = (
            test_values
            .where(
                test_values.isin(
                    top_categories
                ),
                "__OTHER__",
            )
        )

        train_dummy = pd.get_dummies(
            train_mapped,
            prefix=column,
            dtype=float,
        )

        test_dummy = pd.get_dummies(
            test_mapped,
            prefix=column,
            dtype=float,
        )

        test_dummy = test_dummy.reindex(
            columns=train_dummy.columns,
            fill_value=0.0,
        )

        train_dummy[UID] = (
            train_series[UID]
            .to_numpy()
        )

        test_dummy[UID] = (
            test_series[UID]
            .to_numpy()
        )

        train_grouped = (
            train_dummy
            .groupby(UID)
            .sum()
        )

        test_grouped = (
            test_dummy
            .groupby(UID)
            .sum()
        )

        # También proporción de series
        # con cada característica.
        for dummy_column in list(
            train_grouped.columns
        ):
            train_grouped[
                f"{dummy_column}_prop"
            ] = (
                train_grouped[
                    dummy_column
                ]
                / train_n_series
            )

            test_grouped[
                f"{dummy_column}_prop"
            ] = (
                test_grouped[
                    dummy_column
                ]
                / test_n_series
            )

        train_features = (
            train_features
            .join(
                train_grouped,
                how="left",
            )
        )

        test_features = (
            test_features
            .join(
                test_grouped,
                how="left",
            )
        )

    train_features, test_features = (
        train_features.align(
            test_features,
            join="outer",
            axis=1,
        )
    )

    return (
        train_features.reset_index(),
        test_features.reset_index(),
    )


def make_model():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.08,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    data_dir = find_data_dir()

    print("=" * 70)
    print("EXP-005")
    print("Full DICOM Header Metadata")
    print("=" * 70)

    train = pd.read_csv(
        data_dir / "train.csv"
    )

    test = pd.read_csv(
        data_dir / "test.csv"
    )

    train_series_csv = pd.read_csv(
        data_dir / "train_series.csv"
    )

    test_series_csv = pd.read_csv(
        data_dir / "test_series.csv"
    )

    print("\nExtrayendo headers TRAIN...")

    train_series = extract_series_headers(
        train_series_csv,
        data_dir / "train_series",
    )

    print("\nExtrayendo headers TEST...")

    test_series = extract_series_headers(
        test_series_csv,
        data_dir / "test_series",
    )

    print("\nAgregando features...")

    (
        train_features,
        test_features,
    ) = aggregate_features(
        train_series,
        test_series,
    )

    train_model = train[
        [UID] + TARGETS
    ].merge(
        train_features,
        on=UID,
        how="left",
    )

    test_model = test[
        [UID]
    ].merge(
        test_features,
        on=UID,
        how="left",
    )

    feature_columns = [
        column
        for column
        in train_features.columns
        if column != UID
    ]

    # Quitar columnas totalmente vacías
    # en los 58 estudios gold.
    gold_any = (
        train_model[TARGETS]
        .notna()
        .any(axis=1)
    )

    feature_columns = [
        column
        for column in feature_columns
        if (
            train_model.loc[
                gold_any,
                column,
            ]
            .notna()
            .any()
        )
    ]

    print(
        "\nNúmero de features:",
        len(feature_columns),
    )

    X_test = (
        test_model[
            feature_columns
        ]
        .astype(float)
    )

    submission = (
        test[[UID]]
        .copy()
    )

    auc_scores = {}

    for target in TARGETS:
        mask = (
            train_model[target]
            .notna()
        )

        X = (
            train_model.loc[
                mask,
                feature_columns,
            ]
            .astype(float)
        )

        y = (
            train_model.loc[
                mask,
                target,
            ]
            .astype(int)
        )

        print(
            f"\n{target}: "
            f"n={len(y)}, "
            f"pos={int(y.sum())}"
        )

        min_class = int(
            y.value_counts()
            .min()
        )

        n_splits = min(
            5,
            min_class,
        )

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

        model = make_model()

        oof = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        auc = roc_auc_score(
            y,
            oof,
        )

        auc_scores[target] = auc

        print(
            f"OOF AUC: {auc:.4f}"
        )

        model.fit(
            X,
            y,
        )

        submission[target] = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    macro_auc = float(
        np.mean(
            list(
                auc_scores.values()
            )
        )
    )

    print("\n" + "=" * 70)

    print(
        "Macro DICOM Header OOF AUC:",
        f"{macro_auc:.4f}",
    )

    print("=" * 70)

    assert (
        list(submission.columns)
        == [UID] + TARGETS
    )

    assert (
        submission[TARGETS]
        .notna()
        .all()
        .all()
    )

    submission[TARGETS] = (
        submission[TARGETS]
        .clip(
            0.0,
            1.0,
        )
    )

    output_path = Path(
        "/kaggle/working/submission.csv"
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nArchivo creado:"
    )

    print(
        output_path
    )

    print(
        submission.head()
    )


if __name__ == "__main__":
    main()