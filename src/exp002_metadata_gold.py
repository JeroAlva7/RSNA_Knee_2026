from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


UID = "StudyInstanceUID"

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


def find_data_dir():
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("data"),
    ]

    for folder in candidates:
        required = [
            folder / "train.csv",
            folder / "train_series.csv",
            folder / "test.csv",
            folder / "test_series.csv",
        ]

        if all(path.exists() for path in required):
            return folder

    raise FileNotFoundError(
        "No se encontraron los CSV de la competencia."
    )


def clean_category(series):
    return (
        series.astype("string")
        .fillna("missing")
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )


def build_metadata_features(series_df):
    df = series_df.copy()

    df["plane"] = clean_category(df["Anatomical_Plane"])
    df["fluid"] = clean_category(df["Fluid_Sensitive"])
    df["fat"] = clean_category(df["Fat_Suppression"])

    df["plane_fluid"] = df["plane"] + "__" + df["fluid"]
    df["plane_fat"] = df["plane"] + "__" + df["fat"]

    df["protocol"] = (
        df["plane"]
        + "__"
        + df["fluid"]
        + "__"
        + df["fat"]
    )

    features = (
        df.groupby(UID)["SeriesInstanceUID"]
        .nunique()
        .rename("n_series")
        .to_frame()
    )

    categorical_cols = [
        "plane",
        "fluid",
        "fat",
        "plane_fluid",
        "plane_fat",
        "protocol",
    ]

    for col in categorical_cols:
        dummies = pd.get_dummies(
            df[[UID, col]],
            columns=[col],
            prefix=col,
            dtype=float,
        )

        grouped = dummies.groupby(UID).sum()

        features = features.join(
            grouped,
            how="left",
        )

    count_columns = [
        col
        for col in features.columns
        if col != "n_series"
    ]

    denominator = features["n_series"].clip(lower=1)

    for col in count_columns:
        features[f"{col}_prop"] = (
            features[col] / denominator
        )

    return features.reset_index()


def make_model():
    return Pipeline(
        steps=[
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    data_dir = find_data_dir()

    print("Datos:")
    print(data_dir)

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

    train_features = build_metadata_features(
        train_series
    )

    test_features = build_metadata_features(
        test_series
    )

    train_features = train_features.set_index(UID)
    test_features = test_features.set_index(UID)

    train_features, test_features = (
        train_features.align(
            test_features,
            join="outer",
            axis=1,
            fill_value=0.0,
        )
    )

    train_features = train_features.reset_index()
    test_features = test_features.reset_index()

    train_model_df = (
        train[[UID] + TARGETS]
        .merge(
            train_features,
            on=UID,
            how="left",
        )
    )

    test_model_df = (
        test[[UID]]
        .merge(
            test_features,
            on=UID,
            how="left",
        )
    )

    feature_cols = [
        col
        for col in train_features.columns
        if col != UID
    ]

    X_test = (
        test_model_df[feature_cols]
        .fillna(0.0)
        .astype(float)
    )

    submission = test[[UID]].copy()

    cv_scores = {}

    print(
        "Número de features:",
        len(feature_cols),
    )

    print(
        "Estudios test:",
        len(test),
    )

    for target in TARGETS:
        mask = train_model_df[target].notna()

        X = (
            train_model_df.loc[
                mask,
                feature_cols,
            ]
            .fillna(0.0)
            .astype(float)
        )

        y = (
            train_model_df.loc[
                mask,
                target,
            ]
            .astype(int)
        )

        print(
            f"\n{target}: "
            f"n={len(y)}, "
            f"positivos={int(y.sum())}, "
            f"negativos={int((1 - y).sum())}"
        )

        if len(y) == 0:
            submission[target] = 0.5
            cv_scores[target] = np.nan

            print(
                "Sin etiquetas. "
                "Se usa 0.5."
            )

            continue

        if y.nunique() < 2:
            prevalence = float(y.mean())

            submission[target] = prevalence
            cv_scores[target] = np.nan

            print(
                "Solo existe una clase. "
                f"Prevalencia={prevalence:.4f}"
            )

            continue

        model = make_model()

        min_class_count = int(
            y.value_counts().min()
        )

        n_splits = min(
            5,
            min_class_count,
        )

        if n_splits >= 2:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )

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

            cv_scores[target] = auc

            print(
                f"OOF AUC: {auc:.4f}"
            )

        else:
            cv_scores[target] = np.nan

            print(
                "Muy pocos ejemplos "
                "para calcular OOF AUC."
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

    valid_scores = [
        score
        for score in cv_scores.values()
        if pd.notna(score)
    ]

    if valid_scores:
        print(
            "\nMacro OOF AUC aproximado:",
            f"{np.mean(valid_scores):.4f}",
        )

    assert list(
        submission.columns
    ) == [UID] + TARGETS

    assert submission[UID].is_unique

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

    if Path(
        "/kaggle/working"
    ).exists():

        output_path = Path(
            "/kaggle/working/submission.csv"
        )

    else:
        output_path = Path(
            "submission_exp002_local.csv"
        )

    submission.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSubmission creada:"
    )

    print(output_path)

    print(
        submission.head()
    )


if __name__ == "__main__":
    main()