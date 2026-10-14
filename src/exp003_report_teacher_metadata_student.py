from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler    


UID = "StudyInstanceUID"
REPORT = "Report"

RANDOM_STATE = 42

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


# ============================================================
# DATA
# ============================================================

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
        "No se encontraron los archivos de la competencia."
    )


# ============================================================
# METADATA FEATURES
# ============================================================

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

    df["plane"] = clean_category(
        df["Anatomical_Plane"]
    )

    df["fluid"] = clean_category(
        df["Fluid_Sensitive"]
    )

    df["fat"] = clean_category(
        df["Fat_Suppression"]
    )

    df["plane_fluid"] = (
        df["plane"]
        + "__"
        + df["fluid"]
    )

    df["plane_fat"] = (
        df["plane"]
        + "__"
        + df["fat"]
    )

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

    categorical_columns = [
        "plane",
        "fluid",
        "fat",
        "plane_fluid",
        "plane_fat",
        "protocol",
    ]

    for column in categorical_columns:
        dummies = pd.get_dummies(
            df[[UID, column]],
            columns=[column],
            prefix=column,
            dtype=float,
        )

        grouped = (
            dummies
            .groupby(UID)
            .sum()
        )

        features = features.join(
            grouped,
            how="left",
        )

    count_columns = [
        column
        for column in features.columns
        if column != "n_series"
    ]

    denominator = (
        features["n_series"]
        .clip(lower=1)
    )

    for column in count_columns:
        features[
            f"{column}_prop"
        ] = (
            features[column]
            / denominator
        )

    return features.reset_index()


# ============================================================
# TEXT TEACHER
# ============================================================

def make_text_teacher():
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=8000,
        sublinear_tf=True,
    )

    char_tfidf = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=12000,
        sublinear_tf=True,
    )

    features = FeatureUnion(
        [
            ("word", word_tfidf),
            ("char", char_tfidf),
        ]
    )

    model = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("tfidf", features),
            ("model", model),
        ]
    )


# ============================================================
# METADATA STUDENT
# ============================================================

def make_student():
    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    solver="liblinear",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def gold_sample_weights(y):
    y = np.asarray(y)

    n = len(y)

    positives = max(
        int((y == 1).sum()),
        1,
    )

    negatives = max(
        int((y == 0).sum()),
        1,
    )

    positive_weight = (
        n / (2.0 * positives)
    )

    negative_weight = (
        n / (2.0 * negatives)
    )

    return np.where(
        y == 1,
        positive_weight,
        negative_weight,
    )


def fit_soft_student(
    X_gold,
    y_gold,
    X_pseudo,
    pseudo_probabilities,
    pseudo_total_ratio=0.50,
    minimum_confidence=0.10,
):
    """
    Entrena un clasificador de metadata.

    Gold labels:
        peso completo.

    Pseudo-label:
        se representa como dos observaciones:
        y=1 con peso p
        y=0 con peso 1-p.

    De esta forma podemos utilizar probabilidades suaves
    con LogisticRegression.
    """

    X_gold = np.asarray(
        X_gold,
        dtype=float,
    )

    y_gold = np.asarray(
        y_gold,
        dtype=int,
    )

    X_pseudo = np.asarray(
        X_pseudo,
        dtype=float,
    )

    p = np.asarray(
        pseudo_probabilities,
        dtype=float,
    )

    confidence = (
        np.abs(p - 0.5)
        * 2.0
    )

    selected = (
        confidence
        >= minimum_confidence
    )

    X_pseudo_selected = (
        X_pseudo[selected]
    )

    p_selected = p[selected]

    confidence_selected = (
        confidence[selected]
    )

    X_parts = [
        X_gold,
    ]

    y_parts = [
        y_gold,
    ]

    gold_weights = (
        gold_sample_weights(y_gold)
    )

    weight_parts = [
        gold_weights,
    ]

    if len(X_pseudo_selected) > 0:
        raw_positive_weights = (
            p_selected
            * confidence_selected
        )

        raw_negative_weights = (
            (1.0 - p_selected)
            * confidence_selected
        )

        raw_total = (
            raw_positive_weights.sum()
            + raw_negative_weights.sum()
        )

        target_pseudo_weight = (
            pseudo_total_ratio
            * gold_weights.sum()
        )

        if raw_total > 0:
            scale = (
                target_pseudo_weight
                / raw_total
            )
        else:
            scale = 0.0

        positive_weights = (
            raw_positive_weights
            * scale
        )

        negative_weights = (
            raw_negative_weights
            * scale
        )

        X_parts.extend(
            [
                X_pseudo_selected,
                X_pseudo_selected,
            ]
        )

        y_parts.extend(
            [
                np.ones(
                    len(X_pseudo_selected),
                    dtype=int,
                ),
                np.zeros(
                    len(X_pseudo_selected),
                    dtype=int,
                ),
            ]
        )

        weight_parts.extend(
            [
                positive_weights,
                negative_weights,
            ]
        )

    X_train = np.vstack(
        X_parts
    )

    y_train = np.concatenate(
        y_parts
    )

    sample_weights = np.concatenate(
        weight_parts
    )

    model = make_student()

    model.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weights,
    )

    return (
        model,
        int(selected.sum()),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    data_dir = find_data_dir()

    print("=" * 70)
    print("EXP-003")
    print("Report Teacher -> Metadata Student")
    print("=" * 70)

    print("\nDatos:")
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

    train[REPORT] = (
        train[REPORT]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Metadata features
    # --------------------------------------------------------

    train_features = (
        build_metadata_features(
            train_series
        )
        .set_index(UID)
    )

    test_features = (
        build_metadata_features(
            test_series
        )
        .set_index(UID)
    )

    (
        train_features,
        test_features,
    ) = train_features.align(
        test_features,
        join="outer",
        axis=1,
        fill_value=0.0,
    )

    feature_columns = (
        train_features.columns.tolist()
    )

    train_features = (
        train_features
        .reset_index()
    )

    test_features = (
        test_features
        .reset_index()
    )

    model_df = train.merge(
        train_features,
        on=UID,
        how="left",
    )

    test_df = test.merge(
        test_features,
        on=UID,
        how="left",
    )

    X_test = (
        test_df[feature_columns]
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )

    print(
        "\nMetadata features:",
        len(feature_columns),
    )

    print(
        "Train studies:",
        len(train),
    )

    print(
        "Test studies visibles:",
        len(test),
    )

    # --------------------------------------------------------
    # Submission
    # --------------------------------------------------------

    submission = (
        test[[UID]]
        .copy()
    )

    student_auc_scores = {}
    teacher_auc_scores = {}

    # --------------------------------------------------------
    # Target loop
    # --------------------------------------------------------

    for target in TARGETS:
        print("\n" + "-" * 70)
        print(target)
        print("-" * 70)

        gold_mask = (
            model_df[target]
            .notna()
        )

        pseudo_mask = (
            model_df[target]
            .isna()
            &
            (
                model_df[REPORT]
                .str.strip()
                .str.len()
                > 0
            )
        )

        gold_df = (
            model_df.loc[gold_mask]
            .reset_index(drop=True)
        )

        pseudo_df = (
            model_df.loc[pseudo_mask]
            .reset_index(drop=True)
        )

        y_gold = (
            gold_df[target]
            .astype(int)
            .to_numpy()
        )

        X_gold_metadata = (
            gold_df[feature_columns]
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

        X_pseudo_metadata = (
            pseudo_df[feature_columns]
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

        gold_reports = (
            gold_df[REPORT]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        pseudo_reports = (
            pseudo_df[REPORT]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        positives = int(
            y_gold.sum()
        )

        negatives = int(
            len(y_gold)
            - positives
        )

        print(
            f"Gold n={len(y_gold)} | "
            f"positivos={positives} | "
            f"negativos={negatives}"
        )

        print(
            "Reportes disponibles para pseudo-label:",
            len(pseudo_df),
        )

        if (
            len(y_gold) == 0
            or len(np.unique(y_gold)) < 2
        ):
            prevalence = (
                float(y_gold.mean())
                if len(y_gold) > 0
                else 0.5
            )

            submission[target] = prevalence

            student_auc_scores[target] = np.nan
            teacher_auc_scores[target] = np.nan

            continue

        min_class_count = int(
            pd.Series(y_gold)
            .value_counts()
            .min()
        )

        n_splits = min(
            5,
            min_class_count,
        )

        teacher_oof = np.zeros(
            len(gold_df),
            dtype=float,
        )

        student_oof = np.zeros(
            len(gold_df),
            dtype=float,
        )

        if n_splits >= 2:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )

            pseudo_counts = []

            for fold, (
                train_idx,
                valid_idx,
            ) in enumerate(
                cv.split(
                    gold_reports,
                    y_gold,
                ),
                start=1,
            ):
                # --------------------------------------------
                # Teacher text model
                # --------------------------------------------

                teacher = (
                    make_text_teacher()
                )

                teacher.fit(
                    gold_reports[
                        train_idx
                    ],
                    y_gold[
                        train_idx
                    ],
                )

                teacher_valid = (
                    teacher
                    .predict_proba(
                        gold_reports[
                            valid_idx
                        ]
                    )[:, 1]
                )

                teacher_oof[
                    valid_idx
                ] = teacher_valid

                if len(pseudo_reports) > 0:
                    pseudo_probabilities = (
                        teacher
                        .predict_proba(
                            pseudo_reports
                        )[:, 1]
                    )
                else:
                    pseudo_probabilities = (
                        np.array([])
                    )

                # --------------------------------------------
                # Student metadata model
                # --------------------------------------------

                student, n_pseudo = (
                    fit_soft_student(
                        X_gold_metadata[
                            train_idx
                        ],
                        y_gold[
                            train_idx
                        ],
                        X_pseudo_metadata,
                        pseudo_probabilities,
                    )
                )

                pseudo_counts.append(
                    n_pseudo
                )

                student_valid = (
                    student
                    .predict_proba(
                        X_gold_metadata[
                            valid_idx
                        ]
                    )[:, 1]
                )

                student_oof[
                    valid_idx
                ] = student_valid

            teacher_auc = (
                roc_auc_score(
                    y_gold,
                    teacher_oof,
                )
            )

            student_auc = (
                roc_auc_score(
                    y_gold,
                    student_oof,
                )
            )

            teacher_auc_scores[
                target
            ] = teacher_auc

            student_auc_scores[
                target
            ] = student_auc

            print(
                f"Teacher text OOF AUC: "
                f"{teacher_auc:.4f}"
            )

            print(
                f"Student metadata OOF AUC: "
                f"{student_auc:.4f}"
            )

            print(
                "Pseudo-labels usados por fold "
                f"(media): "
                f"{np.mean(pseudo_counts):.0f}"
            )

        else:
            teacher_auc_scores[
                target
            ] = np.nan

            student_auc_scores[
                target
            ] = np.nan

        # ----------------------------------------------------
        # Final teacher using all gold
        # ----------------------------------------------------

        final_teacher = (
            make_text_teacher()
        )

        final_teacher.fit(
            gold_reports,
            y_gold,
        )

        if len(pseudo_reports) > 0:
            final_pseudo_probabilities = (
                final_teacher
                .predict_proba(
                    pseudo_reports
                )[:, 1]
            )
        else:
            final_pseudo_probabilities = (
                np.array([])
            )

        # ----------------------------------------------------
        # Final student
        # ----------------------------------------------------

        final_student, final_pseudo_count = (
            fit_soft_student(
                X_gold_metadata,
                y_gold,
                X_pseudo_metadata,
                final_pseudo_probabilities,
            )
        )

        print(
            "Pseudo-labels usados "
            "en entrenamiento final:",
            final_pseudo_count,
        )

        submission[target] = (
            final_student
            .predict_proba(
                X_test
            )[:, 1]
        )

    # ========================================================
    # RESULTS
    # ========================================================

    valid_teacher_scores = [
        score
        for score in teacher_auc_scores.values()
        if pd.notna(score)
    ]

    valid_student_scores = [
        score
        for score in student_auc_scores.values()
        if pd.notna(score)
    ]

    print("\n" + "=" * 70)
    print("VALIDACIÓN")
    print("=" * 70)

    if valid_teacher_scores:
        print(
            "Macro Teacher Text OOF AUC:",
            f"{np.mean(valid_teacher_scores):.4f}",
        )

    if valid_student_scores:
        print(
            "Macro Student Metadata OOF AUC:",
            f"{np.mean(valid_student_scores):.4f}",
        )

    print("\nScores Student:")

    for target in TARGETS:
        score = student_auc_scores.get(
            target,
            np.nan,
        )

        if pd.notna(score):
            print(
                f"{target:20s} "
                f"{score:.4f}"
            )

    # ========================================================
    # SUBMISSION VALIDATION
    # ========================================================

    expected_columns = (
        [UID]
        + TARGETS
    )

    assert (
        list(submission.columns)
        == expected_columns
    )

    assert (
        submission[UID]
        .is_unique
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

    # ========================================================
    # OUTPUT
    # ========================================================

    if Path(
        "/kaggle/working"
    ).exists():
        output_path = Path(
            "/kaggle/working/submission.csv"
        )
    else:
        output_path = Path(
            "submission_exp003_local.csv"
        )

    submission.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SUBMISSION")
    print("=" * 70)

    print(
        "Archivo creado:"
    )

    print(
        output_path
    )

    print(
        "\nVista previa:"
    )

    print(
        submission.head()
    )


if __name__ == "__main__":
    main()