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

# EXP-003 solo mejoró claramente estos targets
PSEUDO_TARGETS = {
    "ACL",
    "PF OA",
    "Synovitis",
}


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
# METADATA
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
# EXP-002 MODEL
# Gold labels only
# ============================================================

def make_gold_model():
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
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# EXP-003 TEXT TEACHER
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
# EXP-003 METADATA STUDENT
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

    X_selected = (
        X_pseudo[selected]
    )

    p_selected = (
        p[selected]
    )

    confidence_selected = (
        confidence[selected]
    )

    gold_weights = (
        gold_sample_weights(
            y_gold
        )
    )

    X_parts = [
        X_gold,
    ]

    y_parts = [
        y_gold,
    ]

    weight_parts = [
        gold_weights,
    ]

    if len(X_selected) > 0:
        positive_weights = (
            p_selected
            * confidence_selected
        )

        negative_weights = (
            (1.0 - p_selected)
            * confidence_selected
        )

        raw_total = (
            positive_weights.sum()
            + negative_weights.sum()
        )

        target_weight = (
            pseudo_total_ratio
            * gold_weights.sum()
        )

        if raw_total > 0:
            scale = (
                target_weight
                / raw_total
            )
        else:
            scale = 0.0

        positive_weights = (
            positive_weights
            * scale
        )

        negative_weights = (
            negative_weights
            * scale
        )

        X_parts.extend(
            [
                X_selected,
                X_selected,
            ]
        )

        y_parts.extend(
            [
                np.ones(
                    len(X_selected),
                    dtype=int,
                ),
                np.zeros(
                    len(X_selected),
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
    print("EXP-004")
    print("Target Hybrid")
    print("EXP-002 Gold Metadata + EXP-003 Selected Pseudo Targets")
    print("=" * 70)

    print("\nPseudo targets:")
    print(sorted(PSEUDO_TARGETS))

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
        train_features
        .columns
        .tolist()
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

    submission = (
        test[[UID]]
        .copy()
    )

    gold_scores = {}
    hybrid_scores = {}
    teacher_scores = {}
    student_scores = {}

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    for target in TARGETS:
        print("\n" + "-" * 70)
        print(target)
        print("-" * 70)

        use_pseudo = (
            target
            in PSEUDO_TARGETS
        )

        print(
            "Modelo final:",
            (
                "EXP-003 Pseudo Student"
                if use_pseudo
                else "EXP-002 Gold Metadata"
            ),
        )

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
            model_df.loc[
                gold_mask
            ]
            .reset_index(
                drop=True
            )
        )

        pseudo_df = (
            model_df.loc[
                pseudo_mask
            ]
            .reset_index(
                drop=True
            )
        )

        y_gold = (
            gold_df[target]
            .astype(int)
            .to_numpy()
        )

        X_gold = (
            gold_df[
                feature_columns
            ]
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

        X_pseudo = (
            pseudo_df[
                feature_columns
            ]
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

        if (
            len(y_gold) == 0
            or len(np.unique(y_gold)) < 2
        ):
            prevalence = (
                float(y_gold.mean())
                if len(y_gold) > 0
                else 0.5
            )

            submission[target] = (
                prevalence
            )

            gold_scores[target] = np.nan
            hybrid_scores[target] = np.nan

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

        gold_oof = np.zeros(
            len(y_gold),
            dtype=float,
        )

        student_oof = np.zeros(
            len(y_gold),
            dtype=float,
        )

        teacher_oof = np.zeros(
            len(y_gold),
            dtype=float,
        )

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        pseudo_counts = []

        for (
            train_idx,
            valid_idx,
        ) in cv.split(
            X_gold,
            y_gold,
        ):
            # -----------------------------------------------
            # EXP-002 gold model
            # -----------------------------------------------

            gold_model = (
                make_gold_model()
            )

            gold_model.fit(
                X_gold[
                    train_idx
                ],
                y_gold[
                    train_idx
                ],
            )

            gold_oof[
                valid_idx
            ] = (
                gold_model
                .predict_proba(
                    X_gold[
                        valid_idx
                    ]
                )[:, 1]
            )

            # -----------------------------------------------
            # Only selected targets use EXP-003
            # -----------------------------------------------

            if use_pseudo:
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

                teacher_oof[
                    valid_idx
                ] = (
                    teacher
                    .predict_proba(
                        gold_reports[
                            valid_idx
                        ]
                    )[:, 1]
                )

                pseudo_probabilities = (
                    teacher
                    .predict_proba(
                        pseudo_reports
                    )[:, 1]
                )

                student, n_pseudo = (
                    fit_soft_student(
                        X_gold[
                            train_idx
                        ],
                        y_gold[
                            train_idx
                        ],
                        X_pseudo,
                        pseudo_probabilities,
                    )
                )

                pseudo_counts.append(
                    n_pseudo
                )

                student_oof[
                    valid_idx
                ] = (
                    student
                    .predict_proba(
                        X_gold[
                            valid_idx
                        ]
                    )[:, 1]
                )

        # ---------------------------------------------------
        # CV scores
        # ---------------------------------------------------

        gold_auc = (
            roc_auc_score(
                y_gold,
                gold_oof,
            )
        )

        gold_scores[target] = (
            gold_auc
        )

        print(
            f"Gold Metadata OOF AUC: "
            f"{gold_auc:.4f}"
        )

        if use_pseudo:
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

            teacher_scores[
                target
            ] = teacher_auc

            student_scores[
                target
            ] = student_auc

            hybrid_scores[
                target
            ] = student_auc

            print(
                f"Teacher OOF AUC: "
                f"{teacher_auc:.4f}"
            )

            print(
                f"Pseudo Student OOF AUC: "
                f"{student_auc:.4f}"
            )

            print(
                "Pseudo-labels usados por fold "
                f"(media): "
                f"{np.mean(pseudo_counts):.0f}"
            )

        else:
            hybrid_scores[
                target
            ] = gold_auc

        # ---------------------------------------------------
        # Final EXP-002 model
        # ---------------------------------------------------

        final_gold_model = (
            make_gold_model()
        )

        final_gold_model.fit(
            X_gold,
            y_gold,
        )

        # ---------------------------------------------------
        # Final chosen prediction
        # ---------------------------------------------------

        if use_pseudo:
            final_teacher = (
                make_text_teacher()
            )

            final_teacher.fit(
                gold_reports,
                y_gold,
            )

            final_pseudo_probabilities = (
                final_teacher
                .predict_proba(
                    pseudo_reports
                )[:, 1]
            )

            (
                final_student,
                final_pseudo_count,
            ) = fit_soft_student(
                X_gold,
                y_gold,
                X_pseudo,
                final_pseudo_probabilities,
            )

            print(
                "Pseudo-labels entrenamiento final:",
                final_pseudo_count,
            )

            submission[target] = (
                final_student
                .predict_proba(
                    X_test
                )[:, 1]
            )

        else:
            submission[target] = (
                final_gold_model
                .predict_proba(
                    X_test
                )[:, 1]
            )

    # ========================================================
    # RESULTS
    # ========================================================

    valid_gold_scores = [
        score
        for score
        in gold_scores.values()
        if pd.notna(score)
    ]

    valid_hybrid_scores = [
        score
        for score
        in hybrid_scores.values()
        if pd.notna(score)
    ]

    print("\n" + "=" * 70)
    print("VALIDACIÓN EXP-004")
    print("=" * 70)

    print(
        "Macro Gold Metadata OOF AUC:",
        f"{np.mean(valid_gold_scores):.4f}",
    )

    print(
        "Macro Hybrid OOF AUC:",
        f"{np.mean(valid_hybrid_scores):.4f}",
    )

    print(
        "\nTargets usando pseudo-labels:"
    )

    for target in sorted(
        PSEUDO_TARGETS
    ):
        print(
            f"  {target}"
        )

    print(
        "\nScores finales por target:"
    )

    for target in TARGETS:
        method = (
            "PSEUDO"
            if target in PSEUDO_TARGETS
            else "GOLD"
        )

        score = (
            hybrid_scores[target]
        )

        print(
            f"{target:20s} "
            f"{method:6s} "
            f"{score:.4f}"
        )

    # ========================================================
    # SUBMISSION VALIDATION
    # ========================================================

    assert (
        list(submission.columns)
        == [UID] + TARGETS
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
            "submission_exp004_local.csv"
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