from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    train_series = pd.read_csv(DATA_DIR / "train_series.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    test_series = pd.read_csv(DATA_DIR / "test_series.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    targets = [
        col for col in sample.columns
        if col != "StudyInstanceUID"
    ]

    print("\n" + "=" * 70)
    print("TRAIN.CSV")
    print("=" * 70)

    print("Forma:")
    print(train.shape)

    print("\nColumnas:")
    print(train.columns.tolist())

    print("\nPrimeras filas:")
    print(train.head())

    print("\nCantidad de estudios únicos:")
    print(train["StudyInstanceUID"].nunique())

    print("\nEtiquetas objetivo:")
    for target in targets:
        print("-", target)

    print("\n" + "=" * 70)
    print("ETIQUETAS")
    print("=" * 70)

    print("\nCantidad de valores NO nulos por etiqueta:")
    print(train[targets].notna().sum())

    print("\nCantidad de valores nulos por etiqueta:")
    print(train[targets].isna().sum())

    completely_labeled = train[targets].notna().all(axis=1)

    print("\nEstudios con TODAS las etiquetas disponibles:")
    print(int(completely_labeled.sum()))

    print("\nEstudios sin las 12 etiquetas completas:")
    print(int((~completely_labeled).sum()))

    if completely_labeled.any():
        print("\nPrevalencia de cada enfermedad:")
        print(
            train.loc[completely_labeled, targets]
            .mean()
            .sort_values(ascending=False)
        )

    print("\n" + "=" * 70)
    print("REPORTES")
    print("=" * 70)

    if "Report" in train.columns:
        print("\nReportes disponibles:")
        print(train["Report"].notna().sum())

        print("\nReportes faltantes:")
        print(train["Report"].isna().sum())

        first_report = train["Report"].dropna()

        if len(first_report) > 0:
            print("\nEjemplo de reporte:")
            print(first_report.iloc[0])

    print("\n" + "=" * 70)
    print("TRAIN_SERIES.CSV")
    print("=" * 70)

    print("Forma:")
    print(train_series.shape)

    print("\nColumnas:")
    print(train_series.columns.tolist())

    print("\nPrimeras filas:")
    print(train_series.head())

    print("\nNúmero de estudios en train_series:")
    print(train_series["StudyInstanceUID"].nunique())

    print("\nNúmero total de series:")
    print(train_series["SeriesInstanceUID"].nunique())

    if "Anatomical_Plane" in train_series.columns:
        print("\nPlanos anatómicos:")
        print(
            train_series["Anatomical_Plane"]
            .value_counts(dropna=False)
        )

    if "Fluid_Sensitive" in train_series.columns:
        print("\nFluid Sensitive:")
        print(
            train_series["Fluid_Sensitive"]
            .value_counts(dropna=False)
        )

    if "Fat_Suppression" in train_series.columns:
        print("\nFat Suppression:")
        print(
            train_series["Fat_Suppression"]
            .value_counts(dropna=False)
        )

    print("\nSeries por estudio:")
    series_per_study = (
        train_series
        .groupby("StudyInstanceUID")["SeriesInstanceUID"]
        .nunique()
    )

    print(series_per_study.describe())

    print("\n" + "=" * 70)
    print("TEST.CSV")
    print("=" * 70)

    print("Forma:")
    print(test.shape)

    print("\nColumnas:")
    print(test.columns.tolist())

    print("\nContenido:")
    print(test)

    print("\n" + "=" * 70)
    print("TEST_SERIES.CSV")
    print("=" * 70)

    print("Forma:")
    print(test_series.shape)

    print("\nPrimeras filas:")
    print(test_series.head())

    print("\n" + "=" * 70)
    print("SAMPLE_SUBMISSION.CSV")
    print("=" * 70)

    print("Forma:")
    print(sample.shape)

    print("\nColumnas:")
    print(sample.columns.tolist())

    print("\nContenido:")
    print(sample)

    print("\n" + "=" * 70)
    print("COMPROBACIONES")
    print("=" * 70)

    print(
        "StudyInstanceUID está en train:",
        "StudyInstanceUID" in train.columns
    )

    print(
        "StudyInstanceUID está en test:",
        "StudyInstanceUID" in test.columns
    )

    print(
        "Las etiquetas de submission existen en train:",
        all(target in train.columns for target in targets)
    )

    print("\nAnálisis terminado correctamente.")


if __name__ == "__main__":
    main()