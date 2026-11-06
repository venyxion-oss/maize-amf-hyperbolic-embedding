import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# CONFIGURATION

INPUT_CSV = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "FINAL_EXP_RESULTS.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "Statistical_Analysis_MATH"
)

SEED_VALUES = [42, 123, 456, 789, 1011]

MODELS = ["Poincare", "MDS", "Spectral"]

DIMENSIONS = [2, 16, 32]

ENVIRONMENTS = [
    "severe_drought",
    "moderate_drought",
    "normal",
    "wet",
    "very_wet"
]

ENVIRONMENT_ORDER = {
    "severe_drought": 0.2,
    "moderate_drought": 0.4,
    "normal": 0.6,
    "wet": 0.8,
    "very_wet": 1.0
}

METRICS = [
    "MRE",
    "Raw_MRE",
    "Stress",
    "Hierarchy",
    "Physical_Hydraulic",
    "Hydraulic",
    "Hydraulic_Preservation",
    "Embedding_Transport_Preservation",
    "Nutrient_Uptake",
    "Nutrient_Final",
    "Nutrient_Flux"
]

LOWER_IS_BETTER = {
    "MRE": True,
    "Raw_MRE": True,
    "Stress": True,
    "Hierarchy": False,
    "Physical_Hydraulic": False,
    "Hydraulic": False,
    "Hydraulic_Preservation": False,
    "Embedding_Transport_Preservation": False,
    "Nutrient_Uptake": False,
    "Nutrient_Final": False,
    "Nutrient_Flux": False
}


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find experiment output:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "file",
        "environment",
        "theta",
        "model",
        "dimension",
        "seed"
    ] + METRICS

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "TRY.csv is missing required columns:\n"
            + ", ".join(missing)
        )

    for column in METRICS + ["theta", "dimension", "seed"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["environment"] = pd.Categorical(
        df["environment"],
        categories=ENVIRONMENTS,
        ordered=True
    )

    df["model"] = pd.Categorical(
        df["model"],
        categories=MODELS,
        ordered=True
    )

    df["dimension"] = pd.Categorical(
        df["dimension"],
        categories=DIMENSIONS,
        ordered=True
    )

    return df


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality_report(df):
    print_header("DATA QUALITY")

    print(f"Rows in TRY.csv: {len(df):,}")
    print(f"Unique RSML files: {df['file'].nunique()}")
    print(f"Models: {df['model'].dropna().unique().tolist()}")
    print(f"Dimensions: {df['dimension'].dropna().unique().tolist()}")
    print(
        f"Environments: "
        f"{df['environment'].dropna().unique().tolist()}"
    )

    print("\nRows per model:")
    print(df["model"].value_counts(dropna=False))

    print("\nRows per dimension:")
    print(df["dimension"].value_counts(dropna=False))

    print("\nRows per environment:")
    print(df["environment"].value_counts(dropna=False))

    print("\nMissing values:")
    missing = df[METRICS].isna().sum()
    print(missing[missing > 0])

    duplicate_columns = [
        "file",
        "environment",
        "dimension",
        "model",
        "seed"
    ]

    duplicates = df.duplicated(
        subset=duplicate_columns,
        keep=False
    )

    print(
        f"\nDuplicate experimental rows: "
        f"{duplicates.sum()}"
    )

    expected_per_file = (
        len(ENVIRONMENTS)
        * len(DIMENSIONS)
        * len(MODELS)
        * len(SEED_VALUES)
    )

    print(
        f"\nExpected complete rows per file: "
        f"{expected_per_file}"
    )

    rows_per_file = df.groupby("file", observed=True).size()

    print("\nRows per file:")
    print(rows_per_file.describe())

    return {
        "duplicates": int(duplicates.sum()),
        "missing": missing.to_dict()
    }


# ============================================================
# SEED-LEVEL SUMMARY
# ============================================================

def create_seed_summary(df):
    print_header("SEED-LEVEL SUMMARY")

    group_columns = [
        "file",
        "environment",
        "theta",
        "dimension",
        "model"
    ]

    summary = (
        df.groupby(group_columns, observed=True)[METRICS]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "01_seed_level_summary.csv"
        ),
        index=False
    )

    return summary


# ============================================================
# BIOLOGICAL UNIT AGGREGATION
# ============================================================

def aggregate_seeds(df):
    """
    The RSML file is treated as the independent experimental unit.

    The five seeds are repeated stochastic realizations and are
    therefore averaged within:

        file × environment × dimension × model
    """

    print_header("AGGREGATING SEEDS")

    group_columns = [
        "file",
        "environment",
        "theta",
        "dimension",
        "model"
    ]

    aggregated = (
        df.groupby(group_columns, observed=True)[METRICS]
        .mean()
        .reset_index()
    )

    seed_counts = (
        df.groupby(group_columns, observed=True)["seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )

    aggregated = aggregated.merge(
        seed_counts,
        on=group_columns,
        how="left"
    )

    aggregated.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "02_file_level_seed_averaged_data.csv"
        ),
        index=False
    )

    print(
        f"Independent experimental units: "
        f"{aggregated['file'].nunique()} files"
    )

    print(
        "\nDistribution of number of seeds per condition:"
    )
    print(
        aggregated["n_seeds"]
        .value_counts()
        .sort_index()
    )

    return aggregated


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(aggregated):
    print_header("DESCRIPTIVE STATISTICS")

    group_columns = [
        "environment",
        "dimension",
        "model"
    ]

    rows = []

    for keys, group in aggregated.groupby(
        group_columns,
        observed=True
    ):
        environment, dimension, model = keys

        for metric in METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                continue

            rows.append({
                "environment": environment,
                "theta": ENVIRONMENT_ORDER.get(
                    str(environment),
                    np.nan
                ),
                "dimension": dimension,
                "model": model,
                "metric": metric,
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
                "sem": (
                    values.std(ddof=1)
                    / np.sqrt(len(values))
                    if len(values) > 1
                    else np.nan
                )
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "03_descriptive_statistics.csv"
        ),
        index=False
    )

    return result


# ============================================================
# PAIRED EFFECT SIZE
# ============================================================

def rank_biserial_effect(x, y):
    """
    Rank-biserial correlation for paired differences.

    Positive value means x tends to be greater than y.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan

    differences = x - y

    nonzero = differences != 0

    if not np.any(nonzero):
        return 0.0

    differences = differences[nonzero]

    ranks = stats.rankdata(
        np.abs(differences)
    )

    positive = np.sum(
        ranks[differences > 0]
    )

    negative = np.sum(
        ranks[differences < 0]
    )

    denominator = positive + negative

    if denominator == 0:
        return 0.0

    return float(
        (positive - negative)
        / denominator
    )


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_correction(p_values):
    """
    Holm-Bonferroni adjustment.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    result = np.full(
        len(p_values),
        np.nan
    )

    valid_indices = np.where(
        np.isfinite(p_values)
    )[0]

    if len(valid_indices) == 0:
        return result

    valid_p = p_values[valid_indices]

    order = np.argsort(valid_p)

    adjusted = np.empty_like(valid_p)

    running_max = 0.0

    m = len(valid_p)

    for rank, position in enumerate(order):
        adjusted_value = (
            valid_p[position]
            * (m - rank)
        )

        running_max = max(
            running_max,
            adjusted_value
        )

        adjusted[position] = min(
            running_max,
            1.0
        )

    result[valid_indices] = adjusted

    return result


# ============================================================
# FRIEDMAN TEST
# ============================================================

def friedman_model_test(
    aggregated,
    metric,
    dimension,
    environment
):
    subset = aggregated[
        (aggregated["dimension"] == dimension)
        & (aggregated["environment"] == environment)
    ].copy()

    pivot = subset.pivot(
        index="file",
        columns="model",
        values=metric
    )

    pivot = pivot[
        [m for m in MODELS if m in pivot.columns]
    ].dropna()

    if len(pivot) < 3 or len(pivot.columns) < 3:
        return None

    statistic, p_value = stats.friedmanchisquare(
        *[
            pivot[model].values
            for model in MODELS
        ]
    )

    return {
        "metric": metric,
        "dimension": int(dimension),
        "environment": str(environment),
        "n_files": len(pivot),
        "test": "Friedman",
        "statistic": statistic,
        "p_value": p_value
    }


def run_friedman_model_tests(aggregated):
    print_header("FRIEDMAN TESTS: MODEL COMPARISON")

    rows = []

    for metric in METRICS:
        for dimension in DIMENSIONS:
            for environment in ENVIRONMENTS:

                result = friedman_model_test(
                    aggregated,
                    metric,
                    dimension,
                    environment
                )

                if result is not None:
                    rows.append(result)

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df["p_holm_global"] = holm_correction(
            result_df["p_value"].values
        )

        result_df["significant"] = (
            result_df["p_holm_global"] < 0.05
        )

        result_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "04_friedman_model_tests.csv"
            ),
            index=False
        )

    return result_df


# ============================================================
# PAIRWISE WILCOXON MODEL COMPARISONS
# ============================================================

def pairwise_model_tests(aggregated):
    print_header("PAIRWISE WILCOXON MODEL COMPARISONS")

    rows = []

    model_pairs = list(
        combinations(MODELS, 2)
    )

    for metric in METRICS:
        for dimension in DIMENSIONS:
            for environment in ENVIRONMENTS:

                subset = aggregated[
                    (aggregated["dimension"] == dimension)
                    & (aggregated["environment"] == environment)
                ]

                pivot = subset.pivot(
                    index="file",
                    columns="model",
                    values=metric
                )

                for model_a, model_b in model_pairs:

                    if (
                        model_a not in pivot.columns
                        or model_b not in pivot.columns
                    ):
                        continue

                    paired = pivot[
                        [model_a, model_b]
                    ].dropna()

                    if len(paired) < 3:
                        continue

                    x = paired[model_a].values
                    y = paired[model_b].values

                    try:
                        statistic, p_value = stats.wilcoxon(
                            x,
                            y,
                            zero_method="wilcox",
                            alternative="two-sided"
                        )
                    except ValueError:
                        statistic = np.nan
                        p_value = 1.0

                    effect = rank_biserial_effect(
                        x,
                        y
                    )

                    mean_difference = float(
                        np.mean(x - y)
                    )

                    median_difference = float(
                        np.median(x - y)
                    )

                    rows.append({
                        "metric": metric,
                        "dimension": int(dimension),
                        "environment": str(environment),
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_files": len(paired),
                        "mean_a": np.mean(x),
                        "mean_b": np.mean(y),
                        "mean_difference_a_minus_b":
                            mean_difference,
                        "median_difference_a_minus_b":
                            median_difference,
                        "wilcoxon_statistic":
                            statistic,
                        "p_value": p_value,
                        "rank_biserial_effect":
                            effect
                    })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_holm_global"] = holm_correction(
            result["p_value"].values
        )

        result["significant"] = (
            result["p_holm_global"] < 0.05
        )

        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "05_pairwise_wilcoxon_model_tests.csv"
            ),
            index=False
        )

    return result


# ============================================================
# DIMENSION COMPARISON
# ============================================================

def dimension_comparison(aggregated):
    print_header("DIMENSION COMPARISON")

    rows = []

    for metric in METRICS:
        for model in MODELS:
            for environment in ENVIRONMENTS:

                subset = aggregated[
                    (aggregated["model"] == model)
                    & (aggregated["environment"] == environment)
                ]

                pivot = subset.pivot(
                    index="file",
                    columns="dimension",
                    values=metric
                )

                available = [
                    d for d in DIMENSIONS
                    if d in pivot.columns
                ]

                if len(available) < 3:
                    continue

                pivot = pivot[available].dropna()

                if len(pivot) < 3:
                    continue

                statistic, p_value = stats.friedmanchisquare(
                    *[
                        pivot[d].values
                        for d in available
                    ]
                )

                rows.append({
                    "metric": metric,
                    "model": model,
                    "environment": str(environment),
                    "n_files": len(pivot),
                    "test": "Friedman",
                    "statistic": statistic,
                    "p_value": p_value
                })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_holm_global"] = holm_correction(
            result["p_value"].values
        )

        result["significant"] = (
            result["p_holm_global"] < 0.05
        )

        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "06_dimension_friedman_tests.csv"
            ),
            index=False
        )

    return result


# ============================================================
# ENVIRONMENT COMPARISON
# ============================================================

def environment_comparison(aggregated):
    print_header("ENVIRONMENT COMPARISON")

    rows = []

    for metric in METRICS:
        for model in MODELS:
            for dimension in DIMENSIONS:

                subset = aggregated[
                    (aggregated["model"] == model)
                    & (aggregated["dimension"] == dimension)
                ]

                pivot = subset.pivot(
                    index="file",
                    columns="environment",
                    values=metric
                )

                available = [
                    e for e in ENVIRONMENTS
                    if e in pivot.columns
                ]

                if len(available) < 3:
                    continue

                pivot = pivot[available].dropna()

                if len(pivot) < 3:
                    continue

                statistic, p_value = stats.friedmanchisquare(
                    *[
                        pivot[e].values
                        for e in available
                    ]
                )

                rows.append({
                    "metric": metric,
                    "model": model,
                    "dimension": int(dimension),
                    "n_files": len(pivot),
                    "test": "Friedman",
                    "statistic": statistic,
                    "p_value": p_value
                })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_holm_global"] = holm_correction(
            result["p_value"].values
        )

        result["significant"] = (
            result["p_holm_global"] < 0.05
        )

        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "07_environment_friedman_tests.csv"
            ),
            index=False
        )

    return result


# ============================================================
# ENVIRONMENTAL TREND ANALYSIS
# ============================================================

def environmental_trends(aggregated):
    print_header("ENVIRONMENTAL TREND ANALYSIS")

    rows = []

    for metric in METRICS:
        for model in MODELS:
            for dimension in DIMENSIONS:

                subset = aggregated[
                    (aggregated["model"] == model)
                    & (aggregated["dimension"] == dimension)
                ].copy()

                if subset.empty:
                    continue

                file_results = []

                for filename, group in subset.groupby(
                    "file",
                    observed=True
                ):
                    group = group.copy()

                    group["theta_numeric"] = (
                        group["environment"]
                        .astype(str)
                        .map(ENVIRONMENT_ORDER)
                    )

                    valid = group[
                        ["theta_numeric", metric]
                    ].dropna()

                    if len(valid) < 3:
                        continue

                    rho, p_value = stats.spearmanr(
                        valid["theta_numeric"],
                        valid[metric]
                    )

                    if np.isfinite(rho):
                        file_results.append(rho)

                if len(file_results) < 3:
                    continue

                mean_rho = float(
                    np.mean(file_results)
                )

                statistic, p_value = stats.wilcoxon(
                    np.asarray(file_results),
                    np.zeros(len(file_results)),
                    alternative="two-sided"
                )

                rows.append({
                    "metric": metric,
                    "model": model,
                    "dimension": int(dimension),
                    "n_files": len(file_results),
                    "mean_file_spearman_rho":
                        mean_rho,
                    "median_file_spearman_rho":
                        np.median(file_results),
                    "wilcoxon_against_zero":
                        statistic,
                    "p_value":
                        p_value
                })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_holm_global"] = holm_correction(
            result["p_value"].values
        )

        result["significant"] = (
            result["p_holm_global"] < 0.05
        )

        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "08_environmental_trends.csv"
            ),
            index=False
        )

    return result


# ============================================================
# POINCARE CURVATURE
# ============================================================

def analyze_poincare_curvature(df):
    print_header("POINCARE CURVATURE")

    if "curvature" not in df.columns:
        print("No curvature column found.")
        return pd.DataFrame()

    poincare = df[
        df["model"] == "Poincare"
    ].copy()

    poincare["curvature"] = pd.to_numeric(
        poincare["curvature"],
        errors="coerce"
    )

    poincare = poincare[
        np.isfinite(poincare["curvature"])
        & (poincare["curvature"] > 0)
    ]

    if poincare.empty:
        print("No valid Poincare curvature values.")
        return pd.DataFrame()

    group_columns = [
        "environment",
        "dimension"
    ]

    result = (
        poincare.groupby(
            group_columns,
            observed=True
        )["curvature"]
        .agg([
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max"
        ])
        .reset_index()
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "09_poincare_curvature_summary.csv"
        ),
        index=False
    )

    return result


# ============================================================
# METRIC CORRELATIONS
# ============================================================

def metric_correlations(aggregated):
    print_header("METRIC CORRELATIONS")

    rows = []

    available_metrics = [
        metric
        for metric in METRICS
        if metric in aggregated.columns
    ]

    for metric_a, metric_b in combinations(
        available_metrics,
        2
    ):

        pair = aggregated[
            [metric_a, metric_b]
        ].dropna()

        if len(pair) < 5:
            continue

        rho, p_value = stats.spearmanr(
            pair[metric_a],
            pair[metric_b]
        )

        rows.append({
            "metric_a": metric_a,
            "metric_b": metric_b,
            "n": len(pair),
            "spearman_rho": rho,
            "p_value": p_value
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_holm_global"] = holm_correction(
            result["p_value"].values
        )

        result["significant"] = (
            result["p_holm_global"] < 0.05
        )

        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "10_metric_spearman_correlations.csv"
            ),
            index=False
        )

    return result


# ============================================================
# BEST-MODEL ANALYSIS
# ============================================================

def determine_best_models(aggregated):
    print_header("BEST MODEL ANALYSIS")

    rows = []

    for metric in METRICS:
        for dimension in DIMENSIONS:
            for environment in ENVIRONMENTS:

                subset = aggregated[
                    (aggregated["dimension"] == dimension)
                    & (aggregated["environment"] == environment)
                ]

                if subset.empty:
                    continue

                means = (
                    subset.groupby(
                        "model",
                        observed=True
                    )[metric]
                    .mean()
                    .dropna()
                )

                if means.empty:
                    continue

                if LOWER_IS_BETTER.get(metric, False):
                    best_model = means.idxmin()
                    best_value = means.min()
                else:
                    best_model = means.idxmax()
                    best_value = means.max()

                rows.append({
                    "metric": metric,
                    "dimension": int(dimension),
                    "environment": str(environment),
                    "best_model": best_model,
                    "best_mean_value": best_value
                })

    result = pd.DataFrame(rows)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "11_best_model_by_condition.csv"
        ),
        index=False
    )

    return result


# ============================================================
# PERCENT IMPROVEMENT
# ============================================================

def calculate_pairwise_improvements(aggregated):
    print_header("PAIRWISE MODEL IMPROVEMENTS")

    rows = []

    comparisons = [
        ("Poincare", "MDS"),
        ("Poincare", "Spectral"),
        ("MDS", "Spectral")
    ]

    for metric in METRICS:
        for dimension in DIMENSIONS:
            for environment in ENVIRONMENTS:

                subset = aggregated[
                    (aggregated["dimension"] == dimension)
                    & (aggregated["environment"] == environment)
                ]

                pivot = subset.pivot(
                    index="file",
                    columns="model",
                    values=metric
                )

                for model_a, model_b in comparisons:

                    if (
                        model_a not in pivot.columns
                        or model_b not in pivot.columns
                    ):
                        continue

                    paired = pivot[
                        [model_a, model_b]
                    ].dropna()

                    if paired.empty:
                        continue

                    a = paired[model_a].values
                    b = paired[model_b].values

                    mean_a = np.mean(a)
                    mean_b = np.mean(b)

                    if abs(mean_b) < 1e-12:
                        percent_change = np.nan
                    else:
                        if LOWER_IS_BETTER.get(
                            metric,
                            False
                        ):
                            percent_change = (
                                (mean_b - mean_a)
                                / abs(mean_b)
                                * 100.0
                            )
                        else:
                            percent_change = (
                                (mean_a - mean_b)
                                / abs(mean_b)
                                * 100.0
                            )

                    rows.append({
                        "metric": metric,
                        "dimension": int(dimension),
                        "environment": str(environment),
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "percent_improvement_a_vs_b":
                            percent_change
                    })

    result = pd.DataFrame(rows)

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "12_pairwise_percent_improvements.csv"
        ),
        index=False
    )

    return result


# ============================================================
# PLOTS
# ============================================================

def plot_metric_by_environment(
    aggregated,
    metric
):
    for dimension in DIMENSIONS:

        plt.figure(
            figsize=(10, 6)
        )

        for model in MODELS:

            subset = aggregated[
                (aggregated["dimension"] == dimension)
                & (aggregated["model"] == model)
            ].copy()

            if subset.empty:
                continue

            grouped = (
                subset.groupby(
                    "environment",
                    observed=True
                )[metric]
                .agg(["mean", "sem"])
                .reindex(ENVIRONMENTS)
            )

            x = np.arange(
                len(ENVIRONMENTS)
            )

            plt.errorbar(
                x,
                grouped["mean"],
                yerr=grouped["sem"],
                marker="o",
                capsize=4,
                label=model
            )

        plt.xticks(
            np.arange(len(ENVIRONMENTS)),
            [
                "Severe drought",
                "Moderate drought",
                "Normal",
                "Wet",
                "Very wet"
            ],
            rotation=20
        )

        plt.xlabel("Environmental condition")
        plt.ylabel(metric)
        plt.title(
            f"{metric} by environment — D={dimension}"
        )

        plt.legend()
        plt.tight_layout()

        filename = (
            f"{metric}_environment_D{dimension}.png"
        )

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "figures",
                filename
            ),
            dpi=300
        )

        plt.close()


def create_figures(aggregated):
    print_header("CREATING FIGURES")

    figure_dir = os.path.join(
        OUTPUT_DIR,
        "figures"
    )

    os.makedirs(
        figure_dir,
        exist_ok=True
    )

    important_metrics = [
        "MRE",
        "Stress",
        "Hierarchy",
        "Hydraulic_Preservation",
        "Embedding_Transport_Preservation",
        "Nutrient_Uptake",
        "Nutrient_Final",
        "Nutrient_Flux"
    ]

    for metric in important_metrics:
        plot_metric_by_environment(
            aggregated,
            metric
        )

    print(
        f"Figures saved to: {figure_dir}"
    )


# ============================================================
# TEXT REPORT
# ============================================================

def write_report(
    df,
    aggregated,
    friedman,
    pairwise,
    dimensions,
    environments,
    trends
):
    report_path = os.path.join(
        OUTPUT_DIR,
        "STATISTICAL_REPORT.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "COUPLED ZEA MAYS–AMF EMBEDDING EXPERIMENT\n"
        )
        report.write(
            "STATISTICAL ANALYSIS REPORT\n"
        )
        report.write("=" * 80 + "\n\n")

        report.write(
            "EXPERIMENTAL UNIT\n"
        )
        report.write(
            "The RSML/root-network file was treated as the "
            "independent experimental unit. The five seeds "
            "were treated as repeated stochastic runs and "
            "were averaged within each file × environment × "
            "dimension × model condition before inferential "
            "testing.\n\n"
        )

        report.write(
            f"Raw rows: {len(df)}\n"
        )
        report.write(
            f"Unique files: {df['file'].nunique()}\n"
        )
        report.write(
            f"Models: {', '.join(MODELS)}\n"
        )
        report.write(
            f"Dimensions: {', '.join(map(str, DIMENSIONS))}\n"
        )
        report.write(
            f"Environments: {', '.join(ENVIRONMENTS)}\n\n"
        )

        # ----------------------------------------------------
        # Friedman
        # ----------------------------------------------------

        report.write(
            "MODEL COMPARISONS — FRIEDMAN TEST\n"
        )
        report.write("-" * 80 + "\n")

        if friedman.empty:
            report.write(
                "No complete Friedman tests were available.\n\n"
            )
        else:
            significant = friedman[
                friedman["p_holm_global"] < 0.05
            ]

            report.write(
                f"Total tests: {len(friedman)}\n"
            )
            report.write(
                f"Significant after Holm correction: "
                f"{len(significant)}\n\n"
            )

            if not significant.empty:
                for _, row in significant.iterrows():
                    report.write(
                        f"{row['metric']} | "
                        f"D={int(row['dimension'])} | "
                        f"{row['environment']} | "
                        f"Friedman statistic="
                        f"{row['statistic']:.4f} | "
                        f"Holm p="
                        f"{row['p_holm_global']:.6g}\n"
                    )

            report.write("\n")

        # ----------------------------------------------------
        # Pairwise
        # ----------------------------------------------------

        report.write(
            "PAIRWISE MODEL COMPARISONS — WILCOXON\n"
        )
        report.write("-" * 80 + "\n")

        if pairwise.empty:
            report.write(
                "No pairwise tests were available.\n\n"
            )
        else:
            significant = pairwise[
                pairwise["p_holm_global"] < 0.05
            ]

            report.write(
                f"Total pairwise tests: {len(pairwise)}\n"
            )
            report.write(
                f"Significant after Holm correction: "
                f"{len(significant)}\n\n"
            )

            for _, row in significant.iterrows():
                report.write(
                    f"{row['metric']} | "
                    f"D={int(row['dimension'])} | "
                    f"{row['environment']} | "
                    f"{row['model_a']} vs "
                    f"{row['model_b']} | "
                    f"Holm p="
                    f"{row['p_holm_global']:.6g} | "
                    f"rank-biserial="
                    f"{row['rank_biserial_effect']:.4f}\n"
                )

            report.write("\n")

        # ----------------------------------------------------
        # Dimension
        # ----------------------------------------------------

        report.write(
            "DIMENSION COMPARISONS\n"
        )
        report.write("-" * 80 + "\n")

        if dimensions.empty:
            report.write(
                "No dimension tests were available.\n\n"
            )
        else:
            significant = dimensions[
                dimensions["p_holm_global"] < 0.05
            ]

            report.write(
                f"Significant dimension tests after Holm: "
                f"{len(significant)}\n\n"
            )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        report.write(
            "ENVIRONMENT COMPARISONS\n"
        )
        report.write("-" * 80 + "\n")

        if environments.empty:
            report.write(
                "No environment tests were available.\n\n"
            )
        else:
            significant = environments[
                environments["p_holm_global"] < 0.05
            ]

            report.write(
                f"Significant environment tests after Holm: "
                f"{len(significant)}\n\n"
            )

        # ----------------------------------------------------
        # Trends
        # ----------------------------------------------------

        report.write(
            "ENVIRONMENTAL TRENDS\n"
        )
        report.write("-" * 80 + "\n")

        if trends.empty:
            report.write(
                "No environmental trend tests were available.\n\n"
            )
        else:
            significant = trends[
                trends["p_holm_global"] < 0.05
            ]

            report.write(
                f"Significant trends after Holm: "
                f"{len(significant)}\n\n"
            )

            for _, row in significant.iterrows():
                report.write(
                    f"{row['metric']} | "
                    f"{row['model']} | "
                    f"D={int(row['dimension'])} | "
                    f"mean Spearman rho="
                    f"{row['mean_file_spearman_rho']:.4f} | "
                    f"Holm p="
                    f"{row['p_holm_global']:.6g}\n"
                )

        report.write("\n")
        report.write(
            "IMPORTANT INTERPRETATION NOTE\n"
        )
        report.write("-" * 80 + "\n")
        report.write(
            "Statistical significance does not by itself establish "
            "biological superiority. Results should be interpreted "
            "together with effect sizes, descriptive statistics, "
            "model behavior across dimensions, and environmental "
            "patterns.\n"
        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("COUPLED ZEA MAYS–AMF STATISTICAL ANALYSIS")
    print("=" * 90)

    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        INPUT_CSV
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    data_quality_report(
        df
    )

    # --------------------------------------------------------
    # Seed summary
    # --------------------------------------------------------

    create_seed_summary(
        df
    )

    # --------------------------------------------------------
    # Aggregate seeds
    # --------------------------------------------------------

    aggregated = aggregate_seeds(
        df
    )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    descriptive_statistics(
        aggregated
    )

    # --------------------------------------------------------
    # Model tests
    # --------------------------------------------------------

    friedman = run_friedman_model_tests(
        aggregated
    )

    pairwise = pairwise_model_tests(
        aggregated
    )

    # --------------------------------------------------------
    # Dimension tests
    # --------------------------------------------------------

    dimension_results = dimension_comparison(
        aggregated
    )

    # --------------------------------------------------------
    # Environment tests
    # --------------------------------------------------------

    environment_results = environment_comparison(
        aggregated
    )

    # --------------------------------------------------------
    # Environmental trends
    # --------------------------------------------------------

    trend_results = environmental_trends(
        aggregated
    )

    # --------------------------------------------------------
    # Poincare curvature
    # --------------------------------------------------------

    analyze_poincare_curvature(
        df
    )

    # --------------------------------------------------------
    # Metric correlations
    # --------------------------------------------------------

    metric_correlations(
        aggregated
    )

    # --------------------------------------------------------
    # Best models
    # --------------------------------------------------------

    determine_best_models(
        aggregated
    )

    # --------------------------------------------------------
    # Percent improvements
    # --------------------------------------------------------

    calculate_pairwise_improvements(
        aggregated
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    create_figures(
        aggregated
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = write_report(
        df,
        aggregated,
        friedman,
        pairwise,
        dimension_results,
        environment_results,
        trend_results
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print_header("STATISTICAL ANALYSIS COMPLETE")

    print(
        f"Independent files analyzed: "
        f"{aggregated['file'].nunique()}"
    )

    print(
        f"File-level observations: "
        f"{len(aggregated)}"
    )

    print(
        f"\nResults directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nMain report:\n"
        f"{report_path}"
    )

    print(
        "\nGenerated tables include:"
    )

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):
        if filename.endswith(".csv"):
            print(f"  - {filename}")

    print(
        "\nFigures are in:"
    )
    print(
        os.path.join(
            OUTPUT_DIR,
            "figures"
        )
    )


if __name__ == "__main__":
    main()