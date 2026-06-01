"""
Полусеместровый контроль №4
Поиск аномально активных респондентов в SoS.

Запуск:
    python solution.py
или:
    python solution.py --data data_train --output output
    python solution.py --data data_train.zip --output output

Алгоритм:
1. Берем только строки BrandinDelivery = 1 и непустую CategoryDelivery/CategoryNameDelivery.
2. Считаем daily_ots на уровне SubjectID + researchdate + BrandID + CategoryDelivery:
       daily_ots = Weight * count_rows
3. Для каждой CategoryDelivery строим устойчивый порог экстремально высокого daily_ots
   по log1p(daily_ots):
       threshold_log = max(Q99.9, Q3 + 2 * IQR)
   Это не фиксированное число удаляемых людей, а статистический порог хвоста распределения
   внутри каждой категории.
4. Триггер аномалии: count_rows >= 2 и daily_ots выше порога категории.
5. В anomalies.csv попадает не бренд, а вся пара SubjectID + researchdate.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_STATE = 42
CATEGORY_CANDIDATES = ["CategoryDelivery", "CategoryNameDelivery"]
REQUIRED_REASON_COLUMNS = [
    "SubjectID",
    "researchdate",
    "BrandID",
    "Brand",
    "CategoryDelivery",
    "daily_ots",
    "score",
    "threshold",
    "reason",
]


def resolve_data_path(data_arg: str | None) -> Path:
    """Находит папку/архив с данными без ручного переписывания кода."""
    candidates: list[Path] = []
    if data_arg:
        candidates.append(Path(data_arg))
    candidates.extend([
        Path("data_train"),
        Path("data_train.zip"),
        Path("."),
    ])

    for path in candidates:
        if path.exists():
            if path.is_file() and path.suffix.lower() == ".zip":
                extract_dir = Path("_unzipped_data_train")
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)
                return extract_dir
            return path

    raise FileNotFoundError(
        "Не нашел данные. Положите рядом data_train/ или data_train.zip, "
        "либо передайте путь через --data."
    )


def find_parquet_files(data_path: Path) -> list[Path]:
    """Ищет parquet-файлы, исключая папку examples, чтобы не подглядывать в пример ответа."""
    files = []
    for file in data_path.rglob("*.parquet"):
        if "examples" not in {part.lower() for part in file.parts}:
            files.append(file)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"В {data_path} не найдены parquet-файлы с данными.")
    return files


def load_data(data_path: Path) -> pd.DataFrame:
    files = find_parquet_files(data_path)
    frames = [pd.read_parquet(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def get_category_column(df: pd.DataFrame) -> str:
    for col in CATEGORY_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError("Не найден столбец CategoryDelivery или CategoryNameDelivery.")


def normalize_input(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Приводит данные к стабильным типам и единому названию категории."""
    category_col = get_category_column(df)
    df = df.copy()

    if category_col != "CategoryDelivery":
        df["CategoryDelivery"] = df[category_col]
        category_col = "CategoryDelivery"

    df["researchdate"] = pd.to_datetime(df["researchdate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Weight"] = pd.to_numeric(df["Weight"].astype(str), errors="coerce")
    df["BrandinDelivery"] = pd.to_numeric(df["BrandinDelivery"], errors="coerce").fillna(0)

    # ID и BrandID оставляем как строки/исходные значения без изменения смысла.
    # Это помогает избежать проблем с большими числами при промежуточных операциях.
    return df, category_col


def prepare_analysis_rows(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    mask = (
        df["BrandinDelivery"].eq(1)
        & df[category_col].notna()
        & df[category_col].astype(str).str.strip().ne("")
        & df["researchdate"].notna()
        & df["Weight"].notna()
    )
    return df.loc[mask].copy().reset_index(drop=True)


def aggregate_daily_ots(df_analysis: pd.DataFrame, category_col: str) -> pd.DataFrame:
    keys = ["SubjectID", "researchdate", "BrandID", category_col]
    agg = (
        df_analysis.groupby(keys, observed=True)
        .agg(
            count_rows=("QueryText", "size"),
            Weight=("Weight", "first"),
            Brand=("Brand", "first"),
        )
        .reset_index()
    )
    agg["daily_ots"] = agg["Weight"] * agg["count_rows"]
    agg["log_daily_ots"] = np.log1p(agg["daily_ots"])

    brand_day = (
        agg.groupby(["researchdate", "BrandID", category_col], observed=True)
        .agg(
            brand_day_total_ots=("daily_ots", "sum"),
            brand_day_respondents=("SubjectID", "nunique"),
        )
        .reset_index()
    )
    agg = agg.merge(brand_day, on=["researchdate", "BrandID", category_col], how="left")
    agg["respondent_share_in_brand_day"] = agg["daily_ots"] / agg["brand_day_total_ots"]
    return agg


def build_category_thresholds(agg: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """
    Порог считается отдельно для каждой категории.

    Используется логарифм daily_ots, чтобы сверхбольшие веса не ломали масштаб.
    max(Q99.9, Q3 + 2*IQR) делает порог строгим:
    - Q99.9 ловит самый верхний хвост;
    - Q3 + 2*IQR защищает от ситуации, когда в маленькой категории максимум не является явной аномалией.
    """
    stats = (
        agg.groupby(category_col, observed=True)["log_daily_ots"]
        .agg(
            category_n="size",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            q999=lambda s: s.quantile(0.999),
        )
        .reset_index()
    )
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["threshold_log"] = np.maximum(stats["q999"], stats["q3"] + 2.0 * stats["iqr"])
    stats["threshold_ots"] = np.expm1(stats["threshold_log"])
    return stats[[category_col, "category_n", "threshold_log", "threshold_ots"]]


def detect_anomalies(agg: pd.DataFrame, category_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = build_category_thresholds(agg, category_col)
    scored = agg.merge(thresholds, on=category_col, how="left")

    # score > 1 означает, что daily_ots выше порога.
    scored["score"] = scored["daily_ots"] / scored["threshold_ots"]
    scored["threshold"] = scored["threshold_ots"]

    anomaly_mask = scored["score"].gt(1.0) & scored["count_rows"].ge(2)
    reasons = scored.loc[anomaly_mask].copy()

    reasons["reason"] = (
        "daily_ots выше экстремального категорийного порога: "
        "threshold=max(Q99.9, Q3+2*IQR) по log1p(daily_ots); "
        "count_rows=" + reasons["count_rows"].astype(str)
        + "; share_in_brand_day="
        + (reasons["respondent_share_in_brand_day"] * 100).round(1).astype(str)
        + "%"
    )

    reasons = reasons.rename(columns={category_col: "CategoryDelivery"})
    reasons = reasons[REQUIRED_REASON_COLUMNS].sort_values(
        ["researchdate", "SubjectID", "BrandID", "CategoryDelivery"]
    )

    anomalies = (
        reasons[["SubjectID", "researchdate"]]
        .drop_duplicates()
        .sort_values(["researchdate", "SubjectID"])
        .reset_index(drop=True)
    )
    return anomalies, reasons


def mark_removed(df_analysis: pd.DataFrame, anomalies: pd.DataFrame) -> pd.Series:
    marker = anomalies.copy()
    marker["_remove"] = True
    joined = df_analysis.merge(marker, on=["SubjectID", "researchdate"], how="left")
    return joined["_remove"].eq(True)


def compute_total_ots_by_day(df_analysis: pd.DataFrame) -> pd.DataFrame:
    return (
        df_analysis.assign(row_ots=df_analysis["Weight"])
        .groupby("researchdate", observed=True)["row_ots"]
        .sum()
        .reset_index(name="total_ots")
        .sort_values("researchdate")
    )


def compute_category_ots(df_analysis: pd.DataFrame, category_col: str) -> pd.DataFrame:
    return (
        df_analysis.assign(row_ots=df_analysis["Weight"])
        .groupby(category_col, observed=True)["row_ots"]
        .sum()
        .reset_index(name="total_ots")
    )


def save_outputs(
    df_analysis: pd.DataFrame,
    anomalies: pd.DataFrame,
    reasons: pd.DataFrame,
    output_dir: Path,
    category_col: str,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    anomalies.to_csv(output_dir / "anomalies.csv", index=False)
    reasons.to_csv(output_dir / "anomaly_reasons.csv", index=False)

    removed_mask = mark_removed(df_analysis, anomalies).to_numpy()
    before = df_analysis.copy()
    after = df_analysis.loc[~removed_mask].copy()

    plot_total_ots_before_after(before, after, plots_dir / "total_ots_before_after.png")
    plot_category_ots_change(before, after, category_col, plots_dir / "category_ots_change.png")
    plot_daily_anomaly_count(anomalies, plots_dir / "daily_anomaly_count.png")


def plot_total_ots_before_after(before: pd.DataFrame, after: pd.DataFrame, output_path: Path) -> None:
    before_day = compute_total_ots_by_day(before).rename(columns={"total_ots": "before"})
    after_day = compute_total_ots_by_day(after).rename(columns={"total_ots": "after"})
    plot_df = before_day.merge(after_day, on="researchdate", how="left").fillna({"after": 0})

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df["researchdate"], plot_df["before"], marker="o", label="До очистки")
    plt.plot(plot_df["researchdate"], plot_df["after"], marker="o", label="После очистки")
    plt.title("Суммарный OTS по дням до и после удаления аномальных респондентов")
    plt.xlabel("Дата")
    plt.ylabel("OTS")
    plt.xticks(rotation=90, fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_category_ots_change(before: pd.DataFrame, after: pd.DataFrame, category_col: str, output_path: Path) -> None:
    before_cat = compute_category_ots(before, category_col).rename(columns={"total_ots": "before"})
    after_cat = compute_category_ots(after, category_col).rename(columns={"total_ots": "after"})
    plot_df = before_cat.merge(after_cat, on=category_col, how="left").fillna({"after": 0})
    plot_df["change_pct"] = (plot_df["after"] / plot_df["before"] - 1.0) * 100
    plot_df = plot_df.sort_values("change_pct")

    plt.figure(figsize=(16, 7))
    plt.bar(plot_df[category_col].astype(str), plot_df["change_pct"])
    plt.axhline(0, linewidth=1)
    plt.title("Изменение OTS по CategoryDelivery после очистки, %")
    plt.xlabel("Категория")
    plt.ylabel("Изменение OTS, %")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_daily_anomaly_count(anomalies: pd.DataFrame, output_path: Path) -> None:
    counts = anomalies.groupby("researchdate", observed=True).size().reset_index(name="anomaly_count")
    plt.figure(figsize=(14, 5))
    plt.bar(counts["researchdate"], counts["anomaly_count"])
    plt.title("Количество аномальных респондентов по дням")
    plt.xlabel("Дата")
    plt.ylabel("Количество респондентов")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


# Дополнительные аналитические функции из требования 8.2.
def plot_before_after_by_column(
    df_analysis: pd.DataFrame,
    anomalies: pd.DataFrame,
    column: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Строит OTS до/после по любому разрезу: пол, возраст, регион, ресурс, платформа, категория."""
    if column not in df_analysis.columns:
        raise KeyError(f"Столбец {column} отсутствует в данных.")

    removed_mask = mark_removed(df_analysis, anomalies).to_numpy()
    before = df_analysis.assign(row_ots=df_analysis["Weight"]).groupby(column, observed=True)["row_ots"].sum()
    after = (
        df_analysis.loc[~removed_mask]
        .assign(row_ots=lambda x: x["Weight"])
        .groupby(column, observed=True)["row_ots"]
        .sum()
    )
    result = pd.concat([before.rename("before"), after.rename("after")], axis=1).fillna(0).reset_index()
    result["change_pct"] = (result["after"] / result["before"] - 1.0) * 100
    result = result.sort_values("change_pct")

    plt.figure(figsize=(14, 6))
    plt.bar(result[column].astype(str), result["change_pct"])
    plt.axhline(0, linewidth=1)
    plt.title(f"Изменение OTS по {column}, %")
    plt.xlabel(column)
    plt.ylabel("Изменение OTS, %")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return result


def get_anomalous_queries(
    original_df: pd.DataFrame,
    subject_id,
    researchdate: str,
) -> pd.DataFrame:
    """Возвращает QueryText выбранного аномального респондента за выбранный день."""
    df = original_df.copy()
    df["researchdate"] = pd.to_datetime(df["researchdate"], errors="coerce").dt.strftime("%Y-%m-%d")
    mask = df["SubjectID"].eq(subject_id) & df["researchdate"].eq(str(researchdate))
    columns = [
        col
        for col in [
            "SubjectID",
            "researchdate",
            "Start",
            "QueryText",
            "BrandID",
            "Brand",
            "CategoryDelivery",
            "CategoryNameDelivery",
            "ResourceName",
            "ResourceType",
            "Platform",
            "UseType",
            "Weight",
        ]
        if col in df.columns
    ]
    return df.loc[mask, columns].sort_values(columns=[c for c in ["Start", "QueryText"] if c in columns])


def plot_brand_ots_by_day(
    df_analysis: pd.DataFrame,
    anomalies: pd.DataFrame,
    brand_id,
    category_delivery: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """График OTS по дням для выбранного бренда до/после очистки."""
    category_col = get_category_column(df_analysis)
    if category_col != "CategoryDelivery" and "CategoryDelivery" in df_analysis.columns:
        category_col = "CategoryDelivery"

    mask_brand = df_analysis["BrandID"].eq(brand_id) & df_analysis[category_col].astype(str).eq(str(category_delivery))
    part = df_analysis.loc[mask_brand].copy()
    removed_mask = mark_removed(part, anomalies).to_numpy()

    before = compute_total_ots_by_day(part).rename(columns={"total_ots": "before"})
    after = compute_total_ots_by_day(part.loc[~removed_mask]).rename(columns={"total_ots": "after"})
    result = before.merge(after, on="researchdate", how="left").fillna({"after": 0})

    plt.figure(figsize=(12, 5))
    plt.plot(result["researchdate"], result["before"], marker="o", label="До очистки")
    plt.plot(result["researchdate"], result["after"], marker="o", label="После очистки")
    plt.title(f"OTS по дням для BrandID={brand_id}, CategoryDelivery={category_delivery}")
    plt.xlabel("Дата")
    plt.ylabel("OTS")
    plt.xticks(rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return result


def print_summary(df_analysis: pd.DataFrame, anomalies: pd.DataFrame, reasons: pd.DataFrame) -> None:
    total_subject_days = df_analysis[["SubjectID", "researchdate"]].drop_duplicates().shape[0]
    removed_subject_days = anomalies.shape[0]
    removed_share = removed_subject_days / total_subject_days if total_subject_days else 0

    removed_mask = mark_removed(df_analysis, anomalies).to_numpy()
    ots_before = df_analysis["Weight"].sum()
    ots_after = df_analysis.loc[~removed_mask, "Weight"].sum()
    ots_keep = ots_after / ots_before if ots_before else 0

    print("Готово.")
    print(f"Найдено триггеров аномалий: {len(reasons)}")
    print(f"Удаляемых пар SubjectID-date: {removed_subject_days}")
    print(f"Доля удаляемых respondent-day: {removed_share:.4%}")
    print(f"Сохранено OTS после очистки: {ots_keep:.2%}")
    print("Файлы сохранены в output/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SoS anomaly respondent-day detection")
    parser.add_argument("--data", default=None, help="Путь к data_train, data_train.zip или папке с parquet")
    parser.add_argument("--output", default="output", help="Папка для результатов")
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_df = load_data(data_path)
    normalized_df, category_col = normalize_input(original_df)
    df_analysis = prepare_analysis_rows(normalized_df, category_col)
    agg = aggregate_daily_ots(df_analysis, category_col)
    anomalies, reasons = detect_anomalies(agg, category_col)
    save_outputs(df_analysis, anomalies, reasons, output_dir, category_col)
    print_summary(df_analysis, anomalies, reasons)


if __name__ == "__main__":
    main()
