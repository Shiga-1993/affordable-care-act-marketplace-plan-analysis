from __future__ import annotations

import argparse
import base64
import html
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
REPORT_DIR = PROJECT_DIR / "outputs" / "report"
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"

SOURCE_CHECKED_DATE = "2026-05-25"
CMS_SOURCE_PAGE = "https://www.cms.gov/marketplace/resources/data/public-use-files"
DATA_IMPORT_NOTE = (
    "The Centers for Medicare & Medicaid Services source page states 2026 Exchange Public Use File data "
    "were last imported by April 28, 2026."
)

PUF_FILES = {
    "plan": {
        "url": "https://download.cms.gov/marketplace-puf/2026/plan-attributes-puf.zip",
        "zip_name": "plan-attributes-puf.zip",
        "inner_name": "Plan_Attributes_PUF.csv",
    },
    "rate": {
        "url": "https://download.cms.gov/marketplace-puf/2026/rate-puf.zip",
        "zip_name": "rate-puf.zip",
        "inner_name": "Rate_PUF.csv",
    },
    "benefits": {
        "url": "https://download.cms.gov/marketplace-puf/2026/benefits-and-cost-sharing-puf.zip",
        "zip_name": "benefits-and-cost-sharing-puf.zip",
        "inner_name": "Benefits_Cost_Sharing_PUF.csv",
    },
}

PLAN_COLS = [
    "BusinessYear",
    "StateCode",
    "IssuerId",
    "IssuerMarketPlaceMarketingName",
    "MarketCoverage",
    "DentalOnlyPlan",
    "StandardComponentId",
    "PlanId",
    "PlanMarketingName",
    "PlanType",
    "MetalLevel",
    "QHPNonQHPTypeId",
    "CSRVariationType",
    "TEHBInnTier1IndividualMOOP",
    "MEHBInnTier1IndividualMOOP",
    "TEHBDedInnTier1Individual",
    "MEHBDedInnTier1Individual",
    "IsHSAEligible",
    "NationalNetwork",
]

RATE_COLS = ["StateCode", "PlanId", "RatingAreaId", "Tobacco", "Age", "IndividualRate"]

BENEFIT_COLS = [
    "StateCode",
    "StandardComponentId",
    "PlanId",
    "BenefitName",
    "CopayInnTier1",
    "CoinsInnTier1",
    "IsCovered",
]

SELECTED_BENEFITS = {
    "Primary Care Visit to Treat an Injury or Illness": "primary_care",
    "Specialist Visit": "specialist",
    "Generic Drugs": "generic_drugs",
    "Emergency Room Services": "emergency_room",
}

METAL_ORDER = ["Catastrophic", "Bronze", "Expanded Bronze", "Silver", "Gold", "Platinum"]
METAL_COLORS = {
    "Catastrophic": "#64748b",
    "Bronze": "#b45309",
    "Expanded Bronze": "#d97706",
    "Silver": "#6b7280",
    "Gold": "#ca8a04",
    "Platinum": "#0891b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 2026 Affordable Care Act Marketplace plan premiums and cost sharing."
    )
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw downloaded zip files in data/raw.")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, FIG_DIR, REPORT_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.8,
            "legend.fontsize": 8.8,
        }
    )


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 100_000:
        return
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def download_pufs() -> dict[str, Path]:
    paths = {}
    for key, meta in PUF_FILES.items():
        path = RAW_DIR / meta["zip_name"]
        print(f"Downloading {key} Public Use File")
        download_file(meta["url"], path)
        paths[key] = path
    return paths


def parse_money(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if "not applicable" in lowered or lowered in {"nan", "none"}:
        return None
    if "no charge" in lowered:
        return 0.0
    match = re.search(r"-?\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_first_copay_amount(value: object) -> Optional[float]:
    return parse_money(value)


def read_zip_csv(zip_path: Path, inner_name: str, usecols: list[str], **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_name) as f:
            return pd.read_csv(f, usecols=usecols, dtype=str, **kwargs)


def load_plan_components(zip_paths: dict[str, Path]) -> tuple[pd.DataFrame, int]:
    plan_raw = read_zip_csv(zip_paths["plan"], PUF_FILES["plan"]["inner_name"], PLAN_COLS)
    raw_rows = len(plan_raw)

    plan = plan_raw.copy()
    standard_on_exchange = plan["CSRVariationType"].fillna("").str.match(r"^Standard .+ On Exchange Plan$")
    filters = (
        (plan["BusinessYear"] == "2026")
        & (plan["MarketCoverage"] == "Individual")
        & (plan["DentalOnlyPlan"] == "No")
        & standard_on_exchange
        & plan["MetalLevel"].isin(METAL_ORDER)
    )
    plan = plan.loc[filters].copy()
    plan["individual_deductible"] = plan["TEHBDedInnTier1Individual"].map(parse_money)
    plan["individual_deductible"] = plan["individual_deductible"].combine_first(
        plan["MEHBDedInnTier1Individual"].map(parse_money)
    )
    plan["individual_moop"] = plan["TEHBInnTier1IndividualMOOP"].map(parse_money)
    plan["individual_moop"] = plan["individual_moop"].combine_first(
        plan["MEHBInnTier1IndividualMOOP"].map(parse_money)
    )
    plan = plan.rename(
        columns={
            "StateCode": "state",
            "IssuerId": "issuer_id",
            "IssuerMarketPlaceMarketingName": "issuer_name",
            "StandardComponentId": "standard_component_id",
            "PlanId": "plan_variant_id",
            "PlanMarketingName": "plan_name",
            "PlanType": "plan_type",
            "MetalLevel": "metal_level",
            "QHPNonQHPTypeId": "qhp_type",
            "CSRVariationType": "csr_variation",
            "IsHSAEligible": "is_hsa_eligible",
            "NationalNetwork": "national_network",
        }
    )
    keep_cols = [
        "state",
        "issuer_id",
        "issuer_name",
        "standard_component_id",
        "plan_variant_id",
        "plan_name",
        "plan_type",
        "metal_level",
        "qhp_type",
        "csr_variation",
        "individual_deductible",
        "individual_moop",
        "is_hsa_eligible",
        "national_network",
    ]
    plan = plan[keep_cols].drop_duplicates("standard_component_id", keep="first").reset_index(drop=True)
    return plan, raw_rows


def load_age40_rates(zip_paths: dict[str, Path]) -> tuple[pd.DataFrame, int]:
    rate_raw = read_zip_csv(zip_paths["rate"], PUF_FILES["rate"]["inner_name"], RATE_COLS)
    raw_rows = len(rate_raw)
    rate = rate_raw.loc[rate_raw["Age"] == "40"].copy()
    rate["age40_premium"] = pd.to_numeric(rate["IndividualRate"], errors="coerce")
    rate = rate.dropna(subset=["age40_premium"])
    rate = (
        rate.groupby(["StateCode", "PlanId", "RatingAreaId"], as_index=False)
        .agg(age40_premium=("age40_premium", "median"), rate_row_count=("age40_premium", "size"))
        .rename(columns={"StateCode": "state", "PlanId": "standard_component_id", "RatingAreaId": "rating_area"})
    )
    return rate, raw_rows


def load_benefit_copays(zip_paths: dict[str, Path], plan_components: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    variant_ids = set(plan_components["plan_variant_id"])
    chunks = []
    raw_rows = 0
    with zipfile.ZipFile(zip_paths["benefits"]) as zf:
        with zf.open(PUF_FILES["benefits"]["inner_name"]) as f:
            for chunk in pd.read_csv(f, usecols=BENEFIT_COLS, dtype=str, chunksize=250_000):
                raw_rows += len(chunk)
                selected = chunk.loc[
                    chunk["PlanId"].isin(variant_ids) & chunk["BenefitName"].isin(SELECTED_BENEFITS)
                ].copy()
                if not selected.empty:
                    chunks.append(selected)

    if not chunks:
        return pd.DataFrame(columns=["plan_variant_id"]), raw_rows

    benefits = pd.concat(chunks, ignore_index=True)
    benefits["benefit_key"] = benefits["BenefitName"].map(SELECTED_BENEFITS)
    benefits["copay_amount"] = benefits["CopayInnTier1"].map(parse_first_copay_amount)
    benefits["covered_flag"] = benefits["IsCovered"].fillna("").str.lower().eq("covered")
    benefit_copays = benefits.pivot_table(
        index="PlanId",
        columns="benefit_key",
        values="copay_amount",
        aggfunc="median",
    ).reset_index()
    benefit_copays = benefit_copays.rename(
        columns={
            "PlanId": "plan_variant_id",
            "primary_care": "primary_care_copay",
            "specialist": "specialist_copay",
            "generic_drugs": "generic_drug_copay",
            "emergency_room": "emergency_room_copay",
        }
    )
    return benefit_copays, raw_rows


def build_offerings(
    plan_components: pd.DataFrame,
    age40_rates: pd.DataFrame,
    benefit_copays: pd.DataFrame,
) -> pd.DataFrame:
    offerings = plan_components.merge(
        age40_rates,
        on=["state", "standard_component_id"],
        how="inner",
    )
    offerings = offerings.merge(benefit_copays, on="plan_variant_id", how="left")
    offerings["metal_level"] = pd.Categorical(offerings["metal_level"], categories=METAL_ORDER, ordered=True)
    offerings = offerings.sort_values(["state", "metal_level", "issuer_name", "plan_name", "rating_area"]).reset_index(
        drop=True
    )
    return offerings


def build_state_summary(offerings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    silver = offerings.loc[offerings["metal_level"].astype(str) == "Silver"].copy()
    lowest_silver = (
        silver.groupby(["state", "rating_area"], as_index=False)["age40_premium"].min().groupby("state")[
            "age40_premium"
        ]
    )
    lowest_silver_median = lowest_silver.median()
    lowest_silver_p10 = lowest_silver.quantile(0.10)
    lowest_silver_p90 = lowest_silver.quantile(0.90)

    for state, group in offerings.groupby("state", observed=False):
        rows.append(
            {
                "state": state,
                "plan_count": group["standard_component_id"].nunique(),
                "plan_rating_area_count": len(group),
                "issuer_count": group["issuer_id"].nunique(),
                "rating_area_count": group["rating_area"].nunique(),
                "median_age40_premium": group["age40_premium"].median(),
                "p10_age40_premium": group["age40_premium"].quantile(0.10),
                "p90_age40_premium": group["age40_premium"].quantile(0.90),
                "median_individual_deductible": group["individual_deductible"].median(),
                "median_individual_moop": group["individual_moop"].median(),
                "lowest_silver_median_age40_premium": lowest_silver_median.get(state, np.nan),
                "lowest_silver_p10_age40_premium": lowest_silver_p10.get(state, np.nan),
                "lowest_silver_p90_age40_premium": lowest_silver_p90.get(state, np.nan),
                "hsa_plan_share": group["is_hsa_eligible"].eq("Yes").mean(),
                "national_network_share": group["national_network"].eq("Yes").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("median_age40_premium").reset_index(drop=True)


def build_metal_summary(offerings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metal, group in offerings.groupby("metal_level", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "metal_level": str(metal),
                "plan_count": group["standard_component_id"].nunique(),
                "plan_rating_area_count": len(group),
                "issuer_count": group["issuer_id"].nunique(),
                "state_count": group["state"].nunique(),
                "median_age40_premium": group["age40_premium"].median(),
                "p10_age40_premium": group["age40_premium"].quantile(0.10),
                "p90_age40_premium": group["age40_premium"].quantile(0.90),
                "median_individual_deductible": group["individual_deductible"].median(),
                "p10_individual_deductible": group["individual_deductible"].quantile(0.10),
                "p90_individual_deductible": group["individual_deductible"].quantile(0.90),
                "median_individual_moop": group["individual_moop"].median(),
                "median_primary_care_copay": group["primary_care_copay"].median(),
                "primary_care_no_charge_share": group["primary_care_copay"].eq(0).mean(),
                "hsa_plan_share": group["is_hsa_eligible"].eq("Yes").mean(),
            }
        )
    frame = pd.DataFrame(rows)
    frame["metal_level"] = pd.Categorical(frame["metal_level"], categories=METAL_ORDER, ordered=True)
    return frame.sort_values("metal_level").reset_index(drop=True)


def build_quality_summary(
    plan_raw_rows: int,
    rate_raw_rows: int,
    benefits_raw_rows: int,
    plan_components: pd.DataFrame,
    age40_rates: pd.DataFrame,
    offerings: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        ("plan_attributes_raw_rows", plan_raw_rows, "Rows read from the Plan Attributes Public Use File."),
        ("rate_raw_rows", rate_raw_rows, "Rows read from the Rate Public Use File."),
        (
            "benefits_cost_sharing_raw_rows",
            benefits_raw_rows,
            "Rows scanned from the Benefits and Cost Sharing Public Use File.",
        ),
        (
            "standard_on_exchange_medical_plans",
            len(plan_components),
            "Individual, non-dental, standard on-exchange plan components retained.",
        ),
        ("age40_rate_rows", len(age40_rates), "Plan-rating-area age-40 premium rows after grouping."),
        ("joined_plan_rating_area_rows", len(offerings), "Plan-rating-area rows joined to plan attributes."),
        ("states", offerings["state"].nunique(), "States represented in the final analysis scope."),
        ("issuers", offerings["issuer_id"].nunique(), "Issuers represented in the final analysis scope."),
        (
            "missing_premium_rows",
            int(offerings["age40_premium"].isna().sum()),
            "Rows missing age-40 individual premium after join.",
        ),
        (
            "missing_deductible_rows",
            int(offerings["individual_deductible"].isna().sum()),
            "Rows missing parsed individual deductible.",
        ),
        (
            "missing_maximum_out_of_pocket_rows",
            int(offerings["individual_moop"].isna().sum()),
            "Rows missing parsed individual maximum out-of-pocket.",
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "note"])


def save_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def ordered_metal_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["metal_level"] = pd.Categorical(data["metal_level"].astype(str), categories=METAL_ORDER, ordered=True)
    return data.sort_values("metal_level")


def plot_premium_distribution_by_metal(offerings: pd.DataFrame) -> Path:
    data = ordered_metal_data(offerings)
    groups = [
        data.loc[data["metal_level"].astype(str) == metal, "age40_premium"].dropna().to_numpy()
        for metal in METAL_ORDER
        if (data["metal_level"].astype(str) == metal).any()
    ]
    labels = [metal for metal in METAL_ORDER if (data["metal_level"].astype(str) == metal).any()]
    positions = np.arange(len(labels)) * 1.15
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    box = ax.boxplot(
        groups,
        vert=False,
        positions=positions,
        widths=0.62,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(METAL_COLORS[label])
        patch.set_alpha(0.66)
    ax.set_xlabel("Age-40 individual monthly premium")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    return save_figure(fig, FIG_DIR / "premium_distribution_by_metal.png")


def plot_premium_deductible_by_metal(metal_summary: pd.DataFrame) -> Path:
    data = ordered_metal_data(metal_summary)
    y = np.arange(len(data))
    colors = [METAL_COLORS[str(m)] for m in data["metal_level"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharey=True)
    axes[0].barh(y, data["median_age40_premium"], color=colors, alpha=0.82)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(data["metal_level"].astype(str))
    axes[0].set_xlabel("Median age-40 monthly premium")
    axes[0].set_ylabel("")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].grid(axis="y", visible=False)

    axes[1].barh(y, data["median_individual_deductible"], color=colors, alpha=0.82)
    axes[1].set_xlabel("Median individual deductible")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].grid(axis="y", visible=False)

    for ax, column, prefix in [
        (axes[0], "median_age40_premium", "$"),
        (axes[1], "median_individual_deductible", "$"),
    ]:
        max_value = float(data[column].max())
        for ypos, value in zip(y, data[column]):
            ax.text(value + max_value * 0.015, ypos, f"{prefix}{value:,.0f}", va="center", fontsize=8.5)
        ax.set_xlim(0, max_value * 1.18)

    fig.tight_layout()
    return save_figure(fig, FIG_DIR / "premium_deductible_by_metal.png")


def plot_state_market_depth(state_summary: pd.DataFrame) -> Path:
    data = state_summary.copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    sizes = 35 + data["plan_rating_area_count"] / data["plan_rating_area_count"].max() * 380
    scatter = ax.scatter(
        data["issuer_count"],
        data["plan_count"],
        s=sizes,
        c=data["median_age40_premium"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.8,
    )
    label_states = set(data.nlargest(4, "plan_count")["state"]).union({"NH"})
    label_offsets = {
        "TX": (8, 8),
        "FL": (8, -3),
        "WI": (8, 8),
        "NC": (8, 8),
        "NH": (8, 10),
    }
    for _, row in data.loc[data["state"].isin(label_states)].iterrows():
        ax.annotate(
            row["state"],
            (row["issuer_count"], row["plan_count"]),
            xytext=label_offsets.get(row["state"], (7, 5)),
            textcoords="offset points",
            fontsize=8.7,
            color="#111827",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.6},
        )
    ax.set_xlabel("Issuer count")
    ax.set_ylabel("Standard on-exchange medical plan count")
    ax.set_ylim(-28, data["plan_count"].max() + 62)
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Median age-40 premium")
    return save_figure(fig, FIG_DIR / "state_market_depth.png")


def plot_state_plan_design_map(state_summary: pd.DataFrame) -> Path:
    data = state_summary.copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    sizes = 35 + data["plan_count"] / data["plan_count"].max() * 360
    ax.scatter(
        data["median_age40_premium"],
        data["median_individual_deductible"],
        s=sizes,
        color="#2563eb",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.8,
    )
    label_states = {"AK", "WV", "WY", "DE", "HI", "OR", "SC"}
    label_offsets = {
        "AK": (8, 8),
        "WV": (8, 8),
        "WY": (8, 8),
        "DE": (8, 8),
        "HI": (8, 8),
        "OR": (8, 8),
        "SC": (8, -16),
    }
    for _, row in data.loc[data["state"].isin(label_states)].iterrows():
        ax.annotate(
            row["state"],
            (row["median_age40_premium"], row["median_individual_deductible"]),
            xytext=label_offsets.get(row["state"], (7, 5)),
            textcoords="offset points",
            fontsize=8.7,
            color="#111827",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.6},
        )
    ax.set_xlabel("Median age-40 monthly premium")
    ax.set_ylabel("Median individual deductible")
    ax.set_ylim(data["median_individual_deductible"].min() - 280, data["median_individual_deductible"].max() + 420)
    ax.grid(alpha=0.2)
    return save_figure(fig, FIG_DIR / "state_plan_design_map.png")


def plot_lowest_silver_by_state(state_summary: pd.DataFrame) -> Path:
    data = state_summary.dropna(subset=["lowest_silver_median_age40_premium"]).sort_values(
        "lowest_silver_median_age40_premium"
    )
    positions = np.arange(len(data)) * 1.12
    fig_height = max(7.8, len(data) * 0.22)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    ax.scatter(data["lowest_silver_median_age40_premium"], positions, s=34, color="#6b7280")
    ax.set_yticks(positions)
    ax.set_yticklabels(data["state"])
    ax.set_xlabel("Median of lowest-cost Silver premium by rating area")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    return save_figure(fig, FIG_DIR / "lowest_silver_by_state.png")


def plot_primary_care_copay_by_metal(offerings: pd.DataFrame) -> Path:
    data = ordered_metal_data(offerings.dropna(subset=["primary_care_copay"]))
    labels = [metal for metal in METAL_ORDER if (data["metal_level"].astype(str) == metal).any()]
    groups = [
        data.loc[data["metal_level"].astype(str) == metal, "primary_care_copay"].to_numpy() for metal in labels
    ]
    positions = np.arange(len(labels)) * 1.15
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    box = ax.boxplot(
        groups,
        vert=False,
        positions=positions,
        widths=0.62,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(METAL_COLORS[label])
        patch.set_alpha(0.66)
    ax.set_xlabel("Parsed in-network primary care copay")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    return save_figure(fig, FIG_DIR / "primary_care_copay_by_metal.png")


def image_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fmt_int(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def fmt_money(value: object, digits: int = 0) -> str:
    if pd.isna(value):
        return ""
    return f"${float(value):,.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def fmt_float(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"


def html_table(frame: pd.DataFrame, columns: list[str], rename: dict[str, str], max_rows: int = 16) -> str:
    display = frame.loc[:, columns].head(max_rows).copy()
    for column in display.columns:
        if column.endswith("_count") or column in {"plan_count", "issuer_count", "rating_area_count"}:
            display[column] = display[column].map(fmt_int)
        elif "premium" in column or "deductible" in column or "moop" in column or "copay" in column:
            display[column] = display[column].map(lambda x: fmt_money(x, 0))
        elif column.endswith("_share"):
            display[column] = display[column].map(fmt_pct)
    display = display.rename(columns=rename)
    return display.to_html(index=False, escape=True, classes="data-table")


FIGURE_CAPTIONS = {
    "premium_distribution_by_metal": (
        "Age-40 gross monthly premium distribution by metal level. Distributions overlap, so metal level is not a simple price ladder."
    ),
    "premium_deductible_by_metal": (
        "Median monthly premium and median deductible by metal level. Lower-premium tiers generally shift more exposure into deductibles."
    ),
    "state_market_depth": (
        "Issuer count and standard medical plan count by state. Bubble size reflects plan-rating-area rows; color reflects median age-40 premium."
    ),
    "state_plan_design_map": (
        "State-level plan design map. X axis is median age-40 premium; Y axis is median individual deductible; bubble size is plan count."
    ),
    "lowest_silver_by_state": (
        "Median lowest-cost Silver premium by rating area, summarized by state. This is a gross premium proxy, not a subsidy-adjusted net premium."
    ),
    "primary_care_copay_by_metal": (
        "Parsed primary care copay by metal level. Text fields such as 'after deductible' are simplified to the first dollar amount."
    ),
}


def render_report(
    offerings: pd.DataFrame,
    state_summary: pd.DataFrame,
    metal_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    figure_paths: list[Path],
    raw_file_sizes: dict[str, int],
) -> Path:
    total_plan_count = offerings["standard_component_id"].nunique()
    state_count = offerings["state"].nunique()
    issuer_count = offerings["issuer_id"].nunique()
    rating_area_count = offerings[["state", "rating_area"]].drop_duplicates().shape[0]

    bronze_like = offerings.loc[offerings["metal_level"].astype(str).isin(["Bronze", "Expanded Bronze"])]
    silver = offerings.loc[offerings["metal_level"].astype(str) == "Silver"]
    gold = offerings.loc[offerings["metal_level"].astype(str) == "Gold"]
    platinum = offerings.loc[offerings["metal_level"].astype(str) == "Platinum"]

    bronze_median_premium = bronze_like["age40_premium"].median()
    bronze_median_deductible = bronze_like["individual_deductible"].median()
    silver_median_premium = silver["age40_premium"].median()
    silver_median_deductible = silver["individual_deductible"].median()
    gold_median_premium = gold["age40_premium"].median()
    gold_median_deductible = gold["individual_deductible"].median()
    platinum_median_premium = platinum["age40_premium"].median()
    platinum_median_deductible = platinum["individual_deductible"].median()

    lowest_silver = state_summary.dropna(subset=["lowest_silver_median_age40_premium"])
    low_lcs = lowest_silver.nsmallest(1, "lowest_silver_median_age40_premium").iloc[0]
    high_lcs = lowest_silver.nlargest(1, "lowest_silver_median_age40_premium").iloc[0]
    broadest_market = state_summary.nlargest(1, "issuer_count").iloc[0]
    thinnest_market = state_summary.nsmallest(1, "issuer_count").iloc[0]
    highest_median_premium_state = state_summary.nlargest(1, "median_age40_premium").iloc[0]
    highest_deductible_state = state_summary.nlargest(1, "median_individual_deductible").iloc[0]

    bronze_p90 = bronze_like["age40_premium"].quantile(0.90)
    silver_p10 = silver["age40_premium"].quantile(0.10)
    gold_p10 = gold["age40_premium"].quantile(0.10)
    overlap_line = (
        f"The 90th percentile Bronze / Expanded Bronze premium is {fmt_money(bronze_p90, 0)}, "
        f"while the 10th percentile Silver premium is {fmt_money(silver_p10, 0)} and the 10th percentile Gold premium is {fmt_money(gold_p10, 0)}."
    )

    figure_html = "\n".join(
        (
            f'<section class="figure"><img src="{image_to_data_uri(path)}" alt="{html.escape(path.stem)}">'
            f'<figcaption>{html.escape(FIGURE_CAPTIONS.get(path.stem, path.stem.replace("_", " ")))}</figcaption>'
            f"</section>"
        )
        for path in figure_paths
    )

    state_table = html_table(
        state_summary.sort_values("median_age40_premium", ascending=False),
        [
            "state",
            "plan_count",
            "issuer_count",
            "rating_area_count",
            "median_age40_premium",
            "median_individual_deductible",
            "lowest_silver_median_age40_premium",
        ],
        {
            "state": "State",
            "plan_count": "Plans",
            "issuer_count": "Issuers",
            "rating_area_count": "Rating areas",
            "median_age40_premium": "Median premium",
            "median_individual_deductible": "Median deductible",
            "lowest_silver_median_age40_premium": "Median lowest Silver",
        },
        max_rows=18,
    )
    metal_table = html_table(
        metal_summary,
        [
            "metal_level",
            "plan_count",
            "issuer_count",
            "state_count",
            "median_age40_premium",
            "median_individual_deductible",
            "median_individual_moop",
            "median_primary_care_copay",
            "hsa_plan_share",
        ],
        {
            "metal_level": "Metal",
            "plan_count": "Plans",
            "issuer_count": "Issuers",
            "state_count": "States",
            "median_age40_premium": "Median premium",
            "median_individual_deductible": "Median deductible",
            "median_individual_moop": "Median maximum out-of-pocket",
            "median_primary_care_copay": "Median primary care copay",
            "hsa_plan_share": "Health Savings Account-eligible share",
        },
        max_rows=8,
    )
    quality_labels = {
        "plan_attributes_raw_rows": "Plan Attributes source rows",
        "rate_raw_rows": "Rate source rows",
        "benefits_cost_sharing_raw_rows": "Benefits and Cost Sharing source rows",
        "standard_on_exchange_medical_plans": "Standard on-exchange medical plans",
        "age40_rate_rows": "Age-40 rate rows",
        "joined_plan_rating_area_rows": "Joined plan-rating-area rows",
        "states": "States",
        "issuers": "Issuers",
        "missing_premium_rows": "Missing premium rows",
        "missing_deductible_rows": "Missing deductible rows",
        "missing_moop_rows": "Missing maximum out-of-pocket rows",
        "missing_maximum_out_of_pocket_rows": "Missing maximum out-of-pocket rows",
    }
    quality_display = quality_summary.copy()
    quality_display["check"] = quality_display["check"].map(lambda value: quality_labels.get(str(value), str(value)))
    quality_display = quality_display.rename(columns={"check": "Check", "value": "Value", "note": "Note"})
    quality_table = quality_display.to_html(index=False, escape=True, classes="data-table")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2026 Affordable Care Act Marketplace Plan Premium and Cost-Sharing Variation Analysis</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; line-height: 1.55; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ font-size: 32px; line-height: 1.15; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 22px 0 10px; font-size: 17px; }}
    p, li {{ font-size: 16px; }}
    a {{ color: #1d4ed8; }}
    .meta {{ color: #4b5563; margin-bottom: 22px; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 22px 0 26px; }}
    .metric {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .metric span {{ display: block; color: #6b7280; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 22px; }}
    .note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 14px; }}
    .figure {{ margin: 24px 0; }}
    .figure img {{ display: block; width: 100%; height: auto; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
    .figure figcaption {{ margin: 8px 2px 0; color: #4b5563; font-size: 14px; }}
    .data-table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; margin: 12px 0 18px; font-size: 14px; }}
    .data-table th, .data-table td {{ padding: 8px 9px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .data-table th {{ background: #f3f4f6; font-weight: 650; }}
    code {{ background: #e5e7eb; padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 720px) {{ .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 460px) {{ .summary {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>2026 Affordable Care Act Marketplace Plan Premium and Cost-Sharing Variation Analysis</h1>
  <p class="meta">Centers for Medicare &amp; Medicaid Services Health Insurance Exchange Public Use Files, plan year 2026. Source checked {SOURCE_CHECKED_DATE}. {DATA_IMPORT_NOTE}</p>

  <div class="summary">
    <div class="metric"><span>Standard medical plans</span><strong>{fmt_int(total_plan_count)}</strong></div>
    <div class="metric"><span>Plan-rating-area rows</span><strong>{fmt_int(len(offerings))}</strong></div>
    <div class="metric"><span>States</span><strong>{fmt_int(state_count)}</strong></div>
    <div class="metric"><span>Issuers</span><strong>{fmt_int(issuer_count)}</strong></div>
    <div class="metric"><span>Rating areas</span><strong>{fmt_int(rating_area_count)}</strong></div>
    <div class="metric"><span>Median age-40 premium</span><strong>{fmt_money(offerings["age40_premium"].median(), 0)}</strong></div>
  </div>

  <h2>Background</h2>
  <p>Affordable Care Act Marketplace plan data is not just a list of prices. A plan menu combines premium, deductible, maximum out-of-pocket, metal level, issuer availability, and geography. This report turns the public Centers for Medicare &amp; Medicaid Services Exchange Public Use Files into a compact view of those tradeoffs.</p>

  <h2>Objective</h2>
  <p>The goal is to compare plan choice across states and metal levels without confusing plan availability with enrollment or actual household spending. The core question is: where do gross premiums, cost-sharing exposure, and issuer/plan availability move together?</p>

  <h2>Terms Used</h2>
  <ul>
    <li>Affordable Care Act Marketplace: the health insurance marketplace created under the Affordable Care Act, where people can compare and buy individual-market plans.</li>
    <li>Centers for Medicare &amp; Medicaid Services: the federal agency that publishes the exchange data used in this report.</li>
    <li>Public Use File: a downloadable dataset released for public analysis.</li>
    <li>Maximum out-of-pocket: the plan-design field for the annual cap on covered in-network cost sharing for an individual, before exceptions and detailed benefit rules.</li>
    <li>Health Savings Account-eligible plan: a plan design that can be paired with a Health Savings Account under the relevant eligibility rules.</li>
  </ul>

  <h2>Data and Scope</h2>
  <ul>
    <li>Files used: Plan Attributes Public Use File, Rate Public Use File, and Benefits and Cost Sharing Public Use File.</li>
    <li>Scope: individual-market, non-dental, standard on-exchange medical plan variants.</li>
    <li>Premium proxy: age-40 individual monthly <code>IndividualRate</code>, summarized at the plan-rating-area level.</li>
    <li>Cost-sharing proxy: in-network tier 1 individual deductible and maximum out-of-pocket from Plan Attributes.</li>
    <li>Primary care copay proxy: parsed dollar amount for <code>Primary Care Visit to Treat an Injury or Illness</code>.</li>
  </ul>

  <p class="note">Important reading rule: one row here is a plan-rating-area offering. It is not weighted by enrollment, county population, plan selection, subsidy eligibility, or household income.</p>

  <h2>Key Findings</h2>
  <ul>
    <li>The analysis keeps {fmt_int(total_plan_count)} standard on-exchange medical plans across {fmt_int(state_count)} states, {fmt_int(issuer_count)} issuers, and {fmt_int(rating_area_count)} state-rating-area combinations.</li>
    <li>Metal level is useful, but it is not a simple price ladder. {overlap_line}</li>
    <li>Bronze / Expanded Bronze plans have a median age-40 premium of {fmt_money(bronze_median_premium, 0)} and a median deductible of {fmt_money(bronze_median_deductible, 0)}. Silver is {fmt_money(silver_median_premium, 0)} premium and {fmt_money(silver_median_deductible, 0)} deductible. Gold is {fmt_money(gold_median_premium, 0)} premium and {fmt_money(gold_median_deductible, 0)} deductible.</li>
    <li>Platinum is rare in this scope: median premium {fmt_money(platinum_median_premium, 0)} and median deductible {fmt_money(platinum_median_deductible, 0)}, with far fewer plan offerings than Silver or Bronze tiers.</li>
    <li>Market depth differs sharply by state. {html.escape(str(broadest_market.state))} has the most issuers in this scope ({fmt_int(broadest_market.issuer_count)}), while {html.escape(str(thinnest_market.state))} has only {fmt_int(thinnest_market.issuer_count)}.</li>
    <li>The highest median age-40 premium state is {html.escape(str(highest_median_premium_state.state))} at {fmt_money(highest_median_premium_state.median_age40_premium, 0)}. The highest median deductible state is {html.escape(str(highest_deductible_state.state))} at {fmt_money(highest_deductible_state.median_individual_deductible, 0)}.</li>
    <li>Lowest-cost Silver also varies widely. The lowest state median is {html.escape(str(low_lcs.state))} at {fmt_money(low_lcs.lowest_silver_median_age40_premium, 0)}, while the highest is {html.escape(str(high_lcs.state))} at {fmt_money(high_lcs.lowest_silver_median_age40_premium, 0)}.</li>
    <li>The punchline is that a consumer-facing plan menu has at least three dimensions: monthly premium, potential cost-sharing exposure, and how many issuers/plans are actually available in the rating area.</li>
  </ul>

  <h2>Figures</h2>
  {figure_html}

  <h2>Summary Tables</h2>
  <h3>Metal-level summary</h3>
  {metal_table}
  <h3>State summary, sorted by median premium</h3>
  {state_table}

  <h2>Data Quality Checks</h2>
  {quality_table}

  <h2>Limitations</h2>
  <ul>
    <li>This report describes plan availability, not enrollment or plan selections.</li>
    <li>Premiums are gross premiums and do not include advance premium tax credits or other subsidy effects.</li>
    <li>Rows are not weighted by county population, enrollment, household income, morbidity, or issuer market share.</li>
    <li>Some State-based Exchanges are excluded when they do not rely on the federal platform.</li>
    <li>Deductible, maximum out-of-pocket, and copay values are simplified plan-design fields. They do not describe all benefit rules, network restrictions, formularies, or actual out-of-pocket spending.</li>
    <li>This is not medical, legal, compliance, insurance-purchasing, or policy advice.</li>
  </ul>

  <h2>Reproducibility</h2>
  <p>Run <code>python3 run_marketplace_plan_analysis.py</code> from the project directory. The script downloads the 2026 Centers for Medicare &amp; Medicaid Services Exchange Public Use File zip files, creates plan-rating-area summary tables, writes figures, renders this standalone web report, and removes raw zip files unless <code>--keep-raw</code> is passed.</p>

  <h2>Sources</h2>
  <ul>
    <li><a href="{CMS_SOURCE_PAGE}">Centers for Medicare &amp; Medicaid Services Health Insurance Exchange Public Use Files</a></li>
    <li><a href="{PUF_FILES["plan"]["url"]}">2026 Plan Attributes Public Use File</a></li>
    <li><a href="{PUF_FILES["rate"]["url"]}">2026 Rate Public Use File</a></li>
    <li><a href="{PUF_FILES["benefits"]["url"]}">2026 Benefits and Cost Sharing Public Use File</a></li>
  </ul>
</main>
</body>
</html>
"""
    report_path = REPORT_DIR / "review_report.html"
    report_path.write_text(html_text, encoding="utf-8")

    metadata = {
        "source_checked_date": SOURCE_CHECKED_DATE,
        "source_page": CMS_SOURCE_PAGE,
        "data_import_note": DATA_IMPORT_NOTE,
        "raw_file_sizes": raw_file_sizes,
        "report_path": str(report_path),
        "scope": "Individual, non-dental, standard on-exchange medical plan variants; age-40 gross premium.",
    }
    (TABLE_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report_path


def cleanup_raw_files() -> None:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_plot_style()

    zip_paths = download_pufs()
    raw_file_sizes = {key: path.stat().st_size for key, path in zip_paths.items()}

    print("Loading plan attributes")
    plan_components, plan_raw_rows = load_plan_components(zip_paths)
    print(f"Standard on-exchange medical plans: {len(plan_components):,}")

    print("Loading age-40 rates")
    age40_rates, rate_raw_rows = load_age40_rates(zip_paths)
    print(f"Age-40 plan-rating-area rate rows: {len(age40_rates):,}")

    print("Loading selected benefit copays")
    benefit_copays, benefits_raw_rows = load_benefit_copays(zip_paths, plan_components)
    print(f"Selected benefit rows by plan variant: {len(benefit_copays):,}")

    print("Building analysis tables")
    offerings = build_offerings(plan_components, age40_rates, benefit_copays)
    state_summary = build_state_summary(offerings)
    metal_summary = build_metal_summary(offerings)
    quality_summary = build_quality_summary(
        plan_raw_rows,
        rate_raw_rows,
        benefits_raw_rows,
        plan_components,
        age40_rates,
        offerings,
    )

    save_table("marketplace_plan_offerings", offerings)
    save_table("state_summary", state_summary)
    save_table("metal_summary", metal_summary)
    save_table("quality_summary", quality_summary)
    offerings.to_csv(PROCESSED_DIR / "marketplace_plan_offerings.csv", index=False)

    print("Creating figures")
    figure_paths = [
        plot_premium_distribution_by_metal(offerings),
        plot_premium_deductible_by_metal(metal_summary),
        plot_state_market_depth(state_summary),
        plot_state_plan_design_map(state_summary),
        plot_lowest_silver_by_state(state_summary),
        plot_primary_care_copay_by_metal(offerings),
    ]

    print("Rendering report")
    report_path = render_report(
        offerings,
        state_summary,
        metal_summary,
        quality_summary,
        figure_paths,
        raw_file_sizes,
    )

    if not args.keep_raw:
        cleanup_raw_files()

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
