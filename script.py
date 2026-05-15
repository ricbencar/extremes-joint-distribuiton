# -*- coding: utf-8 -*-
"""
================================================================================
Environmental Contour Analysis for ERA5 Wave Data
================================================================================

Purpose
-------
This script performs environmental contour analysis for oceanographic time-series
using significant wave height and mean wave period. It fits a Virocon joint
probability model to Hs and MWP, calculates environmental contours for selected
return periods, and exports a PDF report, individual PNG plots and a detailed
text report.

Input CSV
---------
The script expects an input file named ``input.csv`` with the following columns:

    datetime,swh,mwp,mwd,wind,dwi,u10,v10

Required columns for the contour analysis are:

    datetime  Timestamp.
    swh       Significant wave height, Hs, in metres.
    mwp       Mean wave period, MWP, in seconds.
    mwd       Mean wave direction in degrees, required when sector analysis is
              enabled.

The columns ``wind``, ``dwi``, ``u10`` and ``v10`` may exist in the CSV and are
left unused by this contour analysis.

Period convention
-----------------
The joint distribution model is fitted using MWP directly as the wave-period
variable. For plotting and engineering readability, the contour charts retain the
peak-period convention, using:

    Peak Wave Period, Tp = 1.2 x Mean Wave Period, MWP

Outputs
-------
    contours.pdf   Multi-page PDF report with one plot per analysis case.
    plots/         Directory with high-resolution PNG plots.
    results.txt    Text report with configuration, fitted model details and
                   contour summaries.

Requirements
------------
    pip install pandas numpy matplotlib virocon scipy

Run
---
    python stats_era5_data.py

References
----------
    Virocon documentation: https://virocon.readthedocs.io/
    Haselsteiner, A.F., Sander, A., Ohlendorf, J.H., Thoben, K.D. (2020).
    Global hierarchical models for wind and wave contours: physical
    interpretations of the dependence functions. OMAE 2020.
"""

from __future__ import annotations

import copy
import os
import re
import sys
import traceback
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from virocon import (
    DirectSamplingContour,
    GlobalHierarchicalModel,
    HighestDensityContour,
    IFORMContour,
    ISORMContour,
    calculate_alpha,
    get_OMAE2020_Hs_Tz,
)

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# File and CSV columns.
INPUT_FILE = "input.csv"
TIME_COL = "datetime"
HS_CSV_COL_NAME = "swh"
MWP_CSV_COL_NAME = "mwp"
MWD_CSV_COL_NAME = "mwd"

# Peak-period plotting convention.
TP_FROM_MWP_RATIO = 1.2
TP_PLOT_COL_NAME = "_tp_from_mwp"

# Directional-sector analysis.
PERFORM_SECTOR_ANALYSIS = True
START_DIR_DEGREES = 0
SECTOR_WIDTH_DEGREES = 30

# Contour parameters.
SEA_STATE_DURATION_HOURS = 3.0
RETURN_PERIODS_YEARS = [1, 5, 10, 25, 50, 100, 250]
MIN_SAMPLES_FOR_FIT = 200

# Plot and output configuration.
SWAP_AXES_CONTOUR_PLOT = True
CONTOUR_METHOD_TYPE = "IFORM"  # Options: IFORM, ISORM, HighestDensity, DirectSampling.
PDF_OUTPUT_FILE = "contours.pdf"
PNG_OUTPUT_DIR = "plots"
RESULTS_TXT_FILE = "results.txt"

# Robust fitting strategies for the OMAE 2020 hierarchical model.
COMPLEX_MODEL_FITTING_STRATEGIES = [
    {"id": "default", "description": "Default parameters"},
    {
        "id": "perturbed_p0",
        "description": "Perturbed initial guesses",
        "p0_factor_range": (0.5, 1.5),
        "default_p0_values": {"mu": [1.0, 0.5, 0.1], "sigma": [0.2, 0.1, 0.5]},
    },
    {
        "id": "wide_bounds",
        "description": "Wider parameter bounds",
        "bounds_multiplier_lower": 0.1,
        "bounds_multiplier_upper": 10.0,
        "min_bound_val": 1.0e-9,
    },
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def normalize_angle_degrees(angle: float) -> float:
    """Return an angle normalized to the [0, 360) interval."""
    return float(angle) % 360.0



def format_sector_angle(angle: float) -> str:
    """Format sector edge angles for compact plot and report titles."""
    angle_norm = normalize_angle_degrees(angle)
    rounded = round(angle_norm)
    if np.isclose(angle_norm, rounded):
        return str(int(rounded) % 360)
    return f"{angle_norm:.6g}"



def build_direction_sectors(start_dir_degrees: float, sector_width_degrees: float) -> list[dict[str, Any]]:
    """
    Build directional sectors starting at ``start_dir_degrees`` and covering the
    full 360-degree circle.

    The lower sector limit is inclusive and the upper limit is exclusive. When
    360 degrees is not an exact multiple of the sector width, the final sector is
    shortened so that the full circle is covered without overlap or gaps.
    """
    width = float(sector_width_degrees)
    if width <= 0.0:
        raise ValueError("SECTOR_WIDTH_DEGREES must be greater than zero.")
    if width > 360.0:
        raise ValueError("SECTOR_WIDTH_DEGREES must not exceed 360 degrees.")

    start = normalize_angle_degrees(start_dir_degrees)

    if np.isclose(width, 360.0):
        return [
            {
                "lower": start,
                "upper": start,
                "span": 360.0,
                "wraps": False,
                "is_full_circle": True,
                "title": f"Sector {format_sector_angle(start)}-{format_sector_angle(start)} deg",
            }
        ]

    sectors: list[dict[str, Any]] = []
    covered = 0.0
    tol = 1.0e-12

    while covered < 360.0 - tol:
        span = min(width, 360.0 - covered)
        lower_abs = start + covered
        upper_abs = start + covered + span
        lower = normalize_angle_degrees(lower_abs)
        upper = normalize_angle_degrees(upper_abs)
        wraps = upper_abs > 360.0 + tol or (lower > upper and not np.isclose(lower, upper))

        sectors.append(
            {
                "lower": lower,
                "upper": upper,
                "span": span,
                "wraps": wraps,
                "is_full_circle": False,
                "title": f"Sector {format_sector_angle(lower)}-{format_sector_angle(upper)} deg",
            }
        )
        covered += span

    return sectors



def get_sector_mask(
    direction_series: pd.Series,
    lower_bound: float,
    upper_bound: float,
    is_full_circle: bool = False,
) -> pd.Series:
    """Return a boolean mask for a directional sector."""
    if is_full_circle:
        return pd.Series(True, index=direction_series.index)

    directions = np.mod(direction_series.astype(float), 360.0)
    tol = 1.0e-12

    if lower_bound < upper_bound:
        return (directions >= (lower_bound - tol)) & (directions < (upper_bound - tol))

    return (directions >= (lower_bound - tol)) | (directions < (upper_bound - tol))



def format_summary_table(rows: list[list[str]]) -> str:
    """Format a list of rows as a plain-text aligned table."""
    if not rows or len(rows) <= 1:
        return "No summary data was generated.\n"

    widths = [max(len(str(item)) for item in col) for col in zip(*rows)]
    header, data = rows[0], rows[1:]
    header_line = " | ".join(f"{heading:<{width}}" for heading, width in zip(header, widths))
    separator_line = "-+-".join("-" * width for width in widths)
    data_lines = [
        " | ".join(f"{str(item):<{width}}" for item, width in zip(row, widths))
        for row in data
    ]
    return "\n".join([header_line, separator_line] + data_lines) + "\n"



def sanitize_filename(text: str) -> str:
    """Convert analysis labels into safe file-name fragments."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", text.replace(" ", "_")).lower()



def add_peak_period_plot_column(data: pd.DataFrame) -> pd.DataFrame:
    """Add the derived peak-period column used only for charts and reporting."""
    output = data.copy()
    output[TP_PLOT_COL_NAME] = output[MWP_CSV_COL_NAME].astype(float) * TP_FROM_MWP_RATIO
    return output



def create_dummy_input_file(path: str) -> None:
    """Create a synthetic ERA5-like input file for a basic executable test."""
    num_points = 30 * 365 * 24
    dates = pd.date_range(start="1990-01-01", periods=num_points, freq="h")

    hs = 0.1 + 2.0 * np.random.weibull(1.5, size=num_points)
    hs = np.clip(hs, 0.01, 25.0)

    mu_log_mwp = np.log(4.0) + 0.5 * np.log(np.maximum(hs, 0.1))
    sigma_log_mwp = np.maximum(0.01, 0.1 + 0.2 / np.sqrt(np.maximum(hs, 0.1)))
    mwp = np.exp(np.random.normal(loc=mu_log_mwp, scale=sigma_log_mwp))
    mwp = np.clip(mwp, 1.0, 30.0)

    mwd_1 = np.random.normal(180.0, 45.0, size=num_points)
    mwd_2 = np.random.normal(315.0, 45.0, size=num_points)
    mwd_mix = np.random.choice([0, 1], size=num_points, p=[0.6, 0.4])
    mwd = np.where(mwd_mix == 0, mwd_1, mwd_2) % 360.0

    wind_speed = np.random.weibull(2.0, size=num_points) * 8.0
    dwi = (mwd + np.random.normal(0.0, 20.0, size=num_points)) % 360.0
    u10 = -wind_speed * np.sin(np.deg2rad(dwi))
    v10 = -wind_speed * np.cos(np.deg2rad(dwi))

    dummy_df = pd.DataFrame(
        {
            TIME_COL: dates,
            HS_CSV_COL_NAME: hs,
            MWP_CSV_COL_NAME: mwp,
            MWD_CSV_COL_NAME: mwd,
            "wind": wind_speed,
            "dwi": dwi,
            "u10": u10,
            "v10": v10,
        }
    )
    dummy_df.to_csv(path, index=False)



def load_input_data() -> pd.DataFrame:
    """Load and validate the input CSV file."""
    required_cols = [TIME_COL, HS_CSV_COL_NAME, MWP_CSV_COL_NAME]
    if PERFORM_SECTOR_ANALYSIS:
        required_cols.append(MWD_CSV_COL_NAME)

    try:
        df = pd.read_csv(
            INPUT_FILE,
            usecols=required_cols,
            parse_dates=[TIME_COL],
            index_col=TIME_COL,
        )
    except FileNotFoundError:
        if INPUT_FILE == "input.csv" and not os.path.exists(INPUT_FILE):
            print(f"Input file '{INPUT_FILE}' not found. Creating a synthetic example file.")
            create_dummy_input_file(INPUT_FILE)
            print(f"Synthetic '{INPUT_FILE}' created. Replace it with project data and run again.")
        raise

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Column '{TIME_COL}' could not be parsed as a DatetimeIndex.")

    missing_cols = [column for column in required_cols if column not in df.columns and column != TIME_COL]
    if missing_cols:
        raise ValueError(f"Missing required columns in '{INPUT_FILE}': {', '.join(missing_cols)}")

    key_cols = [HS_CSV_COL_NAME, MWP_CSV_COL_NAME]
    if PERFORM_SECTOR_ANALYSIS:
        key_cols.append(MWD_CSV_COL_NAME)

    df = df.dropna(subset=key_cols).copy()
    if df.empty:
        raise ValueError("Input data are empty after removing rows with missing key values.")

    df[HS_CSV_COL_NAME] = pd.to_numeric(df[HS_CSV_COL_NAME], errors="coerce")
    df[MWP_CSV_COL_NAME] = pd.to_numeric(df[MWP_CSV_COL_NAME], errors="coerce")
    if PERFORM_SECTOR_ANALYSIS:
        df[MWD_CSV_COL_NAME] = pd.to_numeric(df[MWD_CSV_COL_NAME], errors="coerce")

    df = df.dropna(subset=key_cols).copy()
    if df.empty:
        raise ValueError("Input data are empty after numeric conversion of key columns.")

    return add_peak_period_plot_column(df)


# =============================================================================
# MODEL FITTING AND CONTOUR ANALYSIS
# =============================================================================


def apply_fitting_strategy(strategy_config: dict[str, Any], fit_descriptions: list[dict[str, Any]]) -> None:
    """Apply a robustness strategy to a copy of Virocon fit descriptions."""
    if strategy_config["id"] == "perturbed_p0":
        if len(fit_descriptions) <= 1 or not fit_descriptions[1] or "fit_specs" not in fit_descriptions[1]:
            return

        for spec_idx, spec in enumerate(fit_descriptions[1]["fit_specs"]):
            if not spec:
                continue

            param_group_name = "mu" if spec_idx == 0 else "sigma"
            p0_to_use = spec.get("p0")
            if p0_to_use is None:
                p0_to_use = strategy_config.get("default_p0_values", {}).get(param_group_name, [1.0, 1.0, 1.0])
            if not isinstance(p0_to_use, list):
                p0_to_use = list(p0_to_use)

            p_range = strategy_config["p0_factor_range"]
            perturbation = np.random.uniform(p_range[0], p_range[1], size=len(p0_to_use))
            spec["p0"] = [float(p) * float(factor) for p, factor in zip(p0_to_use, perturbation)]

    elif strategy_config["id"] == "wide_bounds":
        if len(fit_descriptions) <= 1 or not fit_descriptions[1] or "fit_specs" not in fit_descriptions[1]:
            return

        for spec in fit_descriptions[1]["fit_specs"]:
            if not spec or "bounds" not in spec or spec["bounds"] is None:
                continue

            original_lower, original_upper = spec["bounds"]
            lower_bounds = np.maximum(
                np.asarray(original_lower, dtype=float) * strategy_config["bounds_multiplier_lower"],
                strategy_config["min_bound_val"],
            )
            upper_bounds = np.asarray(original_upper, dtype=float) * strategy_config["bounds_multiplier_upper"]
            upper_bounds = np.maximum(upper_bounds, lower_bounds + strategy_config["min_bound_val"])
            spec["bounds"] = (lower_bounds.tolist(), upper_bounds.tolist())



def fit_joint_model(
    virocon_input_data: pd.DataFrame,
    base_model_dist_descriptions: list[dict[str, Any]],
    base_fit_descriptions: list[dict[str, Any]],
    result_lines: list[str],
    analysis_title: str,
) -> tuple[GlobalHierarchicalModel | None, str]:
    """Fit the Virocon joint model, with a simplified fallback model if needed."""
    fit_model: GlobalHierarchicalModel | None = None
    last_fit_exception: Exception | None = None
    model_description_for_log = "OMAE 2020 hierarchical model fitted to Hs and MWP"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            from scipy.optimize import OptimizeWarning

            warnings.simplefilter("ignore", category=OptimizeWarning)
        except Exception:
            pass

        for attempt_index, strategy_config in enumerate(COMPLEX_MODEL_FITTING_STRATEGIES, start=1):
            print(
                f"Fitting model for '{analysis_title}' "
                f"({attempt_index}/{len(COMPLEX_MODEL_FITTING_STRATEGIES)}: "
                f"{strategy_config['description']})..."
            )

            current_fit_descriptions = copy.deepcopy(base_fit_descriptions)
            temp_fit_model = GlobalHierarchicalModel(copy.deepcopy(base_model_dist_descriptions))

            try:
                apply_fitting_strategy(strategy_config, current_fit_descriptions)
                temp_fit_model.fit(virocon_input_data, fit_descriptions=current_fit_descriptions)
                fit_model = temp_fit_model
                result_lines.append(f"# Fit successful with strategy: {strategy_config['id']}\n")
                return fit_model, model_description_for_log
            except Exception as exc:
                last_fit_exception = exc
                result_lines.append(f"# Fit attempt {attempt_index} ({strategy_config['id']}) failed: {exc}\n")
                print(f"Fit attempt failed for '{analysis_title}': {exc}")

        warning_msg = (
            f"Full hierarchical model fitting failed for '{analysis_title}'. "
            f"Last error: {last_fit_exception}. Using simplified independent fallback model."
        )
        print(warning_msg)
        result_lines.append(f"\n# {warning_msg}\n")
        model_description_for_log = "Simplified independent fallback model fitted to Hs and MWP"

        try:
            simplified_dist_descriptions = []

            hs_desc = copy.deepcopy(base_model_dist_descriptions[0])
            hs_desc["parameters"] = {name: None for name in hs_desc.get("parameters", {}).keys()}
            simplified_dist_descriptions.append(hs_desc)

            if len(base_model_dist_descriptions) < 2:
                raise ValueError("Base Virocon model has fewer than two dimensions.")

            period_desc = copy.deepcopy(base_model_dist_descriptions[1])
            period_desc.pop("conditional_on", None)
            period_desc["parameters"] = {
                name: None for name in base_model_dist_descriptions[1].get("parameters", {}).keys()
            }
            simplified_dist_descriptions.append(period_desc)

            fit_model = GlobalHierarchicalModel(simplified_dist_descriptions)
            fit_model.fit(virocon_input_data, fit_descriptions=[{"method": "mle"}, {"method": "mle"}])
            result_lines.append("# Simplified fallback model fitting successful.\n")
            return fit_model, model_description_for_log
        except Exception as exc:
            error_msg = f"Simplified fallback model failed for '{analysis_title}': {exc}"
            print(error_msg)
            result_lines.append(f"\n# CRITICAL ERROR: {error_msg}\n")
            return None, model_description_for_log



def perform_analysis_for_dataset(
    data: pd.DataFrame,
    base_model_dist_descriptions: list[dict[str, Any]],
    base_fit_descriptions: list[dict[str, Any]],
    result_lines: list[str],
    summary_table_rows: list[list[str]],
    analysis_title: str,
) -> plt.Figure | None:
    """Fit the joint model, calculate contours and generate one plot."""
    if data.shape[0] < MIN_SAMPLES_FOR_FIT:
        message = (
            f"Skipping '{analysis_title}': insufficient data "
            f"({data.shape[0]} samples; required {MIN_SAMPLES_FOR_FIT})."
        )
        print(message)
        result_lines.append(f"\n# {message}\n")
        return None

    print(f"\n--- Starting analysis: {analysis_title} ---")
    result_lines.append(f"\n{'=' * 80}\n# ANALYSIS: {analysis_title}\n{'=' * 80}\n")
    result_lines.append(f"Number of data points: {data.shape[0]}\n")

    hs_col_virocon = "significant_wave_height"
    period_col_virocon = "zero_upcrossing_period"

    analysis_data = data.copy()
    analysis_data[period_col_virocon] = analysis_data[MWP_CSV_COL_NAME].astype(float)
    analysis_data.rename(columns={HS_CSV_COL_NAME: hs_col_virocon}, inplace=True)

    min_positive_val = 1.0e-6
    original_count = analysis_data.shape[0]
    analysis_data = analysis_data[
        (analysis_data[hs_col_virocon] > min_positive_val)
        & (analysis_data[period_col_virocon] > min_positive_val)
    ].copy()

    filtered_count = analysis_data.shape[0]
    if filtered_count < original_count:
        removed_count = original_count - filtered_count
        print(f"Filtered out {removed_count} rows with non-positive Hs or MWP.")
        result_lines.append(f"# Filtered out {removed_count} rows with non-positive Hs or MWP.\n")

    if analysis_data.shape[0] < MIN_SAMPLES_FOR_FIT:
        message = (
            f"Skipping '{analysis_title}': insufficient positive data "
            f"({analysis_data.shape[0]} samples after filtering; required {MIN_SAMPLES_FOR_FIT})."
        )
        print(message)
        result_lines.append(f"\n# {message}\n")
        return None

    virocon_input_data = analysis_data[[hs_col_virocon, period_col_virocon]]

    fit_model, model_description = fit_joint_model(
        virocon_input_data=virocon_input_data,
        base_model_dist_descriptions=base_model_dist_descriptions,
        base_fit_descriptions=base_fit_descriptions,
        result_lines=result_lines,
        analysis_title=analysis_title,
    )

    if fit_model is None:
        return None

    result_lines.append(f"\n--- Fitted Model Parameters ({model_description}) ---\n")
    result_lines.append(str(fit_model))
    result_lines.append("\n-----------------------------\n")

    try:
        print(f"Calculating contours for '{analysis_title}'...")
        fig_contour, ax_contour = plt.subplots(figsize=(10, 8))

        x_scatter_col = TP_PLOT_COL_NAME if SWAP_AXES_CONTOUR_PLOT else HS_CSV_COL_NAME
        y_scatter_col = HS_CSV_COL_NAME if SWAP_AXES_CONTOUR_PLOT else TP_PLOT_COL_NAME

        ax_contour.scatter(
            data[x_scatter_col],
            data[y_scatter_col],
            alpha=0.1,
            s=5,
            color="black",
            label="All data points",
        )

        table_data_for_plot: list[list[str]] = []
        result_lines.append("\n--- Environmental Contour Results ---\n")
        header = (
            f"{'Return Period':<15} {'Max Hs (m)':<15} "
            f"{'MWP @ Max Hs (s)':<20} {'Tp @ Max Hs (s)':<20}"
        )
        result_lines.append(header)
        result_lines.append("-" * len(header))

        contour_classes = {
            "IFORM": IFORMContour,
            "ISORM": ISORMContour,
            "HighestDensity": HighestDensityContour,
            "DirectSampling": DirectSamplingContour,
        }
        ContourClass = contour_classes.get(CONTOUR_METHOD_TYPE)
        if ContourClass is None:
            raise ValueError(f"Unsupported CONTOUR_METHOD_TYPE: {CONTOUR_METHOD_TYPE}")

        for rp_years in RETURN_PERIODS_YEARS:
            alpha = calculate_alpha(SEA_STATE_DURATION_HOURS, rp_years)

            try:
                contour_obj = ContourClass(fit_model, alpha)
                contour_hs_mwp = np.asarray(contour_obj.coordinates, dtype=float)

                if contour_hs_mwp.size == 0:
                    result_lines.append(f"# Warning: no contour data for {rp_years}-year return period.\n")
                    continue

                contour_hs_tp = contour_hs_mwp.copy()
                contour_hs_tp[:, 1] = contour_hs_mwp[:, 1] * TP_FROM_MWP_RATIO

                x_plot_coords = contour_hs_tp[:, 1] if SWAP_AXES_CONTOUR_PLOT else contour_hs_tp[:, 0]
                y_plot_coords = contour_hs_tp[:, 0] if SWAP_AXES_CONTOUR_PLOT else contour_hs_tp[:, 1]

                line, = ax_contour.plot(x_plot_coords, y_plot_coords, label=f"{rp_years} yr")

                idx_max_hs = int(np.argmax(contour_hs_mwp[:, 0]))
                hs_max_val = float(contour_hs_mwp[idx_max_hs, 0])
                mwp_at_max_hs_val = float(contour_hs_mwp[idx_max_hs, 1])
                tp_at_max_hs_val = mwp_at_max_hs_val * TP_FROM_MWP_RATIO

                text_x = tp_at_max_hs_val if SWAP_AXES_CONTOUR_PLOT else hs_max_val
                text_y = hs_max_val if SWAP_AXES_CONTOUR_PLOT else tp_at_max_hs_val
                ax_contour.text(
                    text_x,
                    text_y,
                    f" {rp_years} yr",
                    color=line.get_color(),
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 0.1},
                )

                table_data_for_plot.append(
                    [
                        f"{rp_years}",
                        f"{hs_max_val:.2f}",
                        f"{mwp_at_max_hs_val:.2f}",
                        f"{tp_at_max_hs_val:.2f}",
                    ]
                )
                result_lines.append(
                    f"{rp_years:<15} {hs_max_val:<15.2f} "
                    f"{mwp_at_max_hs_val:<20.2f} {tp_at_max_hs_val:<20.2f}"
                )
                summary_table_rows.append(
                    [
                        analysis_title,
                        f"{rp_years}",
                        f"{hs_max_val:.2f}",
                        f"{mwp_at_max_hs_val:.2f}",
                        f"{tp_at_max_hs_val:.2f}",
                    ]
                )
            except Exception as exc:
                print(f"Failed to compute {rp_years}-year contour for '{analysis_title}': {exc}")
                result_lines.append(f"# Warning: failed {rp_years}-year contour: {exc}\n")

        ax_contour.set_title(f"Environmental Contours for {analysis_title} ({CONTOUR_METHOD_TYPE})")
        hs_label = "Significant Wave Height, Hs (m)"
        tp_label = "Peak Wave Period, Tp = 1.2 x MWP (s)"
        ax_contour.set_xlabel(tp_label if SWAP_AXES_CONTOUR_PLOT else hs_label)
        ax_contour.set_ylabel(hs_label if SWAP_AXES_CONTOUR_PLOT else tp_label)
        ax_contour.grid(True, linestyle="--", alpha=0.6)
        ax_contour.legend(title="Return Periods", loc="upper right")

        if table_data_for_plot:
            col_labels_plot = [
                "Return\nPeriod (yr)",
                "Max Hs\n(m)",
                "MWP @ Max Hs\n(s)",
                "Tp @ Max Hs\n(s)",
            ]
            table_on_plot = plt.table(
                cellText=table_data_for_plot,
                colLabels=col_labels_plot,
                colWidths=[0.2, 0.2, 0.2, 0.2],
                loc="bottom",
                bbox=[0.0, -0.45, 1.0, 0.3],
            )
            table_on_plot.auto_set_font_size(False)
            table_on_plot.set_fontsize(8)
            fig_contour.subplots_adjust(bottom=0.35)

        clean_filename = sanitize_filename(analysis_title)
        png_filepath = os.path.join(PNG_OUTPUT_DIR, f"contour_{clean_filename}.png")
        fig_contour.savefig(png_filepath, dpi=300, bbox_inches="tight")
        print(f"Saved PNG plot to '{png_filepath}'")
        print(f"--- Analysis complete: {analysis_title} ---")
        return fig_contour

    except Exception as exc:
        error_msg = f"Contour calculation or plotting failed for '{analysis_title}': {exc}"
        print(error_msg)
        traceback.print_exc()
        result_lines.append(f"\n# ERROR: {error_msg}\n# Traceback: {traceback.format_exc()}\n")
        if "fig_contour" in locals() and fig_contour is not None:
            plt.close(fig_contour)
        return None


# =============================================================================
# MAIN PROGRAM
# =============================================================================


def main() -> int:
    """Run the environmental contour analysis."""
    all_detailed_results: list[str] = []
    summary_table_rows: list[list[str]] = [
        ["Analysis Case", "Return Period (yr)", "Max Hs (m)", "MWP @ Max Hs (s)", "Tp @ Max Hs (s)"],
    ]

    try:
        print(f"Loading data from '{INPUT_FILE}'...")
        df_master = load_input_data()
        print(f"Data loaded successfully. Rows available: {df_master.shape[0]}")
    except FileNotFoundError:
        return 1
    except Exception as exc:
        print(f"Error while loading data: {exc}")
        traceback.print_exc()
        return 1

    if PERFORM_SECTOR_ANALYSIS:
        try:
            sector_definitions = build_direction_sectors(START_DIR_DEGREES, SECTOR_WIDTH_DEGREES)
        except Exception as exc:
            print(f"Sector configuration error: {exc}")
            traceback.print_exc()
            return 1
    else:
        sector_definitions = []

    try:
        os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)
        print(f"Output directory ready: '{PNG_OUTPUT_DIR}'")
    except OSError as exc:
        print(f"Could not create output directory '{PNG_OUTPUT_DIR}': {exc}")
        return 1

    try:
        print("Defining Virocon joint distribution model...")
        dist_descriptions_omae, fit_descriptions_omae, _semantics = get_OMAE2020_Hs_Tz()
    except Exception as exc:
        print(f"Could not define Virocon model: {exc}")
        traceback.print_exc()
        return 1

    config_summary_lines = ["\n--- SCRIPT CONFIGURATION ---\n"]
    config_details = {
        "Input File": INPUT_FILE,
        "Hs Column": HS_CSV_COL_NAME,
        "MWP Column": MWP_CSV_COL_NAME,
        "Peak Period for Plots": f"Tp = {TP_FROM_MWP_RATIO:g} x MWP",
        "MWD Column": MWD_CSV_COL_NAME if PERFORM_SECTOR_ANALYSIS else "N/A",
        "Sea State Duration (hours)": SEA_STATE_DURATION_HOURS,
        "Contour Method": CONTOUR_METHOD_TYPE,
        "Return Periods (years)": str(RETURN_PERIODS_YEARS),
        "Sector Analysis Enabled": PERFORM_SECTOR_ANALYSIS,
        "Min. Samples per Fit": MIN_SAMPLES_FOR_FIT,
        "Rows Loaded": df_master.shape[0],
    }
    if PERFORM_SECTOR_ANALYSIS:
        config_details["Sector Width (degrees)"] = SECTOR_WIDTH_DEGREES
        config_details["Sector Start Direction (degrees)"] = START_DIR_DEGREES
        config_details["Number of Generated Sectors"] = len(sector_definitions)

    for key, value in config_details.items():
        config_summary_lines.append(f"{key:<32}: {value}")
    config_summary_lines.append("-" * 32)

    try:
        with PdfPages(PDF_OUTPUT_FILE) as pdf:
            fig_omni = perform_analysis_for_dataset(
                data=df_master,
                base_model_dist_descriptions=dist_descriptions_omae,
                base_fit_descriptions=fit_descriptions_omae,
                result_lines=all_detailed_results,
                summary_table_rows=summary_table_rows,
                analysis_title="Omnidirectional",
            )
            if fig_omni is not None:
                pdf.savefig(fig_omni)
            plt.close("all")

            if PERFORM_SECTOR_ANALYSIS:
                for sector in sector_definitions:
                    sector_mask = get_sector_mask(
                        df_master[MWD_CSV_COL_NAME],
                        sector["lower"],
                        sector["upper"],
                        is_full_circle=sector["is_full_circle"],
                    )
                    sector_data = df_master[sector_mask]

                    fig_sector = perform_analysis_for_dataset(
                        data=sector_data,
                        base_model_dist_descriptions=dist_descriptions_omae,
                        base_fit_descriptions=fit_descriptions_omae,
                        result_lines=all_detailed_results,
                        summary_table_rows=summary_table_rows,
                        analysis_title=sector["title"],
                    )
                    if fig_sector is not None:
                        pdf.savefig(fig_sector)
                    plt.close("all")
    except Exception as exc:
        print(f"Error while writing PDF report '{PDF_OUTPUT_FILE}': {exc}")
        traceback.print_exc()
        return 1

    print(f"PDF report saved to '{PDF_OUTPUT_FILE}'.")

    try:
        final_output_lines = [f"{'=' * 80}\n# Environmental Contour Analysis Report\n{'=' * 80}\n"]
        final_output_lines.append(f"\n{'=' * 80}\n# Overall Results Summary Table\n{'=' * 80}\n")
        final_output_lines.append(format_summary_table(summary_table_rows))
        final_output_lines.extend(config_summary_lines)
        final_output_lines.append(f"\n{'=' * 80}\n# Detailed Analysis Logs\n{'=' * 80}\n")
        final_output_lines.extend(all_detailed_results)

        with open(RESULTS_TXT_FILE, "w", encoding="utf-8", errors="replace") as handle:
            for line in final_output_lines:
                handle.write(str(line).rstrip() + "\n")
        print(f"Results report saved to '{RESULTS_TXT_FILE}'.")
    except Exception as exc:
        print(f"Error while writing results file '{RESULTS_TXT_FILE}': {exc}")
        traceback.print_exc()
        return 1

    print("Script finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
