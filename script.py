# -*- coding: utf-8 -*-
"""
================================================================================
 Virocon Environmental Contour Analysis Script
================================================================================

PURPOSE:
--------
This script performs an environmental contour analysis on oceanographic time 
series data, specifically focusing on significant wave height (Hs) and wave 
period (Tp). It utilizes the 'virocon' Python library to model the joint 
probability distribution of these variables and subsequently calculates 
environmental contours for specified return periods.

Environmental contours are lines of constant exceedance probability for a given
sea state duration and return period. They are crucial for engineering design, 
providing a set of extreme environmental conditions (combinations of Hs and Tp) 
that a marine structure is expected to withstand over its design life.

KEY FEATURES:
-------------
- **Data Input**: Loads time series data from a CSV file containing datetime, 
  significant wave height (Hs), peak period (Tp), and mean wave direction (MWD).
- **Data Handling**: Includes a feature to generate a sample 'input.csv' if not 
  found, allowing for immediate demonstration. It also handles missing values.
- **Joint Distribution Modeling**: Employs the robust OMAE 2020 global 
  hierarchical model, which is predefined in the Virocon library, to describe 
  the joint statistics of Hs and zero-upcrossing period (Tz). 
  The script automatically converts the input peak period (Tp) to Tz.
- **Robust Fitting**: Implements multiple fitting strategies to ensure the model's 
  parameters can be estimated even with challenging datasets. This includes 
  perturbing initial guesses and widening parameter bounds.
- **Fallback Mechanism**: If the complex OMAE 2020 model fails to fit, the 
  script automatically falls back to a simpler model that treats Hs and Tz as
  statistically independent variables, ensuring the analysis can proceed.
- **Sectoral Analysis**: Optionally performs the analysis not just for the entire 
  (omnidirectional) dataset, but also for specific directional sectors based on 
  mean wave direction (MWD), which is critical for direction-sensitive designs.
- **Comprehensive Output**: Generates a multi-page PDF report with contour plots, 
  high-resolution PNG images for each analysis case, and a detailed text file 
  ('results.txt') containing configuration details, fitted model parameters, and 
  contour results.

METHODOLOGY OVERVIEW:
-----------------------
1.  **Joint Probability Model**: The script uses a Global Hierarchical Model (GHM)
    where the joint probability density function f(Hs, Tz) is factored into a 
    marginal distribution for Hs and a conditional distribution for Tz given Hs: 
    f(hs, tz) = f_Hs(hs) * f_Tz|Hs(tz|hs).
2.  **OMAE 2020 Model**: The specific model used is 'get_OMAE2020_Hs_Tz' from 
    Virocon. This model is well-documented and has been tested 
    for wind and wave applications. It typically uses an 
    Exponentiated Weibull distribution for the marginal significant wave height (Hs) 
    and a Lognormal distribution for the zero-upcrossing period (Tz) conditional 
    on Hs. The parameters of the conditional distribution are 
    defined as functions of Hs, known as dependence functions.
3.  **Contour Calculation**: Environmental contours are computed for a given return
    period (T_R) and sea state duration (t_s). These are used to calculate the 
    exceedance probability, alpha (α), per sea state.
    α = t_s / (T_R * N_year), where N_year is the number of sea states in a year.
    The script uses Virocon's `calculate_alpha` function for this purpose.
4.  **Contour Method**: The script defaults to the Inverse First-Order Reliability 
    Method (IFORM) to compute the contour coordinates. IFORM is a 
    standard, widely-used method that transforms the variables into a standard
    normal space to find the contour. Other methods like ISORM are
    also available for selection.

REQUIREMENTS & INSTALLATION:
------------------------------
The script requires Python 3 and the following packages. They can be installed
using pip:

pip install pandas numpy matplotlib virocon

HOW TO USE:
-----------
1.  **Prepare Input Data**: 
    - Create a CSV file named 'input.csv' in the same directory as the script.
    - The CSV file must contain the following columns:
      - `datetime`: Timestamp for each measurement (e.g., '2023-01-01 00:00:00').
      - `swh`: Significant wave height (Hs) in meters.
      - `pp1d`: Peak wave period (Tp) in seconds.
      - `mwd`: Mean wave direction in degrees (0-360), required only if 
               `PERFORM_SECTOR_ANALYSIS` is True.
    - If 'input.csv' is not found, the script will automatically generate a dummy
      file with synthetic data to demonstrate its functionality.

2.  **Configure Parameters**:
    - Open this script and navigate to the 'USER CONFIGURATION' section below.
    - Adjust the parameters as needed for your specific analysis. Key parameters
      include file paths, column names, return periods, and whether to perform
      sectoral analysis. Detailed comments in that section explain each variable.
    - The default `SEA_STATE_DURATION_HOURS` is 3.0, a common and recommended 
      assumption for offshore engineering applications.

3.  **Run the Script**:
    - Open a terminal or command prompt.
    - Navigate to the directory containing this script.
    - Execute the script using: `python script.py`

4.  **Review Outputs**:
    - `contours.pdf`: A multi-page PDF containing plots of the environmental 
      contours for the omnidirectional case and each directional sector.
    - `plots/` directory: Contains high-resolution PNG images of each contour plot.
    - `results.txt`: A detailed report with the configuration used, fitted model
      parameters for each case, and a summary table of the key contour results.

REFERENCES & DOCUMENTATION:
---------------------------
- **Virocon GitHub Repository**: https://github.com/virocon-organization/virocon
- **Virocon Documentation**: https://virocon.readthedocs.io/
- **Primary Scientific Reference for Virocon Models**: 
  Haselsteiner, A.F., Sander, A., Ohlendorf, J.H., Thoben, K.D. (2020) 
  Global hierarchical models for wind and wave contours: physical 
  interpretations of the dependence functions. OMAE 2020.
- **Primary Industry Guidance Reference**:
  DNVGL-RP-C205 (2017). Environmental conditions and environmental loads.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import traceback
import os
import copy
import warnings
import re # For sanitizing filenames

# --- Virocon Library Imports ---
# The Virocon library is central to this script. It provides the tools for
# creating joint probability models and calculating environmental contours.
# For full documentation, see: https://virocon.readthedocs.io/
from virocon import (
    GlobalHierarchicalModel,
    IFORMContour,
    ISORMContour,
    HighestDensityContour,
    DirectSamplingContour,
    calculate_alpha,
    get_OMAE2020_Hs_Tz  # Predefined model for Hs and Tz.
)

# --- 1. USER CONFIGURATION ---
# --------------------------------------------------------------------------
# This section contains all the parameters that the user can modify to
# customize the analysis.

# --- File and CSV Column Names ---
# Define the input file and the names of the relevant columns within it.
INPUT_FILE = 'input.csv'
TIME_COL = 'datetime'           # Datetime column in your CSV.
HS_CSV_COL_NAME = 'swh'         # Column name for Significant Wave Height (Hs) in meters.
TP_CSV_COL_NAME = 'pp1d'        # Column name for Peak Period (Tp) in seconds.
MWD_CSV_COL_NAME = 'mwd'        # Column name for Mean Wave Direction (deg) from which waves are propagating.

# --- Data Conversion ---
# The OMAE2020 model uses zero-upcrossing period (Tz), but datasets often provide
# peak period (Tp). This ratio is used for the conversion.
TP_TZ_RATIO = 1.2               # A typical ratio to convert Tp to Tz. See DNVGL-RP-C205 for relationships.

# --- Sector Analysis Configuration ---
# Set to True to perform analysis on directional sectors. This is useful for
# assessing directional loads. If False, only omnidirectional analysis is run.
PERFORM_SECTOR_ANALYSIS = True
# Defines the width of each directional sector in degrees.
SECTOR_WIDTH_DEGREES = 30

# --- Contouring Parameters ---
# Sea state duration defines the time period over which the sea state is considered
# statistically stationary. A 3-hour duration is a common standard in offshore
# engineering guidelines like DNVGL-RP-C205.
SEA_STATE_DURATION_HOURS = 3.0
# Return periods for which to calculate contours. These are standard values used
# in the design of marine structures.
RETURN_PERIODS_YEARS = [1, 5, 10, 25, 50, 100, 250]
# Minimum number of data points required in a dataset (or sector) to attempt
# fitting a model. This prevents fitting on statistically insignificant samples.
MIN_SAMPLES_FOR_FIT = 200

# --- Plotting & Output Configuration ---
# If True, Tp/Tz is plotted on the X-axis and Hs on the Y-axis, which is a
# common convention in oceanography.
SWAP_AXES_CONTOUR_PLOT = True
# Choose the contouring method. IFORM is a widely used and robust method.
# ISORM is generally more conservative.
# Other options include "HighestDensity" and "DirectSampling".
CONTOUR_METHOD_TYPE = "IFORM"   # Options: "IFORM", "ISORM", "HighestDensity", "DirectSampling"
# Filenames for the output reports.
PDF_OUTPUT_FILE = 'contours.pdf'
PNG_OUTPUT_DIR = 'plots'        # Directory for high-resolution PNG plots.
RESULTS_TXT_FILE = 'results.txt'

# --- Fitting Robustness ---
# The process of fitting a complex statistical model to data can sometimes fail
# if the optimization algorithm doesn't converge. These strategies provide
# alternative attempts to achieve a successful fit. This is a key feature
# for making the script robust and automated.
COMPLEX_MODEL_FITTING_STRATEGIES = [
    # The first attempt uses Virocon's default starting parameters.
    {'id': 'default', 'description': 'Default parameters'},
    # If the default fails, this strategy tries again with randomly perturbed
    # initial guesses (p0) for the model parameters. This can help the optimizer
    # escape local minima.
    {'id': 'perturbed_p0', 'description': 'Perturbed initial guesses (p0)',
     'p0_factor_range': (0.5, 1.5), # The range for the random perturbation factor.
     'default_p0_values': {'mu': [1.0, 0.5, 0.1], 'sigma': [0.2, 0.1, 0.5]}}, # Base p0 values to perturb.
    # This strategy widens the search space (bounds) for the parameters, giving
    # the optimizer more freedom to find a valid solution.
    {'id': 'wide_bounds', 'description': 'Wider parameter bounds',
     'bounds_multiplier_lower': 0.1, 
     'bounds_multiplier_upper': 10.0,
     'min_bound_val': 1e-9} 
]
# --------------------------------------------------------------------------


# --- 2. Main Analysis Function ---
def perform_analysis_for_dataset(data, base_model_dist_descriptions, semantics, base_fit_descriptions, result_lines, summary_table_rows, analysis_title):
    """
    Performs the core Virocon analysis on a given dataset (either omnidirectional
    or a specific sector). It fits the joint probability model, calculates 
    environmental contours, generates plots, and logs all results.

    Args:
        data (pd.DataFrame): The input data for the analysis.
        base_model_dist_descriptions (dict): Virocon model structure definition.
        semantics (dict): Virocon model variable semantics.
        base_fit_descriptions (dict): Virocon model fitting instructions.
        result_lines (list): A list to append detailed log messages to.
        summary_table_rows (list): A list to append summary results to.
        analysis_title (str): Title for the analysis case (e.g., "Omnidirectional").

    Returns:
        matplotlib.figure.Figure: The generated contour plot figure, or None if the
                                  analysis was skipped or failed.
    """
    # --- Check for sufficient data ---
    # It's important not to fit a statistical model on too few data points, as the
    # results would be unreliable.
    if data.shape[0] < MIN_SAMPLES_FOR_FIT:
        message = (f"Skipping '{analysis_title}': Insufficient data "
                   f"({data.shape[0]} samples, required {MIN_SAMPLES_FOR_FIT}).")
        print(message)
        result_lines.append(f"\n# {message}\n")
        return None 

    print(f"\n--- Starting Analysis for: {analysis_title} ---")
    result_lines.append(f"\n{'='*80}\n# ANALYSIS FOR: {analysis_title}\n{'='*80}\n")
    result_lines.append(f"Number of data points in this set: {data.shape[0]}\n")

    # Virocon uses specific internal names for variables. We must match our
    # data columns to these names.
    hs_col_virocon = 'significant_wave_height'
    tz_col_virocon = 'zero_upcrossing_period'

    # --- Data Preparation for Virocon ---
    analysis_data = data.copy()
    # Convert Tp to Tz, as the OMAE2020_Hs_Tz model is defined for Tz.
    analysis_data[tz_col_virocon] = analysis_data[TP_CSV_COL_NAME] / TP_TZ_RATIO
    analysis_data.rename(columns={HS_CSV_COL_NAME: hs_col_virocon}, inplace=True)

    # Filter out any non-positive values for Hs and Tz, which are physically
    # unrealistic and can cause issues with logarithmic transformations in models.
    min_positive_val = 1e-6 
    original_count = analysis_data.shape[0]
    analysis_data = analysis_data[
        (analysis_data[hs_col_virocon] > min_positive_val) &
        (analysis_data[tz_col_virocon] > min_positive_val)
    ].copy() 
    
    filtered_count = analysis_data.shape[0]
    if filtered_count < original_count:
        print(f"  Filtered out {original_count - filtered_count} non-positive Hs/Tz data points.")
        result_lines.append(f"# Note: Filtered out {original_count - filtered_count} non-positive Hs/Tz data points.\n")

    # Re-check sample size after filtering.
    if analysis_data.shape[0] < MIN_SAMPLES_FOR_FIT:
        message = (f"Skipping '{analysis_title}': Insufficient positive data "
                   f"({analysis_data.shape[0]} samples after filtering, required {MIN_SAMPLES_FOR_FIT}).")
        print(message)
        result_lines.append(f"\n# {message}\n")
        return None

    # Virocon expects a DataFrame with only the model's variables.
    virocon_input_data = analysis_data[[hs_col_virocon, tz_col_virocon]]

    # --- Model Fitting with Robustness Strategies ---
    fit_model = None 
    fit_successful = False
    last_fit_exception = None
    model_description_for_log = "Full Complex Model (OMAE 2020)"

    # Suppress warnings that can arise during the optimization process.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            from scipy.optimize import OptimizeWarning
            warnings.simplefilter("ignore", category=OptimizeWarning)
        except ImportError:
            pass 

        # Loop through the defined fitting strategies.
        for i, strategy_config in enumerate(COMPLEX_MODEL_FITTING_STRATEGIES):
            print(f"Fitting full model for '{analysis_title}' (Attempt {i+1}/{len(COMPLEX_MODEL_FITTING_STRATEGIES)}, Strategy: {strategy_config['description']})...")
            # Deep copy the base descriptions to avoid modifying them permanently.
            current_fit_descriptions = copy.deepcopy(base_fit_descriptions)
            temp_fit_model = GlobalHierarchicalModel(base_model_dist_descriptions)

            try:
                # Apply the current strategy's modifications to the fitting setup.
                if strategy_config['id'] == 'perturbed_p0':
                    # Modify the initial guesses (p0) for the optimizer.
                    if len(current_fit_descriptions) > 1 and current_fit_descriptions[1] and 'fit_specs' in current_fit_descriptions[1]:
                        for spec_idx, spec in enumerate(current_fit_descriptions[1]['fit_specs']):
                            if not spec: continue
                            param_group_name = 'mu' if spec_idx == 0 else 'sigma'
                            p0_to_use = spec.get('p0')
                            if p0_to_use is None:
                                default_p0s_for_group = strategy_config.get('default_p0_values', {}).get(param_group_name)
                                if default_p0s_for_group is None:
                                     num_params_expected = 3
                                     default_p0s_for_group = [1.0] * num_params_expected
                                     print(f"  Warning: No specific default p0 for '{param_group_name}', using generic: {default_p0s_for_group}")
                                p0_to_use = default_p0s_for_group
                            if not isinstance(p0_to_use, list): p0_to_use = list(p0_to_use)
                            p_range = strategy_config['p0_factor_range']
                            perturbation = np.random.uniform(p_range[0], p_range[1], size=len(p0_to_use))
                            spec['p0'] = [p * pert for p, pert in zip(p0_to_use, perturbation)]
                            print(f"  Perturbed p0 for {spec.get('var_name', param_group_name)}: {spec['p0']}")

                elif strategy_config['id'] == 'wide_bounds':
                    # Widen the parameter bounds for the optimizer.
                    if len(current_fit_descriptions) > 1 and current_fit_descriptions[1] and 'fit_specs' in current_fit_descriptions[1]:
                        for spec in current_fit_descriptions[1]['fit_specs']:
                            if not spec or 'bounds' not in spec or spec['bounds'] is None:
                                print(f"  Skipping wide_bounds for a spec in '{analysis_title}' as base bounds are missing.")
                                continue
                            mult_lower = strategy_config['bounds_multiplier_lower']
                            mult_upper = strategy_config['bounds_multiplier_upper']
                            min_b_val = strategy_config['min_bound_val']
                            original_lower, original_upper = spec['bounds']
                            original_lower_arr = np.array(original_lower); original_upper_arr = np.array(original_upper)
                            lower_bounds = np.maximum(original_lower_arr * mult_lower, min_b_val)
                            upper_bounds = original_upper_arr * mult_upper
                            upper_bounds = np.maximum(upper_bounds, lower_bounds + min_b_val)
                            spec['bounds'] = (lower_bounds.tolist(), upper_bounds.tolist())
                            print(f"  Widened bounds for {spec.get('var_name', 'param_group')}: {spec['bounds']}")
            except Exception as e_strat_apply:
                print(f"Warning: Could not fully apply strategy '{strategy_config['id']}' due to: {e_strat_apply}. Proceeding with fit attempt.")

            # Attempt to fit the model with the current strategy.
            try:
                temp_fit_model.fit(virocon_input_data, fit_descriptions=current_fit_descriptions)
                fit_model = temp_fit_model 
                fit_successful = True
                print(f"Full model fitting successful for '{analysis_title}' with strategy '{strategy_config['id']}'.")
                result_lines.append(f"# Fit successful with strategy: {strategy_config['id']}\n")
                break # Exit the loop on successful fit.
            except Exception as e:
                last_fit_exception = e
                print(f"Full model fit attempt {i+1} with strategy '{strategy_config['id']}' failed for '{analysis_title}': {e}")
                result_lines.append(f"# Fit attempt {i+1} ({strategy_config['id']}) failed: {e}\n")

        # --- Fallback to Simplified Model ---
        # If all complex model fitting strategies fail, fall back to a simpler,
        # more robust model that assumes statistical independence between Hs and Tz.
        if not fit_successful:
            warning_msg = (f"WARNING: Full complex model fitting failed for '{analysis_title}' after {len(COMPLEX_MODEL_FITTING_STRATEGIES)} attempts. "
                           f"Last error: {last_fit_exception}. Falling back to a simplified model (no Hs-Tz dependency).")
            print(warning_msg)
            result_lines.append(f"\n# {warning_msg}\n")
            model_description_for_log = "Simplified Fallback Model"

            try:
                print(f"Constructing and fitting simplified model for '{analysis_title}'...")
                
                # --- Define a simplified, independent model ---
                # This involves creating new distribution descriptions that remove the
                # conditional dependency of Tz on Hs.
                simplified_dist_descriptions = []
                
                # 1. Hs description: This is the marginal distribution, which remains unchanged
                # in its form, but we ensure its parameters are set to be fitted.
                hs_desc_template = copy.deepcopy(base_model_dist_descriptions[0])
                hs_params_to_fit = {param_name: None for param_name in hs_desc_template.get("parameters", {}).keys()}
                hs_desc_template["parameters"] = hs_params_to_fit
                simplified_dist_descriptions.append(hs_desc_template)

                # 2. Tz description: This becomes a marginal distribution instead of a conditional one.
                if len(base_model_dist_descriptions) > 1:
                    tz_desc_template = copy.deepcopy(base_model_dist_descriptions[1])
                    tz_desc_template.pop('conditional_on', None) # Remove dependency.
                    
                    # The parameters (e.g., 'mu', 'sigma') are now constants to be fitted,
                    # not functions of Hs.
                    tz_params_to_fit = {param_name: None for param_name in base_model_dist_descriptions[1].get("parameters", {}).keys()}
                    tz_desc_template["parameters"] = tz_params_to_fit
                    simplified_dist_descriptions.append(tz_desc_template)
                else: 
                    raise ValueError("Base model descriptions do not have at least two dimensions for Hs and Tz.")

                # The fitting description is now simpler, one for each marginal distribution.
                # Maximum Likelihood Estimation (MLE) is a standard method for this.
                simplified_fit_descriptions = [{'method': 'mle'}, {'method': 'mle'}]
                
                # Initialize and fit the simplified model.
                fit_model = GlobalHierarchicalModel(simplified_dist_descriptions)
                fit_model.fit(virocon_input_data, fit_descriptions=simplified_fit_descriptions)

                fit_successful = True 
                print("Simplified model fitting successful.")
                result_lines.append("# Simplified model fitting successful.\n")
            except Exception as e_simple_fit:
                last_fit_exception = e_simple_fit 
                error_msg = f"CRITICAL ERROR: Simplified fallback model also failed for '{analysis_title}'. Error: {e_simple_fit}"
                print(error_msg)
                result_lines.append(f"\n# {error_msg}\n")
                return None 

    # --- Final check before proceeding ---
    if not fit_successful or fit_model is None:
        error_msg = f"CRITICAL ERROR: All model fitting attempts ultimately failed for '{analysis_title}'. Last error: {last_fit_exception}"
        print(error_msg)
        result_lines.append(f"\n# {error_msg}\n")
        return None

    # Log the parameters of the successfully fitted model.
    result_lines.append(f"\n--- Fitted Model Parameters ({model_description_for_log}) ---\n")
    result_lines.append(str(fit_model)) 
    result_lines.append("\n-----------------------------\n")

    # --- Contour Calculation and Plotting ---
    try:
        print(f"Calculating and plotting contours for '{analysis_title}'...")
        fig_contour, ax_contour = plt.subplots(figsize=(10, 8))

        # Define which variable goes on which axis for the scatter plot.
        x_scatter_col = TP_CSV_COL_NAME if SWAP_AXES_CONTOUR_PLOT else HS_CSV_COL_NAME
        y_scatter_col = HS_CSV_COL_NAME if SWAP_AXES_CONTOUR_PLOT else TP_CSV_COL_NAME

        # Plot the raw data points as a scatter background for context.
        ax_contour.scatter(
            data[x_scatter_col], data[y_scatter_col],
            alpha=0.1, s=5, color='gray', label='All Data Points'
        )

        table_data_for_plot = [] 
        result_lines.append("\n--- Environmental Contour Results ---\n")
        header = f"{'Return Period':<15} {'Max Hs (m)':<15} {'Tz @ Max Hs (s)':<20} {'Tp @ Max Hs (s)':<20}"
        result_lines.append(header)
        result_lines.append("-" * len(header))

        # Loop through each specified return period to calculate and plot a contour.
        for rp_years in RETURN_PERIODS_YEARS:
            # Calculate the exceedance probability 'alpha' for the given return period
            # and sea state duration. This is a fundamental step in contouring.
            alpha = calculate_alpha(SEA_STATE_DURATION_HOURS, rp_years)
            try:
                # Select the appropriate Virocon contour class based on user configuration.
                ContourClass = {
                    "IFORM": IFORMContour, "ISORM": ISORMContour,
                    "HighestDensity": HighestDensityContour, "DirectSampling": DirectSamplingContour
                }.get(CONTOUR_METHOD_TYPE)
                if not ContourClass: raise ValueError(f"Unsupported CONTOUR_METHOD_TYPE: {CONTOUR_METHOD_TYPE}")
                
                contour_obj = ContourClass(fit_model, alpha)

                if contour_obj.coordinates is None or len(contour_obj.coordinates) == 0:
                    print(f"Warning: No contour data generated for {rp_years}-years for '{analysis_title}'.")
                    result_lines.append(f"# Warning: No contour data for {rp_years}-years.\n")
                    continue

                # The model works with Hs and Tz.
                contour_hs_tz = contour_obj.coordinates 
                # Convert Tz back to Tp for plotting and reporting.
                contour_hs_tp = contour_hs_tz.copy()
                contour_hs_tp[:, 1] = contour_hs_tz[:, 1] * TP_TZ_RATIO 

                # Determine coordinates for plotting based on axis swap preference.
                x_plot_coords = contour_hs_tp[:, 1] if SWAP_AXES_CONTOUR_PLOT else contour_hs_tp[:, 0] 
                y_plot_coords = contour_hs_tp[:, 0] if SWAP_AXES_CONTOUR_PLOT else contour_hs_tp[:, 1] 

                # Plot the contour line.
                line, = ax_contour.plot(x_plot_coords, y_plot_coords, label=f'{rp_years} yr')

                # Add a text label to the contour line at its highest point.
                if len(x_plot_coords) > 0 and len(y_plot_coords) > 0:
                    label_idx = np.argmax(contour_hs_tp[:, 0]) # Index of max Hs
                    text_x = contour_hs_tp[label_idx, 1] if SWAP_AXES_CONTOUR_PLOT else contour_hs_tp[label_idx, 0]
                    text_y = contour_hs_tp[label_idx, 0] if SWAP_AXES_CONTOUR_PLOT else contour_hs_tp[label_idx, 1]
                    ax_contour.text(text_x, text_y, f' {rp_years} yr',
                                    color=line.get_color(), fontsize=9, ha='left', va='bottom',
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))

                # Find the maximum significant wave height on this contour and the
                # corresponding wave period. These are important values for design.
                idx_max_hs_on_contour = np.argmax(contour_hs_tp[:, 0])
                hs_max_val = contour_hs_tp[idx_max_hs_on_contour, 0]
                tp_at_max_hs_val = contour_hs_tp[idx_max_hs_on_contour, 1]
                tz_at_max_hs_val = contour_hs_tz[idx_max_hs_on_contour, 1] 

                # Store results for logging and summary tables.
                table_data_for_plot.append([
                    f"{rp_years}", f"{hs_max_val:.2f}",
                    f"{tz_at_max_hs_val:.2f}", f"{tp_at_max_hs_val:.2f}"
                ])
                result_lines.append(f"{rp_years:<15} {hs_max_val:<15.2f} {tz_at_max_hs_val:<20.2f} {tp_at_max_hs_val:<20.2f}")
                summary_table_rows.append([analysis_title, f"{rp_years}", f"{hs_max_val:.2f}", f"{tp_at_max_hs_val:.2f}"])

            except Exception as e_contour:
                print(f"Failed to compute or plot contour for {rp_years}-years for '{analysis_title}': {e_contour}")
                result_lines.append(f"# Warning: Failed contour for {rp_years}-years: {e_contour}\n")
                continue 
        
        # --- Finalize the plot ---
        ax_contour.set_title(f'Environmental Contours for {analysis_title} ({CONTOUR_METHOD_TYPE})')
        hs_label, tp_label = "Significant Wave Height, Hs (m)", "Peak Period, Tp (s)"
        ax_contour.set_xlabel(tp_label if SWAP_AXES_CONTOUR_PLOT else hs_label)
        ax_contour.set_ylabel(hs_label if SWAP_AXES_CONTOUR_PLOT else tp_label)
        ax_contour.grid(True, linestyle='--', alpha=0.6)
        ax_contour.legend(title="Return Periods", loc='upper right')

        # Add a table of results to the bottom of the plot.
        if table_data_for_plot: 
            col_labels_plot = ["Return\nPeriod (yr)", "Max Hs\n(m)", "Tz @ Max Hs\n(s)", "Tp @ Max Hs\n(s)"]
            table_on_plot = plt.table(cellText=table_data_for_plot, colLabels=col_labels_plot,
                              colWidths=[0.2, 0.2, 0.2, 0.2], loc='bottom',
                              bbox=[0.0, -0.45, 1.0, 0.3]) 
            table_on_plot.auto_set_font_size(False)
            table_on_plot.set_fontsize(8) 
            fig_contour.subplots_adjust(bottom=0.35) 

        # --- Save the plot as a high-resolution PNG ---
        # Sanitize the analysis title to create a valid filename.
        clean_filename = re.sub(r'[^a-zA-Z0-9_-]', '', analysis_title.replace(' ', '_')).lower()
        png_filepath = os.path.join(PNG_OUTPUT_DIR, f"contour_{clean_filename}.png")
        try:
            fig_contour.savefig(png_filepath, dpi=300, bbox_inches='tight')
            print(f"Saved PNG plot to '{png_filepath}'")
        except Exception as e_png:
            print(f"Error saving PNG file to '{png_filepath}': {e_png}")
        
        print(f"--- Analysis Complete for: {analysis_title} ---\n")
        return fig_contour 

    except Exception as e_post_fit:
        # Catch any errors that occur during the contouring and plotting phase.
        error_msg = (f"ERROR in post-fitting analysis (contouring/plotting) for '{analysis_title}'. "
                     f"Error: {e_post_fit}")
        print(error_msg)
        traceback.print_exc() 
        result_lines.append(f"\n# {error_msg}\n# Traceback: {traceback.format_exc()}\n")
        # Ensure the figure is closed to prevent memory leaks.
        if 'fig_contour' in locals() and fig_contour is not None:
             plt.close(fig_contour) 
        return None

# --- 3. Helper Function for Summary Table ---
def format_summary_table(rows):
    """Formats a list of lists into a nicely aligned text-based table."""
    if not rows or len(rows) <= 1: 
        return "No summary data was generated.\n"
    # Calculate the maximum width for each column.
    widths = [max(len(str(item)) for item in col) for col in zip(*rows)]
    header, data = rows[0], rows[1:]
    # Format header, separator, and data lines.
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(header, widths))
    separator_line = "-+-".join("-" * w for w in widths)
    data_lines = [" | ".join(f"{str(item):<{w}}" for item, w in zip(row, widths)) for row in data]
    return "\n".join([header_line, separator_line] + data_lines) + "\n"

# --- 4. Script Execution Start ---
# This block runs when the script is executed directly.
if __name__ == "__main__":
    all_detailed_results = [] 
    # Initialize summary table with a header row.
    summary_table_rows = [['Analysis Case', 'Return Period (yr)', 'Max Hs (m)', 'Tp @ Max Hs (s)']]
    
    # --- Data Loading and Initial Checks ---
    try:
        print(f"Loading data from '{INPUT_FILE}'...")
        required_cols = [TIME_COL, HS_CSV_COL_NAME, TP_CSV_COL_NAME]
        if PERFORM_SECTOR_ANALYSIS: required_cols.append(MWD_CSV_COL_NAME)
        
        # Load the CSV using pandas, parsing dates and setting the time column as the index.
        df_master = pd.read_csv(INPUT_FILE, usecols=required_cols,
                                parse_dates=[TIME_COL], index_col=TIME_COL)
        print(f"Data loaded successfully. Initial shape: {df_master.shape}")
        
        # Validate that the index is a DatetimeIndex, which is important for time series operations.
        if not isinstance(df_master.index, pd.DatetimeIndex):
            raise TypeError(f"Index of DataFrame from '{INPUT_FILE}' is not DatetimeIndex.")
        
        # Drop rows with missing values in key columns.
        df_master.dropna(subset=[HS_CSV_COL_NAME, TP_CSV_COL_NAME], inplace=True) 
        if PERFORM_SECTOR_ANALYSIS: df_master.dropna(subset=[MWD_CSV_COL_NAME], inplace=True)
        if df_master.empty: raise ValueError("DataFrame empty after dropping NaNs from key columns.")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        # If the default input file is missing, offer to create a dummy one for demonstration.
        if INPUT_FILE == 'input.csv' and not os.path.exists('input.csv'):
            print("Creating a dummy 'input.csv' file for demonstration...")
            num_points = 30 * 365 * 24 # 30 years of hourly data.
            dates = pd.date_range(start='1990-01-01', periods=num_points, freq='h')
            # Generate synthetic but physically plausible wave data.
            hs = 0.1 + 2.0 * np.random.weibull(1.5, size=num_points); hs = np.clip(hs, 0.01, 25) 
            mu_log_tz = np.log(4) + 0.5 * np.log(np.maximum(hs, 0.1)) 
            sigma_log_tz = np.maximum(0.01, 0.1 + 0.2 / np.sqrt(np.maximum(hs, 0.1)))
            tz = np.exp(np.random.normal(loc=mu_log_tz, scale=sigma_log_tz))
            tp = tz * TP_TZ_RATIO; tp = np.clip(tp, 1.2, 36.0) 
            # Generate bimodal wave direction data.
            mwd1 = np.random.normal(180, 45, size=num_points) 
            mwd2 = np.random.normal(315, 45, size=num_points) 
            mwd_mix = np.random.choice([0, 1], size=num_points, p=[0.6, 0.4]) 
            mwd = np.where(mwd_mix == 0, mwd1, mwd2) % 360 
            df_master = pd.DataFrame({ HS_CSV_COL_NAME: hs, TP_CSV_COL_NAME: tp, MWD_CSV_COL_NAME: mwd }, index=dates)
            df_master.index.name = TIME_COL
            df_master.to_csv('input.csv')
            print("Dummy 'input.csv' created. Please populate it with your own data or run the script again.")
        exit()
    except Exception as e:
        print(f"Error during data loading: {e}"); traceback.print_exc(); exit()

    # Create output directory for PNG plots if it doesn't exist.
    try:
        os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)
        print(f"Output directory '{PNG_OUTPUT_DIR}' ready.")
    except OSError as e:
        print(f"CRITICAL ERROR: Could not create directory '{PNG_OUTPUT_DIR}'. Error: {e}"); exit()

    # --- Define Virocon Model ---
    try:
        print("\nDefining Virocon joint distribution model (OMAE2020_Hs_Tz)...")
        # Load the predefined OMAE 2020 model for Hs and Tz. This returns the
        # necessary dictionaries for model structure, fitting, and semantics.
        dist_descriptions_omae, fit_descriptions_omae, semantics_virocon_model = get_OMAE2020_Hs_Tz()
    except Exception as e:
        print(f"CRITICAL ERROR defining Virocon model structure: {e}"); traceback.print_exc(); exit()
    
    # --- Log Configuration ---
    config_summary_lines = ["\n--- SCRIPT CONFIGURATION ---\n"]
    config_details = {
        "Input File": INPUT_FILE, "Hs Column": HS_CSV_COL_NAME, "Tp Column": TP_CSV_COL_NAME, 
        "MWD Column": MWD_CSV_COL_NAME if PERFORM_SECTOR_ANALYSIS else "N/A",
        "Tp-to-Tz Ratio": TP_TZ_RATIO, "Sea State (hours)": SEA_STATE_DURATION_HOURS,
        "Contour Method": CONTOUR_METHOD_TYPE, "Return Periods (years)": str(RETURN_PERIODS_YEARS),
        "Sector Analysis Enabled": PERFORM_SECTOR_ANALYSIS,
    }
    if PERFORM_SECTOR_ANALYSIS: config_details["Sector Width (degrees)"] = SECTOR_WIDTH_DEGREES
    config_details["Min. Samples per Fit"] = MIN_SAMPLES_FOR_FIT
    for key, value in config_details.items(): config_summary_lines.append(f"{key:<28}: {value}")
    config_summary_lines.append("-" * 30)
    config_summary_lines.append(f"Total data points loaded (after NaN drop): {df_master.shape[0]}\n")

    # --- Run Analysis and Generate PDF ---
    # Use a PdfPages object to save all generated figures into a single PDF file.
    with PdfPages(PDF_OUTPUT_FILE) as pdf:
        # 1. Perform analysis for the omnidirectional (all directions) dataset.
        fig_c_omni = perform_analysis_for_dataset(
            data=df_master, base_model_dist_descriptions=dist_descriptions_omae,
            semantics=semantics_virocon_model, base_fit_descriptions=fit_descriptions_omae,
            result_lines=all_detailed_results, summary_table_rows=summary_table_rows,
            analysis_title="Omnidirectional"
        )
        if fig_c_omni: pdf.savefig(fig_c_omni)
        plt.close('all') # Close all figures to free up memory.

        # 2. If enabled, perform analysis for each directional sector.
        if PERFORM_SECTOR_ANALYSIS:
            sector_edges = np.arange(0, 360 + SECTOR_WIDTH_DEGREES, SECTOR_WIDTH_DEGREES)
            for i in range(len(sector_edges) - 1):
                lower_bound = sector_edges[i]; upper_bound = sector_edges[i+1]
                sector_title = f"Sector {int(lower_bound)}-{int(upper_bound)} deg"
                
                # Filter the master dataframe to get data only for the current sector.
                sector_mask = (df_master[MWD_CSV_COL_NAME] >= lower_bound) & \
                              (df_master[MWD_CSV_COL_NAME] < upper_bound)
                sector_data = df_master[sector_mask]
                
                # Call the main analysis function for the sector's data.
                fig_c_sector = perform_analysis_for_dataset(
                    data=sector_data, base_model_dist_descriptions=dist_descriptions_omae,
                    semantics=semantics_virocon_model, base_fit_descriptions=fit_descriptions_omae,
                    result_lines=all_detailed_results, summary_table_rows=summary_table_rows,
                    analysis_title=sector_title
                )
                if fig_c_sector: pdf.savefig(fig_c_sector)
                plt.close('all') 
    
    print(f"\nAll analyses complete. PDF report saved to '{PDF_OUTPUT_FILE}'.")
    
    # --- Write Final Detailed Text Report ---
    print(f"Writing detailed summary to '{RESULTS_TXT_FILE}'...")
    try:
        final_output_lines = [f"{'='*80}\n# Environmental Contour Analysis Report\n{'='*80}\n"]
        # Add the summary table at the top of the report.
        final_output_lines.append(f"\n{'='*80}\n# Overall Results Summary Table\n{'='*80}\n")
        final_output_lines.append(format_summary_table(summary_table_rows))
        # Add the configuration summary.
        final_output_lines.extend(config_summary_lines)
        # Add the detailed logs from each analysis case.
        final_output_lines.append(f"\n{'='*80}\n# Detailed Analysis Logs (Per Case)\n{'='*80}\n")
        final_output_lines.extend(all_detailed_results)
        
        with open(RESULTS_TXT_FILE, 'w', encoding='utf-8', errors='replace') as f:
            for line in final_output_lines: f.write(str(line).rstrip() + '\n') 
        print(f"Results file saved successfully to '{RESULTS_TXT_FILE}'.")
    except Exception as e:
        print(f"CRITICAL Error writing to results file '{RESULTS_TXT_FILE}': {e}"); traceback.print_exc()

    print("\nScript finished.")