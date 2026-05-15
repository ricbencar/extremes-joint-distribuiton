# Virocon Environmental Contour Analysis for ERA5 Wave Data

Python workflow for fitting a joint metocean probability model to **significant wave height** and **mean wave period**, then computing **environmental contours** for selected return periods with `virocon`.

The script reads ERA5-style wave data from `input.csv`, fits the joint model using:

- `swh` as significant wave height `Hs`;
- `mwp` as mean wave period `MWP`;
- `mwd` as mean wave direction when directional-sector analysis is enabled.

The statistical model is fitted directly with **MWP** as the wave-period variable. For chart readability and compatibility with common engineering presentation practice, the plotted period axis and the reported peak-period values use:

```text
Peak Wave Period, Tp = 1.2 × Mean Wave Period, MWP
```

The script generates:

- `contours.pdf` — multi-page PDF report with one contour plot per analysis case;
- `plots/*.png` — high-resolution PNG contour figures;
- `results.txt` — text report with configuration, fitting logs, model details and contour summaries.

---

## Table of Contents

- [1. Purpose and scope](#1-purpose-and-scope)
- [2. Input data requirements](#2-input-data-requirements)
- [3. Installation](#3-installation)
- [4. Running the script](#4-running-the-script)
- [5. User configuration](#5-user-configuration)
- [6. Workflow implemented by the script](#6-workflow-implemented-by-the-script)
- [7. Period convention used by the script](#7-period-convention-used-by-the-script)
- [8. Environmental contour theory](#8-environmental-contour-theory)
- [9. Directional-sector analysis](#9-directional-sector-analysis)
- [10. Outputs](#10-outputs)
- [11. Robust model fitting and fallback logic](#11-robust-model-fitting-and-fallback-logic)
- [12. Interpreting the results](#12-interpreting-the-results)
- [13. Engineering assumptions and limitations](#13-engineering-assumptions-and-limitations)
- [14. Troubleshooting](#14-troubleshooting)
- [15. References](#15-references)

---

## 1. Purpose and scope

The script `stats_era5_data.py` performs environmental contour analysis for oceanographic time series. It is intended for engineering screening of rare combinations of wave height and period, including both omnidirectional and direction-sector analyses.

The principal variables are:

| Variable | CSV column | Meaning | Units |
|---|---:|---|---:|
| `Hs` | `swh` | Significant wave height | m |
| `MWP` | `mwp` | Mean wave period | s |
| `MWD` | `mwd` | Mean wave direction | degrees |
| `Tp` | derived internally | Peak wave period used only for plotting/reporting | s |

The script fits a bivariate joint probability model to `Hs` and `MWP`, computes contours for selected return periods and extracts the point of maximum `Hs` on each contour. At that point it reports:

- maximum `Hs`;
- `MWP` associated with maximum `Hs`;
- derived `Tp = 1.2 × MWP` associated with maximum `Hs`.

The wind columns contained in the CSV are not used by this script. They may remain in the input file without affecting the analysis.

---

## 2. Input data requirements

The script expects an input file named:

```text
input.csv
```

The expected complete column structure is:

```text
datetime,swh,mwp,mwd,wind,dwi,u10,v10
```

Only a subset is used by the contour analysis.

### 2.1 Required columns

| Column | Required | Used for | Units / convention |
|---|---:|---|---|
| `datetime` | yes | time index | parseable date-time |
| `swh` | yes | significant wave height `Hs` | metres |
| `mwp` | yes | mean wave period `MWP` | seconds |
| `mwd` | only when `PERFORM_SECTOR_ANALYSIS = True` | directional-sector filtering | degrees, normally `[0, 360)` |

### 2.2 Optional columns accepted in the CSV

| Column | Used by this script | Comment |
|---|---:|---|
| `wind` | no | retained in the source CSV only |
| `dwi` | no | retained in the source CSV only |
| `u10` | no | retained in the source CSV only |
| `v10` | no | retained in the source CSV only |

The script reads only the required columns using `pandas.read_csv(..., usecols=...)`. Extra columns can exist in `input.csv` without changing the analysis.

### 2.3 Example input file

```csv
datetime,swh,mwp,mwd,wind,dwi,u10,v10
2023-01-01 00:00:00,2.10,7.05,245,9.2,250,-8.64,-3.15
2023-01-01 01:00:00,2.25,7.25,248,9.8,254,-9.40,-2.74
2023-01-01 02:00:00,1.95,6.75,238,8.5,243,-7.56,-3.88
2023-01-01 03:00:00,3.40,8.00,255,11.1,260,-10.93,-1.93
```

### 2.4 Data validity requirements

The script removes rows with missing or non-numeric values in the required analysis columns. It also removes rows with non-positive `Hs` or `MWP`, because these values are not valid for the fitted wave-height/period distributions.

For a fit to proceed, the remaining dataset or sector subset must contain at least:

```python
MIN_SAMPLES_FOR_FIT = 200
```

---

## 3. Installation

### 3.1 Minimal dependencies

Install the required Python packages with:

```bash
pip install pandas numpy matplotlib virocon scipy
```

### 3.2 Recommended virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas numpy matplotlib virocon scipy
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy matplotlib virocon scipy
```

### 3.3 Main package used for contours

The script uses `virocon` for:

- global hierarchical joint probability modelling;
- fitting of the predefined OMAE 2020 wave-height/period model;
- environmental contour construction;
- return-period exceedance probability conversion through `calculate_alpha()`.

---

## 4. Running the script

Place the following files in the same directory:

```text
stats_era5_data.py
input.csv
```

Then run:

```bash
python stats_era5_data.py
```

During execution the script prints progress messages for:

- input loading;
- output directory creation;
- model definition;
- omnidirectional analysis;
- sector-by-sector analysis, if enabled;
- fitting attempts;
- contour calculation;
- PNG, PDF and text-report export.

If `input.csv` is not found and the configured input filename is still the default `input.csv`, the script creates a synthetic example file and exits with an input-file error. Replace the synthetic file with real data and run the script again.

---

## 5. User configuration

The configuration block is located near the top of `stats_era5_data.py`.

### 5.1 File and column settings

```python
INPUT_FILE = "input.csv"
TIME_COL = "datetime"
HS_CSV_COL_NAME = "swh"
MWP_CSV_COL_NAME = "mwp"
MWD_CSV_COL_NAME = "mwd"
```

These settings define the input CSV and the columns used in the analysis.

### 5.2 Peak-period plotting convention

```python
TP_FROM_MWP_RATIO = 1.2
TP_PLOT_COL_NAME = "_tp_from_mwp"
```

The model is fitted to `MWP`. The derived `Tp` column is used only for plotting and reporting:

```text
Tp = TP_FROM_MWP_RATIO × MWP
```

With the default value:

```text
Tp = 1.2 × MWP
```

### 5.3 Directional-sector analysis

```python
PERFORM_SECTOR_ANALYSIS = True
START_DIR_DEGREES = 0
SECTOR_WIDTH_DEGREES = 30
```

When enabled, the script performs:

1. one omnidirectional analysis;
2. one additional analysis for each generated direction sector.

For the default sector settings, the sectors are:

```text
0-30, 30-60, 60-90, ..., 330-0 degrees
```

### 5.4 Contour settings

```python
SEA_STATE_DURATION_HOURS = 3.0
RETURN_PERIODS_YEARS = [1, 5, 10, 25, 50, 100, 250]
MIN_SAMPLES_FOR_FIT = 200
```

- `SEA_STATE_DURATION_HOURS` defines the assumed sea-state duration for return-period probability conversion.
- `RETURN_PERIODS_YEARS` defines which contours are computed.
- `MIN_SAMPLES_FOR_FIT` prevents model fitting on very small datasets or directional subsets.

### 5.5 Plot and output settings

```python
SWAP_AXES_CONTOUR_PLOT = True
CONTOUR_METHOD_TYPE = "IFORM"
PDF_OUTPUT_FILE = "contours.pdf"
PNG_OUTPUT_DIR = "plots"
RESULTS_TXT_FILE = "results.txt"
```

The available contour methods are:

```text
IFORM
ISORM
HighestDensity
DirectSampling
```

With `SWAP_AXES_CONTOUR_PLOT = True`, the contour plots use:

- x-axis: `Tp = 1.2 × MWP`;
- y-axis: `Hs`.

This is the default engineering-style presentation used by the script.

---

## 6. Workflow implemented by the script

The script performs the following operations:

1. loads `input.csv`;
2. parses `datetime` as the time index;
3. reads `swh`, `mwp` and, when required, `mwd`;
4. removes invalid rows with missing, non-numeric or non-positive `Hs`/`MWP`;
5. creates a derived plotting column `Tp = 1.2 × MWP`;
6. imports the predefined `get_OMAE2020_Hs_Tz()` model from `virocon`;
7. fits the model using `Hs` and `MWP` data;
8. calculates contours for each configured return period;
9. plots measured data and contours in `Tp-Hs` space;
10. extracts maximum `Hs` along each contour and the corresponding `MWP` and derived `Tp`;
11. writes each plot to `plots/` as a PNG;
12. writes all plots to `contours.pdf`;
13. writes model, configuration and contour summaries to `results.txt`;
14. repeats the fit and contour procedure for each directional sector when sector analysis is enabled.

---

## 7. Period convention used by the script

The input file now provides `mwp`, not peak period. The script therefore uses **mean wave period directly in the joint distribution model**.

The relevant implementation behaviour is:

```python
analysis_data[period_col_virocon] = analysis_data[MWP_CSV_COL_NAME].astype(float)
```

The internal Virocon column name is `zero_upcrossing_period` because the predefined upstream model is named and structured as `Hs-Tz`. In this script, that second model dimension is populated with `MWP` values. No conversion from `MWP` to `Tz` is applied.

For charts and tabulated engineering values, the script derives:

```python
Tp = 1.2 × MWP
```

This means:

- model fitting: `Hs-MWP`;
- contour calculation: `Hs-MWP`;
- scatter plots: `Tp-Hs`, where `Tp = 1.2 × MWP`;
- contour plots: `Tp-Hs`, where contour period coordinates are converted from `MWP` to `Tp`;
- results table: both `MWP @ Max Hs` and `Tp @ Max Hs` are reported.

This convention keeps the statistical analysis aligned with the available ERA5 input period column while preserving the previous chart format based on peak wave period.

---

## 8. Environmental contour theory

### 8.1 Return periods and sea-state exceedance probability

A return period is a statistical design measure. A 50-year contour is not expected to be exceeded exactly once every 50 years; rather, it is associated with a sea-state exceedance probability consistent with a 50-year return period under the assumed sea-state duration.

The script uses `virocon.calculate_alpha()` with:

```python
alpha = calculate_alpha(SEA_STATE_DURATION_HOURS, rp_years)
```

Conceptually, for sea-state duration `t_s` in hours and return period `T_R` in years:

```text
alpha = t_s / (T_R × 365.25 × 24)
```

where `alpha` is the target exceedance probability per sea state.

### 8.2 Joint probability model

For a bivariate wave-height/period model, the joint density can be understood as:

```text
f(Hs, MWP) = f(Hs) × f(MWP | Hs)
```

This reflects the fact that wave period is not independent of wave height. High sea states generally occur with different period statistics from low sea states.

### 8.3 Global hierarchical model

The script uses `GlobalHierarchicalModel` from `virocon`. In this approach:

1. the first variable is fitted marginally;
2. the second variable is fitted conditionally on the first variable;
3. dependence functions describe how the conditional distribution parameters vary with the conditioning variable.

The predefined model is imported using:

```python
dist_descriptions_omae, fit_descriptions_omae, _semantics = get_OMAE2020_Hs_Tz()
```

Although the upstream predefined function is named `Hs_Tz`, this script supplies `MWP` as the second model variable.

### 8.4 Contour methods

The script exposes the following contour classes:

| Configuration value | Virocon class |
|---|---|
| `IFORM` | `IFORMContour` |
| `ISORM` | `ISORMContour` |
| `HighestDensity` | `HighestDensityContour` |
| `DirectSampling` | `DirectSamplingContour` |

`IFORM` is the default because it is widely used in practical environmental contour work and is computationally efficient.

Different contour methods can produce different contours because they use different exceedance definitions and construction principles. Results should therefore be interpreted as method-dependent design envelopes rather than unique physical boundaries.

---

## 9. Directional-sector analysis

Directional analysis is controlled by:

```python
PERFORM_SECTOR_ANALYSIS = True
START_DIR_DEGREES = 0
SECTOR_WIDTH_DEGREES = 30
```

### 9.1 Sector generation

The helper function `build_direction_sectors()` creates sectors that cover the full 360° circle.

The lower sector limit is inclusive and the upper limit is exclusive:

```text
lower <= MWD < upper
```

For example:

```text
30-60 degrees means 30° <= MWD < 60°
```

### 9.2 Wrap-around sectors

Sectors crossing north are handled explicitly. For example:

```text
330-0 degrees
```

is treated as:

```text
MWD >= 330° OR MWD < 0°
```

For practical purposes, after angle normalization to `[0, 360)`, this selects directions from `330°` up to but not including `360°`.

A sector such as:

```text
340-10 degrees
```

is treated as:

```text
MWD >= 340° OR MWD < 10°
```

### 9.3 Full-circle special case

If `SECTOR_WIDTH_DEGREES = 360`, the script generates one full-circle sector.

### 9.4 Direction convention

The script does not convert wave-direction conventions. It assumes `mwd` already uses the convention intended for the analysis.

Before using the results for design, confirm whether the source direction is:

- direction waves are **coming from**;
- direction waves are **travelling to**.

ERA5 mean wave direction is commonly interpreted as the direction from which waves come. The script itself does not enforce or transform this convention.

---

## 10. Outputs

### 10.1 `contours.pdf`

The PDF contains one page per successful analysis case:

- omnidirectional contour plot;
- one plot per directional sector, when sector analysis is enabled.

Each plot includes:

- scatter points from the input data;
- environmental contour curves for the configured return periods;
- labels at the maximum-`Hs` point on each contour;
- an embedded table with return period, maximum `Hs`, `MWP @ Max Hs` and `Tp @ Max Hs`.

### 10.2 `plots/`

The script saves one high-resolution PNG per analysis case.

Filename examples:

```text
plots/contour_omnidirectional.png
plots/contour_sector_0-30_deg.png
plots/contour_sector_30-60_deg.png
```

### 10.3 `results.txt`

The text report includes:

- overall summary table;
- script configuration;
- number of loaded rows;
- sector configuration;
- fitting strategy logs;
- fitted model representation;
- contour maxima for all return periods;
- warnings for skipped sectors or failed contours.

The summary table columns are:

```text
Analysis Case | Return Period (yr) | Max Hs (m) | MWP @ Max Hs (s) | Tp @ Max Hs (s)
```

---

## 11. Robust model fitting and fallback logic

Real metocean datasets can be difficult to fit, especially after directional filtering. The script therefore uses a staged fitting strategy.

### 11.1 Primary fitting strategies

The script attempts the full hierarchical model using:

1. default fitting parameters;
2. perturbed initial parameter guesses;
3. wider parameter bounds.

These strategies are defined in:

```python
COMPLEX_MODEL_FITTING_STRATEGIES
```

### 11.2 Simplified fallback model

If all full hierarchical model attempts fail, the script constructs a simplified independent model by removing the conditional dependence of the second variable.

This fallback is intended to keep the workflow operational, but it is not statistically equivalent to the preferred dependent `Hs-MWP` model. If fallback fitting is reported in `results.txt`, review the dataset, sector sample size and model suitability before using the result for design decisions.

### 11.3 Skipped sectors

A sector is skipped when the number of valid samples is below:

```python
MIN_SAMPLES_FOR_FIT
```

This avoids fitting contours to sector subsets that are too small for stable parameter estimation.

---

## 12. Interpreting the results

### 12.1 Maximum Hs is a screening point

The script reports the maximum `Hs` on each contour because it is a useful engineering summary. It is not necessarily the governing design condition for all structures.

The governing response may occur at:

- lower `Hs` but longer period;
- a particular period band close to structural resonance;
- a directionally exposed sector;
- a sea state with larger wave steepness;
- a condition that is critical for overtopping, run-up, motions, mooring loads or armour stability.

### 12.2 Omnidirectional versus sectoral contours

Use the omnidirectional contour for a broad all-directions extreme sea-state envelope.

Use sectoral contours when exposure is direction-dependent, such as for:

- breakwater trunks and roundheads;
- harbour entrances;
- runway-extension maritime fills;
- berth operability;
- moored vessels;
- coastal structures with directional sheltering.

### 12.3 MWP and derived Tp

The fitted model and contour coordinates are based on `MWP`. The plotted period and peak-period report column are derived from:

```text
Tp = 1.2 × MWP
```

Therefore, if a downstream design calculation requires peak period, the reported `Tp @ Max Hs` is a converted engineering value, not an independent input variable fitted by the model.

---

## 13. Engineering assumptions and limitations

### 13.1 Environmental contours are not full response analysis

Environmental contours provide combinations of environmental variables with a target rarity. They do not replace a full long-term response analysis where structural response is integrated over all sea states.

### 13.2 MWP is used directly in a predefined Hs-period model

The upstream predefined model is named `get_OMAE2020_Hs_Tz()`. This script uses that model structure but supplies `MWP` as the period variable. This is a practical modelling convention and should be considered when comparing results with studies that use true zero-upcrossing period `Tz`.

### 13.3 Peak period is derived, not fitted

`Tp` is calculated only for plotting and reporting:

```text
Tp = 1.2 × MWP
```

The ratio is fixed and does not vary with spectral shape, wave age, swell dominance or mixed-sea conditions.

### 13.4 Direction convention must be checked upstream

The script does not transform `mwd`. Confirm the source convention before interpreting sectors as incident-wave sectors.

### 13.5 Sector width is a modelling choice

Narrow sectors improve directional resolution but reduce sample size. Wide sectors improve sample stability but may mix different wave climates.

### 13.6 Fallback results require caution

If the simplified fallback model is used, the resulting contour should be treated as a diagnostic or approximate output, not as equivalent to a successful full hierarchical fit.

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'virocon'`

Install dependencies:

```bash
pip install pandas numpy matplotlib virocon scipy
```

### Missing CSV columns

Check that `input.csv` includes the required columns:

```text
datetime,swh,mwp,mwd,wind,dwi,u10,v10
```

If sector analysis is disabled, `mwd` is not required by the fitting process. If sector analysis is enabled, `mwd` must exist and must be numeric.

### `pp1d` is no longer used

The script expects `mwp`, not `pp1d`. The joint distribution model is fitted using mean wave period.

### Many sectors are skipped

The most common cause is insufficient valid data in those direction sectors. Possible actions:

- increase `SECTOR_WIDTH_DEGREES`;
- reduce `MIN_SAMPLES_FOR_FIT` only if statistically justified;
- use a longer time series;
- disable sector analysis for an initial omnidirectional check.

### Fitting repeatedly fails

Possible causes include:

- sparse directional subset;
- strong outliers;
- non-representative period data;
- unsuitable model structure for the local wave climate;
- too many invalid or zero values.

Practical checks:

- inspect the `swh` and `mwp` ranges;
- plot `Hs` versus `MWP` before fitting;
- widen sectors;
- check whether the fallback model was used;
- verify that the input period is really `MWP` in seconds.

### The plot period values look too high or too low

Remember that the x-axis is not raw `MWP`. It is:

```text
Tp = 1.2 × MWP
```

To plot raw `MWP` instead, the plotting conversion in the script would need to be changed.

### The dummy input file was created

This happens when `input.csv` is missing. Replace it with real ERA5 or project data and run again.

---

## 15. References

### 15.1 Software and documentation

1. **virocon documentation**. User guide and examples.  
   https://virocon.readthedocs.io/

2. **virocon GitHub repository**.  
   https://github.com/virocon-organization/virocon

3. **Haselsteiner, A.F., Lehmkuhl, J., Pape, T., Windmeier, K.-L., Thoben, K.-D. (2019).**  
   *ViroCon: A software to compute multivariate extremes using the environmental contour method.*  
   SoftwareX, 9, 95–101.  
   DOI: https://doi.org/10.1016/j.softx.2019.01.003

4. **Haselsteiner, A.F., Windmeier, K.-L., Ströer, L., Thoben, K.-D. (2022).**  
   *Update 2.0 to "ViroCon: A software to compute multivariate extremes using the environmental contour method".*  
   SoftwareX, 20, 101243.  
   DOI: https://doi.org/10.1016/j.softx.2022.101243

### 15.2 Environmental contour and joint-modelling references

5. **Haselsteiner, A.F., Sander, A., Ohlendorf, J.-H., Thoben, K.-D. (2020).**  
   *Global hierarchical models for wind and wave contours: physical interpretations of the dependence functions.*  
   OMAE 2020.  
   DOI: https://doi.org/10.1115/OMAE2020-18668

6. **Haselsteiner, A.F. et al. (2021).**  
   *A benchmarking exercise for environmental contours.*  
   Relevance: comparison of environmental contour methods and model sensitivity.

7. **Ross, E. et al. (2019).**  
   *A review of environmental contour methods for marine and coastal design.*  
   Relevance: overview of environmental contour concepts, assumptions and limitations.

8. **Winterstein, S.R., Ude, T.C., Cornell, C.A., Bjerager, P., Haver, S. (1993).**  
   *Environmental parameters for extreme response: inverse FORM with omission factors.*  
   Relevance: foundational IFORM reference.

9. **Huseby, A.B., Vanem, E., Natvig, B. (2013).**  
   Direct sampling environmental contour methodology.  
   Relevance: physical-space sampling-based contour construction.

10. **Haselsteiner, A.F., Ohlendorf, J.-H., Wosniok, W., Thoben, K.-D. (2017).**  
    Highest-density environmental contour methodology.  
    Relevance: density-level contour construction.

### 15.3 Engineering guidance context

11. **DNVGL-RP-C205.**  
    *Environmental conditions and environmental loads.*  
    Relevance: offshore environmental-load context and metocean design practice.

12. **IEC 61400-3-1.**  
    *Wind energy generation systems — Design requirements for fixed offshore wind turbines.*  
    Relevance: offshore wind design context where environmental contours are commonly used.
