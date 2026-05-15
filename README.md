# Virocon Environmental Contour Analysis for ERA5 Wave Data

Python workflow for fitting a joint metocean probability model to **significant wave height** and **mean wave period**, then computing **environmental contours** for selected return periods with `virocon`.

The script reads an ERA5-style CSV file named `input.csv` with the following complete column structure:

```text
datetime,swh,mwp,mwd,wind,dwi,u10,v10
```

The contour model uses:

- `swh` as significant wave height, `Hs`, in metres;
- `mwp` as mean wave period, `MWP`, in seconds;
- `mwd` as mean wave direction, required only when directional-sector analysis is enabled.

The statistical model is fitted directly using **MWP** as the period variable. The charts still use the engineering presentation convention:

```text
Peak Wave Period, Tp = 1.2 × Mean Wave Period, MWP
```

Therefore, the model fit is based on `Hs–MWP`, while plots and reported peak-period values show the derived `Tp = 1.2 × MWP`.

The script generates:

- `contours.pdf` — a multi-page PDF report with one contour plot per analysis case;
- `plots/*.png` — high-resolution PNG contour figures;
- `results.txt` — a detailed text report with configuration, fitting logs, fitted-model representation and contour summaries.

---

## 1. Purpose and scope

The script `stats_era5_data.py` performs environmental contour analysis for oceanographic time series. It is intended for engineering screening of rare combinations of wave height and wave period, including:

- omnidirectional analysis;
- directional-sector analysis based on `mwd`;
- several return periods in the same run;
- several contour-construction methods supported by `virocon`.

The principal variables are:

| Variable | CSV column | Meaning | Units |
|---|---:|---|---:|
| `Hs` | `swh` | Significant wave height | m |
| `MWP` | `mwp` | Mean wave period | s |
| `MWD` | `mwd` | Mean wave direction | degrees |
| `Tp` | derived internally | Peak wave period used for plotting/reporting only | s |

The script extracts, for each contour and return period:

- maximum `Hs` along the contour;
- `MWP` associated with maximum `Hs`;
- derived `Tp = 1.2 × MWP` associated with maximum `Hs`.

The wind-related columns in the CSV are accepted but not used by the contour analysis.

---

## 2. Input data requirements

The script expects the input file:

```text
input.csv
```

The expected full CSV header is:

```text
datetime,swh,mwp,mwd,wind,dwi,u10,v10
```

Only the columns required by the selected analysis are read.

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
| `wind` | no | may remain in the source CSV |
| `dwi` | no | may remain in the source CSV |
| `u10` | no | may remain in the source CSV |
| `v10` | no | may remain in the source CSV |

The script reads only the required columns using `pandas.read_csv(..., usecols=...)`. Extra columns do not affect the analysis.

### 2.3 Example input

```csv
datetime,swh,mwp,mwd,wind,dwi,u10,v10
1990-01-01 00:00:00,1.42,6.10,285,7.8,300,-6.76,3.90
1990-01-01 01:00:00,1.55,6.25,288,8.1,302,-6.85,4.32
1990-01-01 02:00:00,1.37,5.95,280,7.4,297,-6.59,3.36
```

### 2.4 Data checks performed by the script

The script:

1. parses `datetime` as the time index;
2. checks that the required columns are present;
3. converts `swh`, `mwp` and, when needed, `mwd` to numeric values;
4. removes rows with missing key values;
5. removes rows with non-positive `Hs` or `MWP` before model fitting.

Non-positive wave heights or periods are physically invalid for the fitted probability model and are excluded.

---

## 3. Installation

Use Python 3.11 or newer where possible.

Install the required packages:

```bash
pip install pandas numpy matplotlib scipy virocon
```

A clean virtual environment is recommended for reproducibility.

### 3.1 Windows virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas numpy matplotlib scipy virocon
```

### 3.2 Linux or macOS virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy matplotlib scipy virocon
```

---

## 4. Running the script

Place `stats_era5_data.py` and `input.csv` in the same directory, then run:

```bash
python stats_era5_data.py
```

The script writes the outputs in the same working directory:

```text
contours.pdf
results.txt
plots/
```

If `input.csv` is missing, the script creates a synthetic example `input.csv`, prints a message and exits. Replace the synthetic file with real project data and run the script again.

---

## 5. User configuration

The script contains a configuration block near the top of the file.

### 5.1 File and column settings

```python
INPUT_FILE = "input.csv"
TIME_COL = "datetime"
HS_CSV_COL_NAME = "swh"
MWP_CSV_COL_NAME = "mwp"
MWD_CSV_COL_NAME = "mwd"
```

These variables define the input file and the CSV columns used by the analysis.

### 5.2 Peak-period plotting convention

```python
TP_FROM_MWP_RATIO = 1.2
TP_PLOT_COL_NAME = "_tp_from_mwp"
```

The model is fitted with `MWP`. The derived peak-period column is used only for plots and output summaries:

```text
Tp = 1.2 × MWP
```

### 5.3 Directional-sector analysis

```python
PERFORM_SECTOR_ANALYSIS = True
START_DIR_DEGREES = 0
SECTOR_WIDTH_DEGREES = 30
```

When sector analysis is enabled, the script runs:

1. one omnidirectional analysis;
2. one additional analysis for each directional sector.

With the default values, the sectors are:

```text
0–30, 30–60, 60–90, ..., 330–0 degrees
```

The lower sector boundary is inclusive and the upper boundary is exclusive.

### 5.4 Contour parameters

```python
SEA_STATE_DURATION_HOURS = 3.0
RETURN_PERIODS_YEARS = [1, 5, 10, 25, 50, 100, 250]
MIN_SAMPLES_FOR_FIT = 200
```

- `SEA_STATE_DURATION_HOURS` defines the duration of one statistically independent sea state.
- `RETURN_PERIODS_YEARS` defines the return periods to compute.
- `MIN_SAMPLES_FOR_FIT` prevents fitting a model to very small samples.

### 5.5 Plot and output settings

```python
SWAP_AXES_CONTOUR_PLOT = True
CONTOUR_METHOD_TYPE = "IFORM"
PDF_OUTPUT_FILE = "contours.pdf"
PNG_OUTPUT_DIR = "plots"
RESULTS_TXT_FILE = "results.txt"
```

When `SWAP_AXES_CONTOUR_PLOT = True`, the plot axes are:

- x-axis: `Tp = 1.2 × MWP`;
- y-axis: `Hs`.

Supported contour methods are:

```text
IFORM
ISORM
HighestDensity
DirectSampling
```

---

## 6. Workflow implemented by the script

The script performs the following sequence:

1. loads `input.csv`;
2. validates and cleans the required columns;
3. derives the plotting-only peak-period column `Tp = 1.2 × MWP`;
4. defines the predefined `virocon` OMAE 2020 `Hs–Tz` model structure;
5. supplies `Hs` and `MWP` to the bivariate model-fitting process;
6. fits the joint probability model;
7. computes environmental contours for the selected return periods;
8. plots the data and contours using the derived `Tp` axis;
9. writes PNG plots and a multi-page PDF;
10. repeats the analysis for directional sectors, if enabled;
11. writes the detailed `results.txt` report.

The period column passed into the `virocon` model uses the internal model field name `zero_upcrossing_period`, because that is the predefined model interface. In this script, that field is populated with the CSV `mwp` values. No `pp1d` or peak-period column is required in the input file.

---

## 7. Period convention used by the script

The input period is:

```text
MWP = mean wave period
```

The fitted joint model uses:

```text
Hs–MWP
```

The plotted and reported peak period is derived as:

```text
Tp = 1.2 × MWP
```

This convention preserves the existing chart style while ensuring that the joint distribution is fitted with the `mwp` column now present in the CSV.

The derived `Tp` values should be interpreted as plotting and reporting aids, not as independently fitted peak-period data.

---

## 8. Environmental contour theory

Environmental contours describe rare combinations of environmental variables. In this script the variables are `Hs` and `MWP`.

For a target return period `T_R`, the sea-state exceedance probability is computed from the sea-state duration:

```text
alpha = sea_state_duration_hours / (return_period_years × 365.25 × 24)
```

The script delegates this calculation to:

```python
calculate_alpha(SEA_STATE_DURATION_HOURS, rp_years)
```

The fitted joint model gives the probability structure of the wave climate. The selected contour method then constructs a curve associated with the target exceedance probability.

### 8.1 Joint modelling

The bivariate probability structure may be written conceptually as:

```text
f(Hs, MWP) = f(Hs) × f(MWP | Hs)
```

This is more physically meaningful than treating wave height and wave period as independent, because period statistics generally depend on sea-state severity.

### 8.2 Predefined Virocon model

The script uses:

```python
get_OMAE2020_Hs_Tz()
```

This predefined model provides a global hierarchical model structure for wave-height and wave-period contour analysis. The script uses that structure while supplying the actual period data from the `mwp` column.

### 8.3 Contour methods

The default method is `IFORM`.

The available methods are:

| Method | General meaning |
|---|---|
| `IFORM` | Inverse First-Order Reliability Method |
| `ISORM` | Inverse Second-Order Reliability Method |
| `HighestDensity` | contour based on a highest-density probability region |
| `DirectSampling` | sampling-based contour construction |

Different contour methods may generate different curves from the same fitted joint model. This is expected and reflects different exceedance-region definitions.

---

## 9. Directional-sector analysis

When `PERFORM_SECTOR_ANALYSIS = True`, the script uses `mwd` to split the dataset into directional sectors.

### 9.1 Sector construction

The sectors start at:

```python
START_DIR_DEGREES
```

and advance by:

```python
SECTOR_WIDTH_DEGREES
```

until the full 360° circle is covered.

If the sector width does not divide 360 exactly, the final sector is shortened to avoid gaps or overlap.

### 9.2 Sector interval convention

The script uses:

```text
lower bound inclusive
upper bound exclusive
```

Example:

```text
30–60 means 30° <= mwd < 60°
```

### 9.3 Wrap-around sectors

Wrap-around sectors are handled explicitly.

Example:

```text
340–10 means mwd >= 340° OR mwd < 10°
```

### 9.4 Direction convention

The script does not convert directional conventions. It assumes that `mwd` is already expressed in the convention intended for the engineering analysis.

Before using directional results, confirm whether the source data direction means:

- waves coming from that direction; or
- waves travelling towards that direction.

---

## 10. Outputs

### 10.1 `contours.pdf`

The PDF report contains one page for each successful analysis case:

- omnidirectional case;
- each directional sector with enough data, when sector analysis is enabled.

Each plot includes:

- scatter plot of the available data;
- contours for the selected return periods;
- labels near the maximum-`Hs` points;
- a compact table with `Hs`, `MWP` and derived `Tp` at maximum `Hs`.

### 10.2 `plots/`

The script writes one high-resolution PNG figure per analysis case:

```text
plots/contour_omnidirectional.png
plots/contour_sector_0-30_deg.png
...
```

The actual filenames are sanitized automatically from the analysis titles.

### 10.3 `results.txt`

The text report contains:

- overall results summary table;
- configuration used in the run;
- number of data points loaded;
- model-fitting attempt logs;
- fitted model representation;
- contour maxima for each return period;
- warnings for skipped sectors or failed contour calculations.

The summary table columns are:

```text
Analysis Case
Return Period (yr)
Max Hs (m)
MWP @ Max Hs (s)
Tp @ Max Hs (s)
```

---

## 11. Robust model fitting and fallback logic

The script includes several robustness measures for practical metocean datasets.

### 11.1 Minimum sample count

Before fitting, each dataset or sector must contain at least:

```python
MIN_SAMPLES_FOR_FIT = 200
```

Cases with fewer samples are skipped.

### 11.2 Positive-value filtering

The script removes rows where:

```text
Hs <= 0
MWP <= 0
```

These values are incompatible with the fitted distributions.

### 11.3 Fitting strategies

The script tries the OMAE 2020 hierarchical model using:

1. default fit descriptions;
2. perturbed initial guesses;
3. widened parameter bounds.

This improves resilience when fitting difficult sectors.

### 11.4 Fallback model

If all full hierarchical fitting attempts fail, the script tries a simplified independent fallback model.

The fallback model is a practical recovery mechanism. It should not be interpreted as equivalent to the preferred dependent `Hs–MWP` model. If fallback results are used, inspect the affected sector or dataset carefully.

---

## 12. Interpreting the results

The maximum `Hs` point on a contour is a useful screening value, but it is not necessarily the governing design state for every structure.

A structure or operation may instead be governed by:

- longer period at lower `Hs`;
- wave steepness;
- directional exposure;
- resonance-sensitive period ranges;
- operational or mooring response criteria;
- overtopping, armour stability or run-up mechanisms.

For engineering use, contour points should normally be checked against the relevant response model, not only against the maximum wave height.

---

## 13. Engineering assumptions and limitations

### 13.1 The method is probabilistic

A 100-year contour does not mean that every point on the curve occurs once every 100 years. It is a contour associated with a target exceedance probability derived from the return period and sea-state duration.

### 13.2 The derived peak period is approximate

The script assumes:

```text
Tp = 1.2 × MWP
```

This is an engineering convention for plotting/reporting in this workflow. It is not a replacement for measured or modelled `Tp` data.

### 13.3 The input direction convention must be verified

Directional-sector results depend directly on the interpretation of `mwd`. The script does not decide whether directions are “coming from” or “going to”.

### 13.4 Sector width is a modelling choice

Narrow sectors provide better directional resolution but fewer data points. Wider sectors provide more stable fits but may mix different wave climates.

### 13.5 Fallback contours require caution

Fallback contours may be useful diagnostically, but they do not preserve the same dependent period-height structure as the full hierarchical model.

---

## 14. Troubleshooting

### 14.1 `ModuleNotFoundError: No module named 'virocon'`

Install dependencies:

```bash
pip install pandas numpy matplotlib scipy virocon
```

### 14.2 Missing input columns

Check that `input.csv` contains the required columns:

```text
datetime,swh,mwp
```

If sector analysis is enabled, it must also contain:

```text
mwd
```

The script no longer requires a `pp1d` column.

### 14.3 The script created a synthetic `input.csv`

This means `input.csv` was not found. Replace the generated synthetic file with real data and run the script again.

### 14.4 Many sectors are skipped

Likely cause: fewer than `MIN_SAMPLES_FOR_FIT` rows in those sectors.

Possible adjustments:

- increase sector width;
- reduce `MIN_SAMPLES_FOR_FIT` only if justified;
- use a longer time series;
- disable sector analysis.

### 14.5 Fitting fails repeatedly

Possible causes include:

- insufficient data;
- strongly clustered sector data;
- extreme outliers;
- non-representative period-height dependence;
- unsuitable directional subdivision.

Suggested checks:

- inspect the raw data;
- check units;
- check whether `mwp` is positive and realistic;
- check direction convention;
- test wider sectors;
- inspect `results.txt` for the exact fitting messages.

### 14.6 Plots look correct but period values seem too high or low

Remember that the plotted period is not the raw `mwp`. It is:

```text
Tp = 1.2 × MWP
```

The model fit and contour calculation use `MWP`.

---

## 15. References

- Virocon documentation: <https://virocon.readthedocs.io/>
- Virocon GitHub repository: <https://github.com/virocon-organization/virocon>
- Haselsteiner, A. F., Sander, A., Ohlendorf, J. H., and Thoben, K.-D. (2020). *Global hierarchical models for wind and wave contours: physical interpretations of the dependence functions*. OMAE 2020.
- Haselsteiner, A. F., Lehmkuhl, J., Pape, T., Windmeier, K.-L., and Thoben, K.-D. (2019). *ViroCon: A software to compute multivariate extremes using the environmental contour method*. SoftwareX.
- Haselsteiner, A. F., Windmeier, K.-L., Ströer, L., and Thoben, K.-D. (2022). *Update 2.0 to ViroCon: A software to compute multivariate extremes using the environmental contour method*. SoftwareX.
- Winterstein, S. R., Ude, T. C., Cornell, C. A., Bjerager, P., and Haver, S. (1993). *Environmental parameters for extreme response: inverse FORM with omission factors*.
