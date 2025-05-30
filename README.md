Extremes Joint Distribution Contour Analysis
================================================================================

PURPOSE:
--------
This script performs an environmental contour analysis on oceanographic time 
series data, specifically focusing on significant wave height (Hs) and peak 
period (Tp). It utilizes the 'virocon' Python library to model the joint 
probability distribution of these variables and subsequently calculates 
environmental contours for specified return periods.

Environmental contours are lines of constant exceedance probability for a given
sea state duration and return period. They are crucial for engineering design, 
providing a set of extreme environmental conditions (combinations of Hs and Tp) 
that a marine structure is expected to withstand over its design life.

![figure](https://github.com/user-attachments/assets/b2c6ba0b-c849-4cc5-925a-eaf5e5622d2d)
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

```pip install pandas numpy matplotlib virocon```

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
