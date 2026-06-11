# Duration-Conscious Target Trial Emulation

Code supplement for the manuscript on duration-conscious target trial emulation with structural nested mean models.

## Files

- `run_simulation.py`: simulation study and Monte Carlo output generation.
- `tirzepatide_dtte.py`: empirical tirzepatide versus SGLT2 active-comparator analysis.
- `_run_baseline_only.py`: helper to rerun the baseline Monte Carlo simulation without the scenario grid.
- `requirements.txt`: Python package dependencies.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the simulation

Run the full simulation pipeline:

```bash
python run_simulation.py
```

Run only the baseline Monte Carlo analysis:

```bash
python _run_baseline_only.py
```

The simulation is self-contained and does not require external data.

Expected simulation outputs include:

- `Table1_DGP_Summary.csv`
- `Table2_Estimator_Performance.csv`
- `Table3_MonteCarlo_Performance.csv`
- `Figure1_ITT_Cumulative_Incidence.png`
- `Figure2_Duration_Effects_True_vs_SNMM.png`
- `Figure3_Final_RiskDifference.png`
- `appendix/TableA1_Full_DGP_Parameters.csv`
- `appendix/TableA2_Detailed_Summary.csv`
- `appendix/TableA3_IPCW_Diagnostics.csv`
- `appendix/TableA4_Attrition_Patterns.csv`
- `appendix/TableA5_Covariate_Balance.csv`
- `appendix/MC_Raw_Results.csv`
- `appendix/MC_Scenario_Sensitivity.csv`

## Running the empirical analysis

The empirical analysis requires a restricted claims-derived person-month panel that is not included in this repository.

Place the restricted file at:

```text
raw data/dynamic_tte_oud_gl.csv
```

Then run:

```bash
python tirzepatide_dtte.py
```

Expected empirical outputs include:

- `tzp_oud_analytic_prepped.csv`
- `Table4_Tirzepatide_OUD_Results.csv`
- `Tirzepatide_Baseline_Balance.csv`
- `Tirzepatide_PS_Diagnostics.csv`
- `Tirzepatide_Duration_Response_Curves.csv`
- `Tirzepatide_Duration_Support.csv`
- `Tirzepatide_Cumulative_Incidence_Curves.csv`
- `Figure4_Tirzepatide_ITT_Cumulative_Incidence.png`
- `Figure5_Tirzepatide_Duration_Effects.png`
- `FigureA_Tirzepatide_Duration_Support.png`

## Empirical data schema

The raw empirical file should be a person-month panel with one row per patient-month.

Required core columns:

- `patient_id`: patient identifier
- `t`: follow-up month, integer-valued
- `A_tzp`: tirzepatide exposure indicator for the month
- `A_sglt2`: SGLT2 inhibitor exposure indicator for the month
- `Y`: overdose outcome indicator for the month
- `first_tzp_date`: first tirzepatide date, or missing/null if none
- `first_sglt2_date`: first SGLT2 inhibitor date, or missing/null if none
- `first_claim_date`: eligibility/index date
- `last_claim_date`: last observed claim date

Required time-varying covariate columns:

- `tv_buprenorphine`
- `tv_naltrexone`
- `tv_methadone`

Baseline covariates used when available:

- `age`
- `flag_male`
- `flag_female`
- `flag_Medicare`
- `flag_Medicaid`
- `flag_Commercial`
- `bs_depression`
- `bs_anxiety`
- `bs_bipolar`
- `bs_schizophrenia`
- `bs_adhd`
- `bs_personality_d`
- `bs_any_sud`
- `bs_aud`
- `bs_oud_dependence`
- `bs_oud_unspecified`
- `bs_prior_overdose`
- `bs_suicidal_ideation`
- `bs_cvd`
- `bs_ckd`
- `bs_liver`
- `bs_hypertension`
- `bs_hyperlipidemia`
- `bs_sleep_apnea`
- `bs_obesity`
- `bs_chronic_pain`
- `bs_buprenorphine`
- `bs_naltrexone`
- `bs_methadone`
- `bs_n_claims`
- `bs_n_inpatient`

If `flag_male` and `flag_female` are absent, the script attempts to derive them from sex/gender fields when present.

## Data availability

The simulation code is fully reproducible from the scripts in this repository. The empirical raw data are restricted clinical/claims data and are not distributed with the code. Researchers with appropriate approvals can rerun the empirical script using a de-identified person-month panel matching the schema above.

## Repository visibility

This repository is private at the time of manuscript preparation. For peer review, either provide reviewers access to the private repository or submit a code archive generated from this repository. The repository can be made public after final review if desired.
