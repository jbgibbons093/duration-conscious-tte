"""Tirzepatide versus SGLT2 empirical DTTE analysis."""

import os
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# Paths
BASE_OUTDIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_OUTDIR, 'raw data', 'dynamic_tte_oud_gl.csv')
INTERMEDIARY_PATH = os.path.join(BASE_OUTDIR, 'tzp_oud_analytic_prepped.csv')
os.makedirs(BASE_OUTDIR, exist_ok=True)

# Settings
T = 18
GRACE = 6
TRUNC = (0.01, 0.99)
SEED = 42
SPLINE_TYPE = "piecewise"
ON_KNOT_Q = [0.50]
WASH_KNOT_Q = [0.50]
NEVER_KNOT_Q = [0.33, 0.67]
RCS_ON_KNOT_Q = [0.10, 0.50, 0.90]
RCS_WASH_KNOT_Q = [0.10, 0.50, 0.90]
B_BOOT = 200
LINK_FAM = sm.families.Binomial(link=sm.families.links.CLogLog())

TV_CONFOUNDERS = [
    'tv_buprenorphine', 'tv_naltrexone', 'tv_methadone'
]

BASELINE_COVS = [
    'age',
    'flag_male', 'flag_female',
    'flag_Medicare', 'flag_Medicaid', 'flag_Commercial',
    'bs_depression', 'bs_anxiety', 'bs_bipolar', 'bs_schizophrenia',
    'bs_adhd', 'bs_personality_d',
    'bs_any_sud', 'bs_aud', 'bs_oud_dependence', 'bs_oud_unspecified',
    'bs_prior_overdose', 'bs_suicidal_ideation',
    'bs_cvd', 'bs_ckd', 'bs_liver', 'bs_hypertension',
    'bs_hyperlipidemia', 'bs_sleep_apnea', 'bs_obesity', 'bs_chronic_pain',
    'bs_buprenorphine', 'bs_naltrexone', 'bs_methadone',
    'bs_n_claims', 'bs_n_inpatient'
]

BASELINE_MODEL_COVS = [
    'age',
    'flag_male',
    'flag_Medicare', 'flag_Medicaid', 'flag_Commercial',
    'bs_depression', 'bs_anxiety', 'bs_bipolar', 'bs_schizophrenia',
    'bs_adhd', 'bs_personality_d',
    'bs_any_sud', 'bs_aud', 'bs_oud_dependence', 'bs_oud_unspecified',
    'bs_prior_overdose', 'bs_suicidal_ideation',
    'bs_cvd', 'bs_ckd', 'bs_liver', 'bs_hypertension',
    'bs_hyperlipidemia', 'bs_sleep_apnea', 'bs_obesity', 'bs_chronic_pain',
    'bs_buprenorphine', 'bs_naltrexone', 'bs_methadone',
    'bs_n_claims', 'bs_n_inpatient'
]

BALANCE_LABELS = {
    'age': 'Age',
    'flag_male': 'Male',
    'flag_female': 'Female',
    'flag_Medicare': 'Medicare',
    'flag_Medicaid': 'Medicaid',
    'flag_Commercial': 'Commercial insurance',
    'bs_depression': 'Depression',
    'bs_anxiety': 'Anxiety',
    'bs_bipolar': 'Bipolar disorder',
    'bs_schizophrenia': 'Schizophrenia',
    'bs_adhd': 'ADHD',
    'bs_personality_d': 'Personality disorder',
    'bs_any_sud': 'Any substance use disorder',
    'bs_aud': 'Alcohol use disorder',
    'bs_oud_dependence': 'OUD dependence',
    'bs_oud_unspecified': 'Unspecified OUD',
    'bs_prior_overdose': 'Prior overdose',
    'bs_suicidal_ideation': 'Suicidal ideation',
    'bs_cvd': 'Cardiovascular disease',
    'bs_ckd': 'Chronic kidney disease',
    'bs_liver': 'Liver disease',
    'bs_hypertension': 'Hypertension',
    'bs_hyperlipidemia': 'Hyperlipidemia',
    'bs_sleep_apnea': 'Sleep apnea',
    'bs_obesity': 'Obesity',
    'bs_chronic_pain': 'Chronic pain',
    'bs_buprenorphine': 'Baseline buprenorphine',
    'bs_naltrexone': 'Baseline naltrexone',
    'bs_methadone': 'Baseline methadone',
    'bs_n_claims': 'Baseline claims count',
    'bs_n_inpatient': 'Baseline inpatient count',
}


# Utilities
def _zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)

def _truncate_weights(w, lo_q, hi_q):
    lo, hi = np.nanquantile(w, lo_q), np.nanquantile(w, hi_q)
    return np.clip(w, lo, hi)

def _numerical_gradient(func, x0, eps=1e-5):
    f0 = func(x0)
    grad = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        grad[i] = (func(xp) - func(xm)) / (2 * eps)
    return grad

def _gmm_jacobian(moment_func, beta, eps=1e-5):
    g0 = moment_func(beta).mean(axis=0)
    p = len(beta)
    q = len(g0)
    J = np.zeros((q, p))
    for j in range(p):
        bp = beta.copy(); bp[j] += eps
        bm = beta.copy(); bm[j] -= eps
        J[:, j] = (moment_func(bp).mean(axis=0) - moment_func(bm).mean(axis=0)) / (2 * eps)
    return J

def _cluster_robust_glm_cov(model, exog, endog, weights, groups):
    """One-way cluster-robust covariance for a fitted GLM."""
    mu = np.asarray(model.fittedvalues, dtype=float)
    y = np.asarray(endog, dtype=float)
    X = np.asarray(exog, dtype=float)
    w = np.asarray(weights, dtype=float)
    groups = np.asarray(groups)
    scores = (w * (y - mu))[:, None] * X
    uniq, inv = np.unique(groups, return_inverse=True)
    cluster_scores = np.zeros((len(uniq), X.shape[1]), dtype=float)
    np.add.at(cluster_scores, inv, scores)
    bread = np.asarray(model.normalized_cov_params, dtype=float)
    meat = cluster_scores.T @ cluster_scores
    cov = bread @ meat @ bread
    if len(uniq) > 1:
        cov *= len(uniq) / (len(uniq) - 1)
    return cov

def _cluster_omega(G, groups, ridge=1e-8):
    """Cluster-robust moment covariance for the step-2 weight matrix and J."""
    G = np.asarray(G, dtype=float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(uniq), G.shape[1]), dtype=float)
    np.add.at(sums, inv, G)
    Omega = (sums.T @ sums) / G.shape[0]
    return Omega + ridge * np.eye(G.shape[1])


def _cluster_gmm_cov(moment_matrix, beta_hat, W_opt, groups):
    """Cluster-robust sandwich covariance for overidentified GMM."""
    D = _gmm_jacobian(moment_matrix, beta_hat)
    G = moment_matrix(beta_hat)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    cluster_sums = np.zeros((len(uniq), G.shape[1]), dtype=float)
    np.add.at(cluster_sums, inv, G)
    n_obs = G.shape[0]
    S = (cluster_sums.T @ cluster_sums) / (n_obs ** 2)
    bread = D.T @ W_opt @ D
    bread_inv = np.linalg.pinv(bread)
    middle = D.T @ W_opt @ S @ W_opt @ D
    V_beta = bread_inv @ middle @ bread_inv
    if len(uniq) > 1:
        V_beta *= len(uniq) / (len(uniq) - 1)
    return V_beta

def _add_grace_strategy_censoring(clones, grace_len):
    """Grace-period strategy censoring."""
    clones = clones.sort_values(['id', 'Z_clone', 't']).copy()
    grace_mask = clones['t'] <= grace_len
    clones['A_tzp_grace'] = (clones['A'] * grace_mask).astype(int)
    clones['A_sglt2_grace'] = (clones['A_sglt2'] * grace_mask).astype(int)
    ever_tzp_by_grace = clones.groupby(['id', 'Z_clone'])['A_tzp_grace'].transform('max')
    ever_sglt2_by_grace = clones.groupby(['id', 'Z_clone'])['A_sglt2_grace'].transform('max')

    c_tzp_strategy_opp = (clones['Z_clone'] == 1) & grace_mask & (clones['A_sglt2'] == 1)
    c_tzp_strategy_missed = (clones['Z_clone'] == 1) & (clones['t'] == grace_len) & (ever_tzp_by_grace == 0)
    c_sglt2_strategy_opp = (clones['Z_clone'] == 0) & grace_mask & (clones['A'] == 1)
    c_sglt2_strategy_missed = (clones['Z_clone'] == 0) & (clones['t'] == grace_len) & (ever_sglt2_by_grace == 0)

    clones['C_adherence'] = (
        c_tzp_strategy_opp | c_tzp_strategy_missed |
        c_sglt2_strategy_opp | c_sglt2_strategy_missed
    ).astype(int)
    clones.drop(columns=['A_tzp_grace', 'A_sglt2_grace'], inplace=True)
    return clones

def _add_pre_switch_censoring(df):
    """Censor before cross-over."""
    df = df.sort_values(['id', 't']).copy()
    df['first_t'] = df.groupby('id')['t'].transform('min')

    opp_on = np.where(
        df['Z'].values == 1,
        df['A_sglt2'].values.astype(int),
        df['A_tzp'].values.astype(int),
    )
    df['switch_now'] = opp_on.astype(int)

    first_switch = (
        df.loc[df['switch_now'] == 1, ['id', 't']]
          .groupby('id', as_index=True)['t']
          .min()
          .rename('first_switch_t')
    )
    df = df.merge(first_switch, on='id', how='left')

    immediate_switch = df['first_switch_t'].notna() & (df['first_switch_t'] <= df['first_t'])
    df['drop_immediate_switcher'] = immediate_switch.astype(int)

    df['C_switch'] = 0
    prior_switch = (
        df['first_switch_t'].notna() &
        (df['first_switch_t'] > df['first_t']) &
        (df['t'] == (df['first_switch_t'] - 1))
    )
    df.loc[prior_switch, 'C_switch'] = 1
    return df


def _harmonize_sex_flags(df):
    """Force female to be the complement of male for patient-level balance/modeling."""
    if 'flag_male' not in df.columns:
        return df

    male = pd.to_numeric(df['flag_male'], errors='coerce').fillna(0)
    male = (male > 0).astype(int)
    df['flag_male'] = male
    df['flag_female'] = 1 - male
    return df


# Spline basis
def piecewise_basis(d, knots):
    d = np.asarray(d, dtype=float)
    knots = np.sort(np.asarray(knots, dtype=float))
    s_list = []
    prev = 0.0
    for k in knots:
        width = max(k - prev, 1e-6)
        seg = np.minimum(np.maximum(d - prev, 0.0), width)
        s_list.append(seg)
        prev = k
    s_last = np.maximum(d - prev, 0.0)
    s_list.append(s_last)
    return np.column_stack(s_list)

def rcs_basis(d, knots):
    """Restricted cubic spline basis: k knots -> k-1 basis functions."""
    d = np.asarray(d, dtype=float)
    knots = np.sort(np.asarray(knots, dtype=float))
    k = len(knots)
    if k < 3:
        raise ValueError("RCS requires at least 3 knots")

    def _truncpow3(x):
        return np.where(x > 0, x ** 3, 0.0)

    t_last = knots[-1]
    t_pen = knots[-2]
    denom = max(t_last - t_pen, 1e-6)
    scale = max((t_last - knots[0]) ** 2, 1e-6)

    cols = [d]
    for j in range(k - 2):
        tj = knots[j]
        col = (
            _truncpow3(d - tj)
            - _truncpow3(d - t_pen) * (t_last - tj) / denom
            + _truncpow3(d - t_last) * (t_pen - tj) / denom
        )
        cols.append(col / scale)
    return np.column_stack(cols)

def make_basis(d, knots, spline_type=SPLINE_TYPE):
    if spline_type == "rcs":
        return rcs_basis(d, knots)
    return piecewise_basis(d, knots)

def _evaluate_curve(durations, knots, beta_coeffs, cov=None, spline_type=SPLINE_TYPE):
    """Evaluate duration curve and optional analytical CI band."""
    dur = np.asarray(durations, dtype=float)
    beta = np.asarray(beta_coeffs, dtype=float)
    B = make_basis(dur, knots, spline_type=spline_type)
    est = B @ beta
    if cov is None:
        se = np.full_like(est, np.nan, dtype=float)
    else:
        cov = np.asarray(cov, dtype=float)
        se = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', B, cov, B), 0.0, np.inf))
    lo = est - 1.96 * se
    hi = est + 1.96 * se
    return est, lo, hi

def _summarize_duration_support(df, duration_col, label):
    """Summarize identification support by duration month."""
    if df.empty:
        return pd.DataFrame(columns=[
            'component', 'duration', 'n_clone_person_periods', 'n_unique_patients',
            'n_events', 'event_rate', 'weight_sum', 'effective_n'
        ])

    out = df.copy()
    out['sw_sq'] = out['sw'] ** 2
    g = (out.groupby(duration_col, as_index=False)
           .agg(
               n_clone_person_periods=('id', 'size'),
               n_unique_patients=('id', 'nunique'),
               n_events=('Y', 'sum'),
               weight_sum=('sw', 'sum'),
               weight_sum_sq=('sw_sq', 'sum'),
           )
           .rename(columns={duration_col: 'duration'}))
    g['component'] = label
    g['event_rate'] = np.where(
        g['n_clone_person_periods'] > 0,
        g['n_events'] / g['n_clone_person_periods'],
        np.nan,
    )
    g['effective_n'] = np.where(
        g['weight_sum_sq'] > 0,
        (g['weight_sum'] ** 2) / g['weight_sum_sq'],
        np.nan,
    )
    return g[[
        'component', 'duration', 'n_clone_person_periods', 'n_unique_patients',
        'n_events', 'event_rate', 'weight_sum', 'effective_n'
    ]]

def _summarize_state_support(df, label):
    """Summarize identification support for a non-duration state indicator."""
    if df.empty:
        return pd.DataFrame(columns=[
            'component', 'duration', 'n_clone_person_periods', 'n_unique_patients',
            'n_events', 'event_rate', 'weight_sum', 'effective_n'
        ])

    out = df.copy()
    out['sw_sq'] = out['sw'] ** 2
    weight_sum = out['sw'].sum()
    weight_sum_sq = out['sw_sq'].sum()
    return pd.DataFrame([{
        'component': label,
        'duration': 1,
        'n_clone_person_periods': len(out),
        'n_unique_patients': out['id'].nunique(),
        'n_events': out['Y'].sum(),
        'event_rate': out['Y'].mean() if len(out) else np.nan,
        'weight_sum': weight_sum,
        'effective_n': (weight_sum ** 2) / weight_sum_sq if weight_sum_sq > 0 else np.nan,
    }])


# Data prep
def load_and_prepare(raw_path=RAW_PATH, save_path=INTERMEDIARY_PATH,
                     grace=GRACE, verbose=True, include_baseline_covs=False):
    """
    Load raw person-month panel from Databricks export.
    Filter to tirzepatide (Z=1) vs SGLT2 (Z=0) groups.
    Construct Z, A, X, time-varying MOUD indicators, and censoring flags.
    Save prepped intermediary file.
    """
    t0 = time.time()
    if verbose:
        print("Loading raw panel data...")

    if raw_path.endswith('.xlsx'):
        df = pd.read_excel(raw_path, engine='calamine')
    else:
        df = pd.read_csv(raw_path)

    df = _harmonize_sex_flags(df)

    if verbose:
        print(f"  Raw panel: {len(df):,} person-months, "
              f"{df['patient_id'].nunique():,} patients")

    patient_info = df.groupby('patient_id').agg(
        first_tzp_date=('first_tzp_date', 'first'),
        first_sglt2_date=('first_sglt2_date', 'first'),
        eligibility_date=('first_claim_date', 'first'),
    ).reset_index()

    patient_info['has_tzp'] = patient_info['first_tzp_date'].notna() & (patient_info['first_tzp_date'] != 'null')
    patient_info['has_sglt2'] = patient_info['first_sglt2_date'].notna() & (patient_info['first_sglt2_date'] != 'null')

    if verbose:
        print(f"  Patients with tirzepatide: {patient_info['has_tzp'].sum():,}")
        print(f"  Patients with SGLT2: {patient_info['has_sglt2'].sum():,}")
        print(f"  Patients with both: {(patient_info['has_tzp'] & patient_info['has_sglt2']).sum():,}")
        print(f"  Patients with neither: {(~patient_info['has_tzp'] & ~patient_info['has_sglt2']).sum():,}")

    keep_pids = patient_info.loc[
        patient_info['has_tzp'] | patient_info['has_sglt2'],
        'patient_id'
    ]
    df = df[df['patient_id'].isin(keep_pids)].copy()

    if verbose:
        print(f"  After filtering to TZP/SGLT2: {len(df):,} person-months, "
              f"{df['patient_id'].nunique():,} patients")

    df = df.rename(columns={'patient_id': 'id'})

    df['A'] = df['A_tzp'].astype(int)
    df['A_sglt2'] = df['A_sglt2'].astype(int)

    df['Y'] = df['Y'].astype(int)

    df['t'] = df['t'].astype(int)
    df = df[df['t'] <= T].copy()

    grace_assign = (
        df.loc[df['t'] <= grace]
          .groupby('id', as_index=False)
          .agg(
              init_tzp_grace=('A', 'max'),
              init_sglt2_grace=('A_sglt2', 'max'),
          )
    )
    grace_assign['both_grace'] = (
        (grace_assign['init_tzp_grace'] == 1) &
        (grace_assign['init_sglt2_grace'] == 1)
    )
    grace_assign['neither_grace'] = (
        (grace_assign['init_tzp_grace'] == 0) &
        (grace_assign['init_sglt2_grace'] == 0)
    )
    grace_assign['eligible_strategy'] = ~(grace_assign['both_grace'] | grace_assign['neither_grace'])
    grace_assign['Z'] = grace_assign['init_tzp_grace'].astype(int)
    df = df.merge(
        grace_assign[['id', 'init_tzp_grace', 'init_sglt2_grace',
                      'both_grace', 'neither_grace', 'eligible_strategy', 'Z']],
        on='id', how='left'
    )
    df = df[df['eligible_strategy']].copy()
    df['Z'] = df['Z'].fillna(0).astype(int)

    if verbose:
        z_counts = df[['id', 'Z']].drop_duplicates()['Z'].value_counts()
        n_both = int(grace_assign['both_grace'].sum())
        n_neither = int(grace_assign['neither_grace'].sum())
        print(f"  Dropped both-drug grace initiators: {n_both:,}")
        print(f"  Dropped neither-drug grace initiators: {n_neither:,}")
        print(f"  Z=1 (TZP within grace, no SGLT2): {z_counts.get(1, 0):,}")
        print(f"  Z=0 (SGLT2 within grace, no TZP): {z_counts.get(0, 0):,}")

    df = _add_pre_switch_censoring(df)
    n_immediate_switch = int(df[['id', 'drop_immediate_switcher']]
                               .drop_duplicates()['drop_immediate_switcher'].sum())
    if n_immediate_switch > 0:
        df = df[df['drop_immediate_switcher'] == 0].copy()

    df['last_claim_date'] = pd.to_datetime(df['last_claim_date'], errors='coerce')
    admin_end = pd.Timestamp('2026-02-01')
    df['last_month'] = df.groupby('id')['t'].transform('max')
    df['C_ltfu'] = (
        (df['t'] == df['last_month']) &
        (df['last_claim_date'].notna()) &
        (df['last_claim_date'] < admin_end - pd.Timedelta(days=60)) &
        (df['Y'] == 0)
    ).astype(int)
    df['C_any'] = np.maximum(df['C_ltfu'], df['C_switch']).astype(int)

    if verbose:
        print(f"  LTFU censored patients: {df.groupby('id')['C_ltfu'].max().sum():,}")
        print(f"  Switch-censored patients: {df.groupby('id')['C_switch'].max().sum():,}")
        print(f"  Any-censored patients: {df.groupby('id')['C_any'].max().sum():,}")
        print(f"  Dropped immediate switchers: {n_immediate_switch:,}")

    for tv_col in TV_CONFOUNDERS:
        if tv_col not in df.columns:
            df[tv_col] = 0
        df[tv_col] = df[tv_col].fillna(0).astype(float)

    baseline = df[df['t'] == 1].copy()
    model_covs = [c for c in BASELINE_MODEL_COVS if c in baseline.columns]
    if len(model_covs) > 0:
        ps_design = baseline[model_covs].fillna(0).values.astype(float)
        ps_design = sm.add_constant(ps_design)
        if verbose:
            print("  Fitting baseline propensity model for X...")
        try:
            bl_model = sm.GLM(baseline['Z'].values, ps_design,
                              family=sm.families.Binomial()).fit(disp=0)
            bl_pred = bl_model.predict(ps_design)
            bl_pred = np.clip(bl_pred, 1e-6, 1 - 1e-6)
            baseline['X'] = np.log(bl_pred / (1 - bl_pred))

            treated = baseline['Z'].values.astype(bool)
            n_pos, n_neg = int(treated.sum()), int((~treated).sum())
            if n_pos > 0 and n_neg > 0:
                ranks = pd.Series(bl_pred).rank().values
                u_stat = ranks[treated].sum() - n_pos * (n_pos + 1) / 2
                c_stat = float(u_stat / (n_pos * n_neg))
                ps_diag = pd.DataFrame([{
                    "n_z1": n_pos, "n_z0": n_neg,
                    "ps_min": float(bl_pred.min()), "ps_max": float(bl_pred.max()),
                    "ps_mean_z1": float(bl_pred[treated].mean()),
                    "ps_mean_z0": float(bl_pred[~treated].mean()),
                    "c_statistic": c_stat,
                }])
                try:
                    ps_diag.to_csv(os.path.join(BASE_OUTDIR,
                                   "Tirzepatide_PS_Diagnostics.csv"), index=False)
                except PermissionError as e:
                    if verbose:
                        print(f"  Warning: could not write PS diagnostics ({e})")
                if verbose:
                    print(f"  Baseline PS c-statistic: {c_stat:.3f}")
        except Exception as e:
            if verbose:
                print(f"  Warning: baseline model failed ({e}), using covariate sum")
            baseline['X'] = baseline[model_covs].fillna(0).sum(axis=1).astype(float)
        df = df.merge(baseline[['id', 'X']], on='id', how='left')
        df['X'] = df['X'].fillna(0)
    else:
        df['X'] = 0.0

    keep_cols = ['id', 't', 'Z', 'A', 'A_sglt2', 'X', 'Y',
                 'C_ltfu', 'C_switch', 'C_any'] + list(TV_CONFOUNDERS)
    if include_baseline_covs:
        keep_cols.extend([c for c in BASELINE_COVS if c in df.columns])
    df = df[keep_cols].copy()
    df = df.sort_values(['id', 't']).reset_index(drop=True)

    if verbose:
        print(f"\n  Final analysis dataset: {len(df):,} person-months, "
              f"{df['id'].nunique():,} patients")
        print(f"  Events (Y=1): {df['Y'].sum():,}")
        print(f"  Tirzepatide months (A=1): {df['A'].sum():,}")
        z_pct = df[['id', 'Z']].drop_duplicates()['Z'].mean()
        print(f"  Z=1 (tirzepatide strategy): {z_pct:.3%}")
        print(f"  LTFU censored rows: {df['C_ltfu'].sum():,}")
        print(f"  Switch-censored rows: {df['C_switch'].sum():,}")
        print(f"  Any-censored rows: {df['C_any'].sum():,}")
        print(f"  Time range: t={df['t'].min()} to {df['t'].max()}")

    if save_path is not None:
        df.to_csv(save_path, index=False)
        if verbose:
            print(f"\n  Saved intermediary: {save_path}")
    if verbose:
        print(f"  Loaded and prepped in {time.time()-t0:.0f}s")

    return df


def build_empirical_balance_table(df, covariates=BASELINE_COVS):
    """Baseline covariate balance before and after stabilized IPTW."""
    base = df.loc[df['t'] == 1].copy()
    if base.empty:
        return pd.DataFrame()

    covs = [c for c in covariates if c in base.columns]
    if not covs:
        return pd.DataFrame()

    ps = 1.0 / (1.0 + np.exp(-base['X'].values))
    ps = np.clip(ps, 0.01, 0.99)
    p_z1 = base['Z'].mean()
    base['w_iptw'] = np.where(base['Z'] == 1, p_z1 / ps, (1 - p_z1) / (1 - ps))
    base['w_iptw'] = _truncate_weights(base['w_iptw'], TRUNC[0], TRUNC[1])

    def _moments(x, w=None):
        xv = pd.to_numeric(x, errors='coerce').fillna(0.0).to_numpy(dtype=float)
        if w is None:
            mean = float(np.mean(xv))
            var = float(np.mean((xv - mean) ** 2))
        else:
            wv = np.asarray(w, dtype=float)
            mean = float(np.sum(wv * xv) / np.sum(wv))
            var = float(np.sum(wv * (xv - mean) ** 2) / np.sum(wv))
        return mean, var

    rows = []
    for cov in covs:
        z0 = base['Z'] == 0
        z1 = base['Z'] == 1

        mean0_u, var0_u = _moments(base.loc[z0, cov])
        mean1_u, var1_u = _moments(base.loc[z1, cov])
        denom_u = np.sqrt(max((var0_u + var1_u) / 2, 1e-12))
        smd_u = (mean1_u - mean0_u) / denom_u

        mean0_w, var0_w = _moments(base.loc[z0, cov], base.loc[z0, 'w_iptw'])
        mean1_w, var1_w = _moments(base.loc[z1, cov], base.loc[z1, 'w_iptw'])
        denom_w = np.sqrt(max((var0_w + var1_w) / 2, 1e-12))
        smd_w = (mean1_w - mean0_w) / denom_w

        rows.append({
            'covariate': cov,
            'label': BALANCE_LABELS.get(cov, cov),
            'z0_unweighted_mean': mean0_u,
            'z1_unweighted_mean': mean1_u,
            'smd_unweighted': smd_u,
            'z0_weighted_mean': mean0_w,
            'z1_weighted_mean': mean1_w,
            'smd_weighted': smd_w,
        })

    out = pd.DataFrame(rows)
    out['abs_smd_unweighted'] = out['smd_unweighted'].abs()
    out['abs_smd_weighted'] = out['smd_weighted'].abs()
    return out.sort_values(['abs_smd_unweighted', 'label'], ascending=[False, True]).reset_index(drop=True)


# Treatment-history timers
def add_spline_timers_vec(df, grace_len=GRACE,
                          on_q=None, wash_q=None,
                          never_q=NEVER_KNOT_Q,
                          fixed_knots_on=None, fixed_knots_wash=None,
                          fixed_knots_never=None,
                          spline_type=SPLINE_TYPE):
    """Compute timers and spline bases for ON/WASH/NEVER states."""
    df = df.sort_values(['id', 'Z_clone', 't']).copy()
    if on_q is None:
        on_q = RCS_ON_KNOT_Q if spline_type == "rcs" else ON_KNOT_Q
    if wash_q is None:
        wash_q = RCS_WASH_KNOT_Q if spline_type == "rcs" else WASH_KNOT_Q

    A = df['A'].values
    A_sglt2 = df['A_sglt2'].values if 'A_sglt2' in df.columns else np.zeros(len(df), dtype=int)
    gid = df['id'].astype(str) + '_' + df['Z_clone'].astype(str)
    is_new_group = (gid != gid.shift(1)).astype(int).values

    n = len(df)
    t_on = np.zeros(n, dtype=int)
    ever_on = np.zeros(n, dtype=int)
    t_off = np.zeros(n, dtype=int)
    t_never = np.zeros(n, dtype=int)

    for i in range(n):
        pure_tzp = (A[i] == 1 and A_sglt2[i] == 0)
        if is_new_group[i]:
            t_on[i] = int(pure_tzp)
            ever_on[i] = A[i]
            t_off[i] = 0
            t_never[i] = 1 - A[i]
        elif A_sglt2[i] == 1:
            t_on[i] = 0
            ever_on[i] = max(ever_on[i-1], A[i])
            t_off[i] = 0
            t_never[i] = 0 if ever_on[i] else t_never[i-1] + 1
        elif A[i] == 1:
            t_on[i] = t_on[i-1] + 1
            ever_on[i] = 1
            t_off[i] = 0
            t_never[i] = 0
        else:
            t_on[i] = 0
            ever_on[i] = max(ever_on[i-1], 0)
            if ever_on[i] and (t_on[i-1] > 0 or t_off[i-1] > 0):
                t_off[i] = t_off[i-1] + 1
                t_never[i] = 0
            elif ever_on[i]:
                t_off[i] = 0
                t_never[i] = 0
            else:
                t_off[i] = 0
                t_never[i] = t_never[i-1] + 1

    df['t_on'] = t_on
    df['ever_on'] = ever_on
    df['t_off'] = t_off
    df['t_never'] = t_never

    def choose_knots(d_pos, q_list):
        fallback = np.array([2.0, 4.0, 6.0], dtype=float)
        if len(d_pos) == 0 or np.max(d_pos) <= 1:
            knots = fallback[:max(len(q_list), 1)]
        else:
            q_arr = np.clip(np.asarray(q_list, dtype=float), 0.0, 1.0)
            knots = np.quantile(d_pos, q_arr)
            knots = np.unique(knots)
        if spline_type == "rcs":
            base = np.unique(np.asarray(knots, dtype=float))
            if base.size < 3:
                extra = fallback.copy()
                knots = np.unique(np.concatenate([base, extra]))
            knots = np.sort(np.asarray(knots, dtype=float))[:3] if np.unique(knots).size > 3 else np.sort(np.asarray(knots, dtype=float))
            if np.unique(knots).size < 3:
                knots = np.array([1.0, 3.0, 6.0], dtype=float)
            return np.unique(knots)
        return np.unique(knots)

    knots_on = fixed_knots_on if fixed_knots_on is not None else \
        choose_knots(df.loc[df['t_on'] > 0, 't_on'].values, on_q)
    knots_wash = fixed_knots_wash if fixed_knots_wash is not None else \
        choose_knots(df.loc[(df['t_off'] > 0) & (df['ever_on'] == 1), 't_off'].values, wash_q)
    knots_never = fixed_knots_never if fixed_knots_never is not None else \
        choose_knots(df.loc[df['t_never'] > 0, 't_never'].values, never_q)

    B_on = make_basis(df['t_on'].values, knots_on, spline_type=spline_type)
    for j in range(B_on.shape[1]):
        df[f'A_on_sp{j+1}'] = B_on[:, j]

    B_wash = make_basis(df['t_off'].values, knots_wash, spline_type=spline_type)
    for j in range(B_wash.shape[1]):
        df[f'wash_sp{j+1}'] = B_wash[:, j]

    B_never = piecewise_basis(df['t_never'].values, knots_never)
    for j in range(B_never.shape[1]):
        df[f'never_sp{j+1}'] = B_never[:, j]

    return df, knots_on, knots_wash, knots_never


# Canonical TTE
def run_canonical_tte(df, trunc=TRUNC, censor_col='C_ltfu',
                      analysis_label='Canonical TTE', verbose=False):
    """Canonical TTE with IPCW."""
    work = df.copy()
    if censor_col not in work.columns:
        if censor_col == 'C_any' and 'C_ltfu' in work.columns:
            censor_col = 'C_ltfu'
        else:
            raise KeyError(f"censor_col '{censor_col}' not found in data")

    work['X_s'] = _zscore(work['X'])
    work['t_s'] = _zscore(work['t'])
    work['A_lag'] = work.groupby('id')['A'].shift(1).fillna(0)

    ps = 1.0 / (1.0 + np.exp(-work['X'].values))
    ps = np.clip(ps, 0.01, 0.99)
    p_z1 = work.groupby('id')['Z'].first().mean()
    w_iptw = np.where(work['Z'] == 1, p_z1 / ps, (1 - p_z1) / (1 - ps))

    if verbose:
        print(f"  IPTW: P(Z=1)={p_z1:.4f}, w_iptw mean={w_iptw.mean():.3f}, "
              f"sd={w_iptw.std():.3f}, range=[{w_iptw.min():.3f}, {w_iptw.max():.3f}]")

    tv_cols_present = [c for c in TV_CONFOUNDERS if c in work.columns]
    X_ltfu = add_constant(work[['t_s', 'A_lag'] + tv_cols_present])
    y_ltfu = 1 - work[censor_col]
    try:
        m_ltfu = sm.GLM(y_ltfu, X_ltfu, family=LINK_FAM).fit(disp=0)
        p_notC = m_ltfu.predict(X_ltfu)
        p_notC = np.clip(p_notC, 0.01, 1.0)
    except Exception:
        p_notC = np.ones(len(work))

    num_notC = work.groupby('t')[censor_col].transform(lambda s: 1 - s.mean())
    work['w_ltfu'] = num_notC / p_notC

    work['sw'] = _truncate_weights(w_iptw * work['w_ltfu'], trunc[0], trunc[1])

    c_prev = work.groupby('id')[censor_col].shift(1).fillna(0)
    work['censor_cum_prev'] = c_prev.groupby(work['id']).cumsum()
    work = work[work['censor_cum_prev'] == 0].copy()

    tfe = pd.get_dummies(work['t'].astype(int), prefix='t', drop_first=True, dtype=float)
    Xout = add_constant(pd.concat([work[['Z']].reset_index(drop=True),
                                    tfe.reset_index(drop=True)], axis=1))

    try:
        model = sm.GLM(work['Y'].values, Xout,
                       family=LINK_FAM,
                       freq_weights=work['sw'].values).fit(disp=0)
    except Exception as e:
        if verbose:
            print(f"  TTE model failed: {e}")
        return {'est_RD': np.nan, 'se_rd': np.nan}

    curves = {}
    for z_val in [0, 1]:
        Xpred = Xout.copy()
        Xpred['Z'] = z_val
        pred_h = model.predict(Xpred)
        pred_h = np.clip(pred_h, 0, 1)

        dfw_pred = work[['t', 'sw']].copy()
        dfw_pred['pred_h'] = pred_h

        gsum = (dfw_pred.assign(hw=dfw_pred['pred_h'] * dfw_pred['sw'])
                .groupby('t', as_index=False)
                .agg(hazard_sum=('hw', 'sum'), w_sum=('sw', 'sum')))
        gsum['hazard'] = gsum['hazard_sum'] / gsum['w_sum']
        gsum['cum_event'] = 1 - np.cumprod(1 - gsum['hazard'].values)
        curves[z_val] = gsum[['t', 'hazard', 'cum_event']].copy()

    final_0 = curves[0]['cum_event'].iloc[-1]
    final_1 = curves[1]['cum_event'].iloc[-1]
    est_rd = float(final_1 - final_0)

    try:
        V_theta = _cluster_robust_glm_cov(
            model=model, exog=Xout.values, endog=work['Y'].values,
            weights=work['sw'].values, groups=work['id'].values)

        def _rd_from_coefs(theta):
            cum = {}
            for z_val in [0, 1]:
                Xp = Xout.copy()
                Xp['Z'] = z_val
                lin = Xp.values @ theta
                h = 1 - np.exp(-np.exp(lin))
                h = np.clip(h, 0, 1)
                dp = work[['t', 'sw']].copy()
                dp['h'] = h
                gs = (dp.assign(hw=dp['h'] * dp['sw'])
                      .groupby('t', as_index=False)
                      .agg(hs=('hw', 'sum'), ws=('sw', 'sum')))
                gs['hz'] = gs['hs'] / gs['ws']
                cum[z_val] = 1 - np.prod(1 - gs['hz'].values)
            return cum[1] - cum[0]

        grad = _numerical_gradient(_rd_from_coefs, model.params.values)
        se_rd = float(np.sqrt(grad @ V_theta @ grad))
    except Exception:
        se_rd = np.nan

    if verbose:
        print(f"  [{analysis_label}] censor_col={censor_col}, RD = {est_rd:.4f}, SE = {se_rd:.4f}")

    return {
        'est_RD': est_rd, 'se_rd': se_rd,
        'curves': curves, 'model': model,
        'weights': work['sw'].describe().to_dict(),
        'analysis_label': analysis_label,
        'censor_col': censor_col,
        'z_definition': (
            f'tirzepatide within {GRACE} months vs SGLT2 within {GRACE} months; '
            'post-grace opposite-drug switch retained'
        )
    }


# SNMM g-estimation
def run_snmm_g(df, grace_len=GRACE, trunc=TRUNC,
               on_q=None, wash_q=None, never_q=NEVER_KNOT_Q,
               fixed_knots_on=None, fixed_knots_wash=None,
               fixed_knots_never=None, spline_type=SPLINE_TYPE,
               censor_col='C_ltfu', verbose=False):
    """SNMM g-estimation."""
    t0 = time.time()
    if censor_col not in df.columns:
        if censor_col == 'C_any' and 'C_ltfu' in df.columns:
            censor_col = 'C_ltfu'
        else:
            raise KeyError(f"censor_col '{censor_col}' not found in data")

    clones = pd.concat([df.assign(Z_clone=0), df.assign(Z_clone=1)])
    clones = clones.sort_values(['id', 'Z_clone', 't']).reset_index(drop=True)
    clones = _add_grace_strategy_censoring(clones, grace_len)

    work = clones.copy()
    work['X_s'] = _zscore(work['X'])
    work['t_s'] = _zscore(work['t'])
    work['A_lag'] = work.groupby(['id', 'Z_clone'])['A'].shift(1).fillna(0)
    tv_cols_present = [c for c in TV_CONFOUNDERS if c in work.columns]

    ps = 1.0 / (1.0 + np.exp(-work['X'].values))
    ps = np.clip(ps, 0.01, 0.99)
    p_z1 = work.groupby('id')['Z'].first().mean()
    w_iptw = np.where(work['Z'] == 1, p_z1 / ps, (1 - p_z1) / (1 - ps))

    if verbose:
        print(f"  IPTW: P(Z=1)={p_z1:.4f}, w_iptw mean={w_iptw.mean():.3f}, "
              f"sd={w_iptw.std():.3f}, range=[{w_iptw.min():.3f}, {w_iptw.max():.3f}]")

    X_ltfu = add_constant(work[['t_s', 'A_lag'] + tv_cols_present])
    y_ltfu = 1 - work[censor_col]
    try:
        m_ltfu = sm.GLM(y_ltfu, X_ltfu, family=LINK_FAM).fit(disp=0)
        p_notC = m_ltfu.predict(X_ltfu)
        p_notC = np.clip(p_notC, 0.01, 1.0)
    except Exception:
        p_notC = np.ones(len(work))

    num_notC = work.groupby('t')[censor_col].transform(lambda s: 1 - s.mean())
    work['w_ltfu'] = num_notC / p_notC

    grace_mask = work['t'] <= grace_len
    work['w_adherence'] = 1.0
    if grace_mask.any():
        df_ad = work.loc[grace_mask].copy()
        X_ad = add_constant(df_ad[['t_s', 'X_s', 'A_lag', 'Z_clone'] + tv_cols_present])
        y_ad = 1 - df_ad['C_adherence']
        try:
            m_ad = sm.GLM(y_ad, X_ad, family=LINK_FAM).fit(disp=0)
            p_notCad = np.clip(m_ad.predict(X_ad), 0.01, 1.0)
            num_notCad = df_ad.groupby(['t', 'Z_clone'])['C_adherence'].transform(
                lambda s: 1 - s.mean())
            work.loc[grace_mask, 'w_adherence'] = num_notCad.values / p_notCad
        except Exception:
            work.loc[grace_mask, 'w_adherence'] = 1.0

    work['sw'] = _truncate_weights(w_iptw * work['w_ltfu'] * work['w_adherence'], trunc[0], trunc[1])

    work['adh_cum'] = work.groupby(['id', 'Z_clone'])['C_adherence'].cumsum()
    work['C_ltfu_prev'] = work.groupby(['id', 'Z_clone'])[censor_col].shift(1).fillna(0)
    work['ltfu_cum_prev'] = work.groupby(['id', 'Z_clone'])['C_ltfu_prev'].cumsum()
    work = work[(work['adh_cum'] == 0) & (work['ltfu_cum_prev'] == 0)].copy()

    if verbose:
        print(f"  Post-censoring: {len(work):,} clone-person-months")
        for zc in [0, 1]:
            n = work[work['Z_clone'] == zc]['id'].nunique()
            print(f"    Z_clone={zc}: {n:,} patients")

    work, knots_on, knots_wash, knots_never = add_spline_timers_vec(
        work, grace_len, on_q=on_q, wash_q=wash_q, never_q=never_q,
        fixed_knots_on=fixed_knots_on, fixed_knots_wash=fixed_knots_wash,
        fixed_knots_never=fixed_knots_never, spline_type=spline_type)

    if verbose:
        print(f"  Knots ON: {knots_on}")
        print(f"  Knots WASH: {knots_wash}")

    analysis = work.copy()
    analysis['A_pre'] = analysis['A'] * (analysis['t'] <= grace_len).astype(int)
    analysis['pure_tzp_on'] = (
        (analysis['A'] == 1) &
        (analysis['A_sglt2'] == 0)
    ).astype(float)
    analysis['sglt2_with_or_after_tzp'] = (
        (analysis['A_sglt2'] == 1) &
        (analysis['ever_on'] == 1)
    ).astype(float)
    analysis['wash_active'] = (
        (analysis['A'] == 0) &
        (analysis['A_sglt2'] == 0) &
        (analysis['ever_on'] == 1) &
        (analysis['t_off'] > 0)
    ).astype(float)
    X_base = analysis[tv_cols_present + ['A_pre']].copy()
    tfe = pd.get_dummies(analysis['t'].astype(int), prefix='t', drop_first=True, dtype=float)
    Xnuis = add_constant(pd.concat([X_base.reset_index(drop=True),
                                     tfe.reset_index(drop=True)], axis=1))
    y = analysis['Y'].values.astype(float)

    try:
        nuis = sm.GLM(y, Xnuis, family=LINK_FAM,
                      freq_weights=analysis['sw'].values).fit(disp=0)
    except Exception as e:
        if verbose:
            print(f"  Nuisance model failed: {e}")
        return {'est_RD': np.nan, 'se_rd': np.nan, 'error': str(e)}

    lin0 = Xnuis.to_numpy() @ nuis.params.to_numpy()

    Xz = add_constant(analysis[['X_s']])
    try:
        z_model = sm.GLM(analysis['Z'].values, Xz, family=sm.families.Binomial()).fit(disp=0)
        z_hat = np.clip(z_model.predict(Xz), 0.01, 0.99)
    except Exception:
        z_hat = np.repeat(analysis['Z'].mean(), len(analysis))
    analysis['Z_resid'] = analysis['Z'].values - z_hat

    on_cols = sorted([c for c in analysis.columns if c.startswith('A_on_sp')])
    wash_cols = sorted([c for c in analysis.columns if c.startswith('wash_sp')])
    p_on = len(on_cols)
    p_wash = len(wash_cols)

    if p_on + p_wash == 0:
        if verbose:
            print("  No spline columns found!")
        return {'est_RD': np.nan, 'se_rd': np.nan, 'error': 'No splines'}

    S_on_p = analysis[on_cols].to_numpy()
    S_wash_p = analysis[wash_cols].to_numpy()
    p_switch = int(analysis.loc[analysis['t'].values > grace_len, 'sglt2_with_or_after_tzp'].sum() > 0)
    S_switch_p = (
        analysis[['sglt2_with_or_after_tzp']].to_numpy()
        if p_switch else np.zeros((len(analysis), 0))
    )

    N = len(analysis)
    p = p_on + p_wash + p_switch
    idx_on_end = p_on
    idx_wash_end = p_on + p_wash

    A_arr = analysis['pure_tzp_on'].values.astype(float)
    wash_gate = analysis['wash_active'].values.astype(float)
    w_arr = analysis['sw'].values
    Z_resid = analysis['Z_resid'].values
    t_arr = analysis['t'].values
    post_grace = t_arr > grace_len

    wZ = (w_arr * Z_resid)[:, None]
    if tv_cols_present:
        any_moud_arr = analysis[tv_cols_present].max(axis=1).values.astype(float)
    else:
        any_moud_arr = np.zeros(N)
    any_moud_post = any_moud_arr[post_grace]
    any_moud_cent = any_moud_arr - (any_moud_post.mean() if any_moud_post.sum() != 0 else 0)
    wZL = (w_arr * Z_resid * any_moud_cent)[:, None]

    def _compute_blip(beta):
        beta = np.asarray(beta, dtype=float)
        blip = np.zeros(N)
        if p_on:
            blip += A_arr * (S_on_p @ beta[:idx_on_end])
        if p_wash:
            blip += wash_gate * (S_wash_p @ beta[idx_on_end:idx_wash_end])
        if p_switch:
            blip += S_switch_p @ beta[idx_wash_end:]
        return blip

    def moment_matrix(beta):
        blip = _compute_blip(beta)
        mu = 1 - np.exp(-np.exp(lin0 + blip))
        mu = np.clip(mu, 0, 1)
        resid = (y - mu).reshape(-1, 1)
        g_on_z = wZ * (A_arr[:, None] * S_on_p * resid)
        g_wash_z = wZ * (wash_gate[:, None] * S_wash_p * resid)
        g_switch_z = wZ * (S_switch_p * resid)
        g_on_zl = wZL * (A_arr[:, None] * S_on_p * resid)
        g_wash_zl = wZL * (wash_gate[:, None] * S_wash_p * resid)
        g_switch_zl = wZL * (S_switch_p * resid)
        G = np.concatenate(
            [g_on_z, g_wash_z, g_switch_z, g_on_zl, g_wash_zl, g_switch_zl],
            axis=1
        )
        return G[post_grace]

    q = 2 * p

    def gmm_objective(beta, W):
        G = moment_matrix(beta)
        g_bar = G.mean(axis=0)
        return float(g_bar @ W @ g_bar)

    beta0 = np.zeros(p)

    W1 = np.eye(q)
    try:
        sol1 = minimize(gmm_objective, beta0, args=(W1,),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12})
    except Exception as e:
        if verbose:
            print(f"  GMM step 1 failed: {e}")
        return {'est_RD': np.nan, 'se_rd': np.nan, 'error': str(e)}

    G1 = moment_matrix(sol1.x)
    post_groups = analysis.loc[post_grace, 'id'].to_numpy()
    Omega = _cluster_omega(G1, post_groups)
    try:
        W_opt = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        W_opt = np.linalg.pinv(Omega)

    try:
        sol2 = minimize(gmm_objective, sol1.x, args=(W_opt,),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12})
        beta_hat = sol2.x
    except Exception as e:
        if verbose:
            print(f"  GMM step 2 failed: {e}")
        return {'est_RD': np.nan, 'se_rd': np.nan, 'error': str(e)}

    if np.any(np.abs(beta_hat) > 10):
        if verbose:
            print(f"  Warning: extreme coefficients detected: {beta_hat}")
        return {'est_RD': np.nan, 'se_rd': np.nan,
                'error': 'Extreme coefficients', 'beta': beta_hat}

    if verbose:
        switch_msg = beta_hat[idx_wash_end:] if p_switch else np.array([])
        print(
            f"  SNMM beta_hat: ON={beta_hat[:idx_on_end]}, "
            f"WASH={beta_hat[idx_on_end:idx_wash_end]}, "
            f"SGLT2-with/after-TZP={switch_msg}"
        )

    G_final = moment_matrix(beta_hat)
    n_obs = len(np.unique(post_groups))
    n_rows_pg = G_final.shape[0]
    g_bar = G_final.mean(axis=0)
    J_stat = n_rows_pg * float(g_bar @ W_opt @ g_bar)
    J_df = q - p
    J_pval = 1 - chi2.cdf(J_stat, J_df) if J_df > 0 else np.nan

    if verbose:
        print(f"  J-test: J={J_stat:.3f}, df={J_df}, p={J_pval:.4f}")

    V_beta = _cluster_gmm_cov(
        moment_matrix=moment_matrix, beta_hat=beta_hat,
        W_opt=W_opt, groups=post_groups)
    se_beta = np.sqrt(np.diag(np.abs(V_beta)))

    blip_all = _compute_blip(beta_hat)
    analysis['pred_h'] = 1 - np.exp(-np.exp(lin0 + blip_all))
    analysis['pred_h'] = np.clip(analysis['pred_h'], 0, 1)

    Zc_arr = analysis['Z_clone'].values
    curves = {}
    for z_val in [0, 1]:
        mask = Zc_arr == z_val
        gsum = (analysis.loc[mask].assign(hw=analysis.loc[mask, 'pred_h'] * analysis.loc[mask, 'sw'])
                .groupby('t', as_index=False)
                .agg(hazard_sum=('hw', 'sum'), w_sum=('sw', 'sum')))
        gsum['hazard'] = gsum['hazard_sum'] / gsum['w_sum']
        gsum['cum_event'] = 1 - np.cumprod(1 - gsum['hazard'].values)
        curves[z_val] = gsum[['t', 'hazard', 'cum_event']].copy()

    final_0 = curves[0]['cum_event'].iloc[-1]
    final_1 = curves[1]['cum_event'].iloc[-1]
    est_rd = float(final_1 - final_0)

    sw_arr_snmm = analysis['sw'].values
    t_arr_snmm = analysis['t'].values

    def _rd_from_beta(beta_v):
        blip_v = _compute_blip(beta_v)
        h_v = 1.0 - np.exp(-np.exp(lin0 + blip_v))
        h_v = np.clip(h_v, 0, 1)
        cum = {}
        for z_val in [0, 1]:
            mask_z = Zc_arr == z_val
            surv = 1.0
            for tv in sorted(np.unique(t_arr_snmm)):
                sel = mask_z & (t_arr_snmm == tv)
                if sel.any():
                    hz = np.sum(h_v[sel] * sw_arr_snmm[sel]) / np.sum(sw_arr_snmm[sel])
                else:
                    hz = 0.0
                surv *= (1.0 - hz)
            cum[z_val] = 1.0 - surv
        return cum[1] - cum[0]

    try:
        grad_rd = _numerical_gradient(_rd_from_beta, beta_hat)
        se_rd = float(np.sqrt(np.abs(grad_rd @ V_beta @ grad_rd)))
    except Exception:
        se_rd = np.nan

    if verbose:
        print(f"  [SNMM] censor_col={censor_col}, RD = {est_rd:.4f}, SE = {se_rd:.4f}")
        print(f"  Elapsed: {time.time()-t0:.0f}s")

    max_dur = T
    dur_grid = np.arange(0, max_dur + 1, dtype=float)
    on_beta = beta_hat[:idx_on_end]
    wash_beta = beta_hat[idx_on_end:idx_wash_end]
    V_on = V_beta[:idx_on_end, :idx_on_end] if idx_on_end > 0 else None
    V_wash = V_beta[idx_on_end:idx_wash_end, idx_on_end:idx_wash_end] if p_wash > 0 else None

    modeled_on, modeled_on_lo, modeled_on_hi = _evaluate_curve(
        dur_grid, knots_on, on_beta, cov=V_on, spline_type=spline_type)
    modeled_wash, modeled_wash_lo, modeled_wash_hi = _evaluate_curve(
        dur_grid, knots_wash, wash_beta, cov=V_wash, spline_type=spline_type)

    support_post = analysis.loc[
        post_grace,
        ['id', 'Y', 'sw', 'A', 'A_sglt2', 'pure_tzp_on', 'ever_on', 't_on', 't_off',
         'wash_active', 'sglt2_with_or_after_tzp']
    ].copy()
    support_on = support_post[(support_post['pure_tzp_on'] == 1) & (support_post['t_on'] > 0)].copy()
    support_wash = support_post[support_post['wash_active'] == 1].copy()
    support_switch = support_post[support_post['sglt2_with_or_after_tzp'] == 1].copy()
    duration_support = pd.concat([
        _summarize_duration_support(support_on, 't_on', 'ON'),
        _summarize_duration_support(support_wash, 't_off', 'WASHOUT'),
        _summarize_state_support(support_switch, 'SGLT2_WITH_OR_AFTER_TZP'),
    ], ignore_index=True)

    return {
        'est_RD': est_rd, 'se_rd': se_rd,
        'beta': beta_hat, 'se_beta': se_beta, 'V_beta': V_beta,
        'J_stat': J_stat, 'J_df': J_df, 'J_pval': J_pval,
        'curves': curves,
        'knots_on': knots_on, 'knots_wash': knots_wash,
        'spline_type': spline_type,
        'modeled_on': modeled_on.tolist(),
        'modeled_on_lo': modeled_on_lo.tolist(),
        'modeled_on_hi': modeled_on_hi.tolist(),
        'modeled_wash': modeled_wash.tolist(),
        'modeled_wash_lo': modeled_wash_lo.tolist(),
        'modeled_wash_hi': modeled_wash_hi.tolist(),
        'duration_support': duration_support,
        'p_on': p_on, 'p_wash': p_wash, 'p_switch': p_switch,
        'switch_effect': float(beta_hat[idx_wash_end]) if p_switch else np.nan,
        'switch_effect_se': float(se_beta[idx_wash_end]) if p_switch else np.nan,
        'weights': {'mean': float(analysis['sw'].mean()),
                    'std': float(analysis['sw'].std()),
                    'min': float(analysis['sw'].min()),
                    'max': float(analysis['sw'].max())},
        'censor_col': censor_col,
        'n_obs_post_grace': n_obs,
        'z_definition': (
            f'tirzepatide within {grace_len} months vs SGLT2 within {grace_len} months; '
            'post-grace opposite-drug switch retained'
        )
    }


# Cluster bootstrap
def _one_snmm_bootstrap(b_idx, df, knots_on, knots_wash, spline_type, seed,
                        grace_len=GRACE, censor_col='C_ltfu'):
    """Single bootstrap resample: redraw patients, re-fit SNMM with fixed knots."""
    rng = np.random.default_rng(seed)
    uniq_ids = df['id'].unique()
    n = len(uniq_ids)
    picked = rng.choice(uniq_ids, size=n, replace=True)

    pieces = []
    for new_id, src_id in enumerate(picked):
        sub = df[df['id'] == src_id].copy()
        sub['id'] = new_id
        pieces.append(sub)
    boot_df = pd.concat(pieces, ignore_index=True)

    try:
        res = run_snmm_g(
            boot_df, grace_len=grace_len, censor_col=censor_col,
            fixed_knots_on=knots_on,
            fixed_knots_wash=knots_wash,
            fixed_knots_never=None,
            spline_type=spline_type,
            verbose=False
        )
        if not np.isfinite(res.get('est_RD', np.nan)):
            return {'b': b_idx, 'ok': False, 'err': res.get('error', 'NaN RD')}
        return {
            'b': b_idx,
            'ok': True,
            'est_RD': res['est_RD'],
            'beta': np.asarray(res['beta']).tolist(),
            'modeled_on': res['modeled_on'],
            'modeled_wash': res['modeled_wash'],
        }
    except Exception as e:
        return {'b': b_idx, 'ok': False, 'err': str(e)}


def run_cluster_bootstrap(df, full_res, B=B_BOOT, seed=SEED, n_jobs=-1,
                          grace_len=GRACE, censor_col='C_ltfu',
                          checkpoint_path=None, verbose=True):
    """Cluster bootstrap on patient id with knots fixed at full-data values."""
    import pickle
    from joblib import Parallel, delayed

    t0 = time.time()
    knots_on = full_res['knots_on']
    knots_wash = full_res['knots_wash']
    spline_type = full_res['spline_type']
    max_on = len(full_res['modeled_on']) - 1
    max_wash = len(full_res['modeled_wash']) - 1

    rng_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=B)

    results_by_b = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            results_by_b = pickle.load(f)
        if verbose:
            print(f"  [BOOT] Loaded checkpoint with {len(results_by_b)} completed tasks", flush=True)

    remaining = [b for b in range(B) if b not in results_by_b]
    batch_size = max(n_jobs, 1) if n_jobs > 0 else 32
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_one_snmm_bootstrap)(
                b_idx=b, df=df,
                knots_on=knots_on, knots_wash=knots_wash,
                spline_type=spline_type, seed=int(rng_seeds[b]),
                grace_len=grace_len, censor_col=censor_col
            )
            for b in batch
        )
        for r in batch_results:
            results_by_b[r['b']] = r
        if checkpoint_path:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(results_by_b, f)
        if verbose:
            print(f"  [BOOT] {len(results_by_b)}/{B} resamples done "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    results = [results_by_b[b] for b in range(B)]
    ok_results = [r for r in results if r['ok']]
    fail_results = [r for r in results if not r['ok']]
    if verbose:
        print(f"  [BOOT] {len(ok_results)}/{B} converged; {len(fail_results)} failures", flush=True)
        if fail_results:
            print(f"  [BOOT] first 3 failures: {fail_results[:3]}", flush=True)
    if not ok_results:
        raise RuntimeError("All bootstrap replications failed")

    rds = np.array([r['est_RD'] for r in ok_results])
    curves_on = np.array([r['modeled_on'] for r in ok_results])
    curves_wash = np.array([r['modeled_wash'] for r in ok_results])

    curve_rows = []
    for comp, curves, point, alo, ahi, max_d in [
        ('ON', curves_on, full_res['modeled_on'],
         full_res['modeled_on_lo'], full_res['modeled_on_hi'], max_on),
        ('WASH', curves_wash, full_res['modeled_wash'],
         full_res['modeled_wash_lo'], full_res['modeled_wash_hi'], max_wash),
    ]:
        lo = np.quantile(curves, 0.025, axis=0)
        hi = np.quantile(curves, 0.975, axis=0)
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        for i in range(max_d + 1):
            curve_rows.append({
                'component': comp, 'duration': i,
                'point_estimate': float(point[i]),
                'analytical_lo': float(alo[i]), 'analytical_hi': float(ahi[i]),
                'boot_mean': float(mean[i]), 'boot_sd': float(sd[i]),
                'boot_lo_2p5': float(lo[i]), 'boot_hi_97p5': float(hi[i]),
            })

    summary = {
        'point': float(full_res['est_RD']),
        'analytical_SE': float(full_res['se_rd']),
        'bootstrap_SE': float(rds.std(ddof=1)),
        'bootstrap_mean': float(rds.mean()),
        'bootstrap_lo_2p5': float(np.quantile(rds, 0.025)),
        'bootstrap_hi_97p5': float(np.quantile(rds, 0.975)),
        'B_total': B,
        'B_converged': len(ok_results),
    }
    if verbose:
        print(f"  [BOOT] RD = {summary['point']:.5f}, boot SE = {summary['bootstrap_SE']:.5f}, "
              f"95% percentile CI = ({summary['bootstrap_lo_2p5']:.5f}, "
              f"{summary['bootstrap_hi_97p5']:.5f})", flush=True)
        print(f"  [BOOT] Total runtime {(time.time()-t0)/60:.1f} min", flush=True)

    return {'summary': summary, 'curve_bands': pd.DataFrame(curve_rows),
            'rd_draws': rds}


# Figures
def plot_figure4(tte_result, snmm_result, T_val=T, grace_len=GRACE,
                 save_path=None):
    """Figure 4 cumulative incidence plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {0: 'C0', 1: 'C1'}
    labels = {0: 'Z=0 (SGLT2 within grace)', 1: 'Z=1 (Tirzepatide within grace)'}

    for ax, (title, result) in zip(axes, [
        ('Canonical TTE', tte_result),
        ('SNMM (g-estimation)', snmm_result)
    ]):
        if result.get('curves') is None:
            ax.set_title(f'{title}\n(estimation failed)')
            continue
        for z in [0, 1]:
            c = result['curves'][z]
            ax.plot(c['t'], c['cum_event'], color=colors[z],
                    label=labels[z], linewidth=2)
        ax.axvline(x=grace_len, color='gray', ls='--', lw=1, alpha=0.5,
                   label=f'Grace period (G={grace_len})')
        ax.set_xlabel('Month')
        ax.set_ylabel('Cumulative Incidence of Opioid Overdose')
        ax.set_xticks(np.arange(1, T_val + 1))
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Figure 4: ITT Cumulative Incidence - Tirzepatide vs SGLT2',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_figure5(snmm_result, T_val=T, save_path=None, boot=None):
    """Figure 5 duration curves."""
    if snmm_result.get('modeled_on') is None:
        print("  Cannot plot Figure 5: SNMM failed")
        return

    on_lo = snmm_result.get('modeled_on_lo')
    on_hi = snmm_result.get('modeled_on_hi')
    wash_lo = snmm_result.get('modeled_wash_lo')
    wash_hi = snmm_result.get('modeled_wash_hi')
    band_label = '95% CI'
    ci_note = 'analytical 95% CIs'
    if boot is not None:
        bands = boot['curve_bands']
        on_b = bands[bands['component'] == 'ON'].sort_values('duration')
        wash_b = bands[bands['component'] == 'WASH'].sort_values('duration')
        on_lo, on_hi = on_b['boot_lo_2p5'].values, on_b['boot_hi_97p5'].values
        wash_lo, wash_hi = wash_b['boot_lo_2p5'].values, wash_b['boot_hi_97p5'].values
        band_label = '95% bootstrap percentile CI'
        ci_note = 'cluster-bootstrap percentile 95% CIs'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(0, T_val + 1)
    knots_on = snmm_result.get('knots_on', np.array([]))
    knots_wash = snmm_result.get('knots_wash', np.array([]))

    ax = axes[0]
    if on_lo is not None:
        ax.fill_between(x, on_lo, on_hi, color='C0', alpha=0.20, label=band_label)
    ax.plot(x, snmm_result['modeled_on'], 'C0-', lw=2.0, label='SNMM estimate')
    ax.axhline(0, color='k', lw=0.8)
    for k in knots_on:
        ax.axvline(k, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Duration (months)', fontsize=11)
    ax.set_ylabel('Dynamic effect (cloglog scale)', fontsize=11)
    ax.set_xticks(np.arange(0, T_val + 1))
    ax.set_title('ON (tirzepatide)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if wash_lo is not None:
        ax.fill_between(x, wash_lo, wash_hi, color='C1', alpha=0.20, label=band_label)
    ax.plot(x, snmm_result['modeled_wash'], 'C1-', lw=2.0, label='SNMM estimate')
    ax.axhline(0, color='k', lw=0.8)
    for k in knots_wash:
        ax.axvline(k, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Duration (months)', fontsize=11)
    ax.set_xticks(np.arange(0, T_val + 1))
    ax.set_title('WASHOUT (off both after tirzepatide)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        'Figure 5: SNMM-Estimated Dynamic Treatment Effects\n'
        f'(Tirzepatide vs SGLT2, {ci_note})',
        fontsize=13, y=1.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_duration_support(snmm_result, T_val=T, save_path=None):
    """Duration support plot."""
    support = snmm_result.get('duration_support')
    if support is None or len(support) == 0:
        print("  Cannot plot duration support: diagnostics unavailable")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    colors = {'ON': 'C0', 'WASHOUT': 'C1', 'SGLT2_WITH_OR_AFTER_TZP': 'C2'}
    titles = {
        'ON': 'ON duration support',
        'WASHOUT': 'Off-both washout support',
        'SGLT2_WITH_OR_AFTER_TZP': 'SGLT2 with/after TZP support',
    }

    for ax, component in zip(axes[:2], ['ON', 'WASHOUT']):
        sub = support[support['component'] == component].copy()
        idx = np.arange(1, T_val + 1)
        sub = sub.set_index('duration').reindex(idx, fill_value=0).reset_index()

        ax.bar(
            sub['duration'],
            sub['n_clone_person_periods'],
            width=0.8,
            color=colors[component],
            alpha=0.75,
            label='Clone-person-periods'
        )
        ax.set_xlabel('Duration (months)')
        ax.set_ylabel('Clone-person-periods')
        ax.set_title(titles[component])
        ax.set_xticks(idx)
        ax.grid(alpha=0.25, axis='y')

        ax2 = ax.twinx()
        ax2.plot(
            sub['duration'],
            sub['effective_n'],
            color='black',
            marker='o',
            ms=3.5,
            lw=1.4,
            label='Effective n'
        )
        ax2.set_ylabel('Effective n')
        ax2.set_ylim(bottom=0)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=9, loc='upper right')

    ax = axes[2]
    sub = support[support['component'] == 'SGLT2_WITH_OR_AFTER_TZP'].copy()
    if sub.empty:
        n_periods = 0
        eff_n = 0
        n_events = 0
    else:
        n_periods = float(sub['n_clone_person_periods'].iloc[0])
        eff_n = float(sub['effective_n'].iloc[0])
        n_events = float(sub['n_events'].iloc[0])
    ax.bar([0], [n_periods], width=0.55, color=colors['SGLT2_WITH_OR_AFTER_TZP'],
           alpha=0.75, label='Clone-person-periods')
    ax.set_xticks([0])
    ax.set_xticklabels(['Retained state'])
    ax.set_ylabel('Clone-person-periods')
    ax.set_title(titles['SGLT2_WITH_OR_AFTER_TZP'])
    ax.grid(alpha=0.25, axis='y')
    ax2 = ax.twinx()
    ax2.plot([0], [eff_n], color='black', marker='o', ms=4, lw=0,
             label='Effective n')
    ax2.set_ylabel('Effective n')
    ax2.set_ylim(bottom=0)
    ax.text(
        0, n_periods * 0.95 if n_periods else 0,
        f'Events: {int(n_events)}',
        ha='center', va='top', fontsize=9
    )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=9, loc='upper right')

    fig.suptitle('Duration support / positivity diagnostics', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_results_table(tte_result, snmm_result, boot=None):
    """Table 4 results."""
    rows = []
    rd_tte = tte_result.get('est_RD', np.nan)
    se_tte = tte_result.get('se_rd', np.nan)
    z_def = tte_result.get('z_definition') or snmm_result.get('z_definition') or '-'
    rows.append({
        'Estimator': 'Canonical TTE',
        'RD': f'{rd_tte:.4f}' if not np.isnan(rd_tte) else '-',
        'SE': f'{se_tte:.4f}' if not np.isnan(se_tte) else '-',
        '95% CI': f'({rd_tte-1.96*se_tte:.4f}, {rd_tte+1.96*se_tte:.4f})'
            if not np.isnan(rd_tte) and not np.isnan(se_tte) else '-',
        'CI basis': 'analytical',
        'J-stat': '-', 'J p-value': '-', 'Z definition': z_def
    })

    rd_snmm = snmm_result.get('est_RD', np.nan)
    se_snmm = snmm_result.get('se_rd', np.nan)
    j_stat = snmm_result.get('J_stat', np.nan)
    j_pval = snmm_result.get('J_pval', np.nan)
    if boot is not None:
        s = boot['summary']
        ci_snmm = f"({s['bootstrap_lo_2p5']:.4f}, {s['bootstrap_hi_97p5']:.4f})"
        se_str = f"{s['bootstrap_SE']:.4f}"
        ci_basis = f"cluster-bootstrap percentile ({s['B_converged']}/{s['B_total']} converged)"
    else:
        ci_snmm = (f'({rd_snmm-1.96*se_snmm:.4f}, {rd_snmm+1.96*se_snmm:.4f})'
                   if not np.isnan(rd_snmm) and not np.isnan(se_snmm) else '-')
        se_str = f'{se_snmm:.4f}' if not np.isnan(se_snmm) else '-'
        ci_basis = 'analytical'
    rows.append({
        'Estimator': 'SNMM (g-estimation)',
        'RD': f'{rd_snmm:.4f}' if not np.isnan(rd_snmm) else '-',
        'SE': se_str,
        '95% CI': ci_snmm,
        'CI basis': ci_basis,
        'J-stat': f'{j_stat:.3f}' if not np.isnan(j_stat) else '-',
        'J p-value': f'{j_pval:.4f}' if not np.isnan(j_pval) else '-',
        'Z definition': z_def
    })

    if 'beta' in snmm_result and snmm_result['beta'] is not None:
        beta = snmm_result['beta']
        se_b = snmm_result.get('se_beta', np.full_like(beta, np.nan))
        p_on = snmm_result.get('p_on', 0)
        for i in range(p_on):
            rows.append({
                'Estimator': f'  ON spline coef {i+1}',
                'RD': f'{beta[i]:.4f}', 'SE': f'{se_b[i]:.4f}',
                '95% CI': f'({beta[i]-1.96*se_b[i]:.4f}, {beta[i]+1.96*se_b[i]:.4f})',
                'CI basis': 'analytical', 'J-stat': '', 'J p-value': '', 'Z definition': ''
            })
        p_wash = snmm_result.get('p_wash', 0)
        for i in range(p_wash):
            idx = p_on + i
            rows.append({
                'Estimator': f'  WASH spline coef {i+1}',
                'RD': f'{beta[idx]:.4f}', 'SE': f'{se_b[idx]:.4f}',
                '95% CI': f'({beta[idx]-1.96*se_b[idx]:.4f}, {beta[idx]+1.96*se_b[idx]:.4f})',
                'CI basis': 'analytical', 'J-stat': '', 'J p-value': '', 'Z definition': ''
            })
        p_switch = snmm_result.get('p_switch', 0)
        if p_switch:
            idx = p_on + p_wash
            rows.append({
                'Estimator': '  SGLT2 with/after TZP state coef',
                'RD': f'{beta[idx]:.4f}', 'SE': f'{se_b[idx]:.4f}',
                '95% CI': f'({beta[idx]-1.96*se_b[idx]:.4f}, {beta[idx]+1.96*se_b[idx]:.4f})',
                'CI basis': 'analytical', 'J-stat': '', 'J p-value': '', 'Z definition': ''
            })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("=" * 70)
    print("TIRZEPATIDE / OUD - DTTE APPLICATION")
    print("=" * 70)
    print(f"  Treatment: Tirzepatide vs SGLT2 (active comparator)")
    print(f"  Outcome:   Opioid overdose (T40.0-T40.4, T40.6)")
    print(f"  Grace:     {GRACE} months")
    print(f"  Follow-up: {T} months")

    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    df = load_and_prepare(verbose=True, include_baseline_covs=True)

    print("\n" + "=" * 70)
    print("STEP 2: RUNNING ESTIMATORS")
    print("=" * 70)

    print("\n--- Canonical TTE ---")
    tte_result = run_canonical_tte(
        df, trunc=TRUNC, censor_col='C_ltfu',
        analysis_label='Canonical TTE (treatment-policy: switch retained)',
        verbose=True)

    print("\n--- SNMM g-estimation ---")
    snmm_result = run_snmm_g(
        df, grace_len=GRACE, trunc=TRUNC, censor_col='C_ltfu',
        verbose=True)

    print(f"\n--- Cluster bootstrap (B={B_BOOT}) ---")
    boot_result = run_cluster_bootstrap(
        df, snmm_result, B=B_BOOT, seed=SEED, n_jobs=-1,
        checkpoint_path=os.path.join(BASE_OUTDIR,
                                     'Tirzepatide_Duration_Bootstrap_Checkpoint.pkl'),
        verbose=True)

    boot_result['curve_bands'].to_csv(
        os.path.join(BASE_OUTDIR, 'Tirzepatide_Duration_Response_Bootstrap.csv'),
        index=False)
    pd.DataFrame([dict(estimate='SNMM RD (18-month)', **boot_result['summary'])]).to_csv(
        os.path.join(BASE_OUTDIR, 'Tirzepatide_RD_Bootstrap_Summary.csv'),
        index=False)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    table4 = build_results_table(tte_result, snmm_result, boot=boot_result)
    table4.to_csv(os.path.join(BASE_OUTDIR, 'Table4_Tirzepatide_OUD_Results.csv'),
                  index=False)
    print("\nTable 4: Tirzepatide / OUD Application Results")
    print(table4.to_string(index=False))

    if 'weights' in snmm_result:
        print(f"\nSNMM Weight diagnostics: {snmm_result['weights']}")
    if 'weights' in tte_result:
        print(f"TTE Weight diagnostics: {tte_result['weights']}")

    balance_table = build_empirical_balance_table(df)
    if len(balance_table) > 0:
        balance_table.to_csv(
            os.path.join(BASE_OUTDIR, 'Tirzepatide_Baseline_Balance.csv'),
            index=False
        )
        print("Baseline balance table saved.")

    if snmm_result.get('modeled_on') is not None:
        durations = list(range(0, T + 1))
        dur_df = pd.DataFrame({
            'duration': durations,
            'on_effect': snmm_result['modeled_on'],
            'on_effect_lo': snmm_result.get('modeled_on_lo'),
            'on_effect_hi': snmm_result.get('modeled_on_hi'),
            'wash_effect': snmm_result['modeled_wash'],
            'wash_effect_lo': snmm_result.get('modeled_wash_lo'),
            'wash_effect_hi': snmm_result.get('modeled_wash_hi')
        })
        dur_df.to_csv(os.path.join(BASE_OUTDIR,
                      'Tirzepatide_Duration_Response_Curves.csv'), index=False)
        print("\nDuration response curves saved.")

    if snmm_result.get('duration_support') is not None:
        snmm_result['duration_support'].to_csv(
            os.path.join(BASE_OUTDIR, 'Tirzepatide_Duration_Support.csv'),
            index=False
        )
        print("Duration support diagnostics saved.")

    ci_rows = []
    for label, result in [('Canonical TTE', tte_result), ('SNMM', snmm_result)]:
        if result.get('curves') is not None:
            for z_val in [0, 1]:
                c = result['curves'][z_val]
                for _, row in c.iterrows():
                    ci_rows.append({
                        'estimator': label, 'Z': z_val,
                        't': int(row['t']),
                        'hazard': row['hazard'],
                        'cum_incidence': row['cum_event']
                    })
    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        ci_df.to_csv(os.path.join(BASE_OUTDIR,
                     'Tirzepatide_Cumulative_Incidence_Curves.csv'), index=False)
        print("Cumulative incidence curves saved.")

    print("\nGenerating figures...")
    plot_figure4(
        tte_result, snmm_result, T_val=T, grace_len=GRACE,
        save_path=os.path.join(BASE_OUTDIR,
                               'Figure4_Tirzepatide_ITT_Cumulative_Incidence.png'))
    plot_figure5(
        snmm_result, T_val=T, boot=boot_result,
        save_path=os.path.join(BASE_OUTDIR,
                               'Figure5_Tirzepatide_Duration_Effects.png'))
    plot_duration_support(
        snmm_result, T_val=T,
        save_path=os.path.join(BASE_OUTDIR,
                               'FigureA_Tirzepatide_Duration_Support.png'))

    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print(f"  {INTERMEDIARY_PATH}")
    print(f"  Table4_Tirzepatide_OUD_Results.csv")
    print(f"  Tirzepatide_Baseline_Balance.csv")
    print(f"  Tirzepatide_PS_Diagnostics.csv")
    print(f"  Tirzepatide_Duration_Response_Curves.csv")
    print(f"  Tirzepatide_Duration_Support.csv")
    print(f"  Tirzepatide_Cumulative_Incidence_Curves.csv")
    print(f"  Figure4_Tirzepatide_ITT_Cumulative_Incidence.png")
    print(f"  Figure5_Tirzepatide_Duration_Effects.png")
    print(f"  FigureA_Tirzepatide_Duration_Support.png")
    print("=" * 70)
    print("DONE")
