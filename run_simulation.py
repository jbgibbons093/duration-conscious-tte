"""Simulation study for duration-conscious target trial emulation."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
import os
from scipy.stats import chi2, norm
from joblib import Parallel, delayed

# Paths
BASE_OUTDIR = os.path.dirname(os.path.abspath(__file__))
APPENDIX_OUTDIR = os.path.join(BASE_OUTDIR, "appendix")

os.makedirs(BASE_OUTDIR, exist_ok=True)
os.makedirs(APPENDIX_OUTDIR, exist_ok=True)


# Settings
N             = 10000
T             = 12
GRACE         = 3
CONF_STRENGTH = 1
B_BOOT        = 100
M_SIMS        = 100
B_MC_BOOT     = 50
FIX_KNOTS_IN_MC = True
RUN_MC        = True
TRUNC         = (0.01, 0.99)
SEED          = 123
ON_KNOT_Q    = [0.10, 0.50, 0.90]
WASH_KNOT_Q  = [0.10, 0.50, 0.90]
NEVER_KNOT_Q = [0.33, 0.67]
SPLINE_TYPE  = "rcs"
LINK_FAM  = sm.families.Binomial(link=sm.families.links.CLogLog())

# Utilities
def _zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)

def _truncate_weights(w, lo=0.025, hi=0.975):
    w = np.asarray(w, float)
    finite = np.isfinite(w)
    a, b = np.quantile(w[finite], [lo, hi])
    return np.clip(w, a, b)

def _sanitize_design(X):
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(float)

def inv_cloglog(eta):
    """Inverse complementary log-log link."""
    return 1.0 - np.exp(-np.exp(eta))


# Data generating process
def simulate_trial_data(
    N=3000, T=10, grace_len=3, conf_strength=CONF_STRENGTH, seed=123,
    base_logitA=-1.00, beta_Z=0.60, beta_L_to_A=-0.20, beta_X_to_A=0.15, beta_U_to_A=0.25,
    rho_A=0.50, psi_off=-0.80, sd_eta_A=0, sd_tau_A=0,
    base_logitY=-3.00, beta_L_to_Y=0.70, beta_A_to_Y=-0.70, beta_U_to_Y=0.45, sd_eta_Y=0,
    base_logit_notC=0.70, gamma_A=0.20, gamma_L=-0.15, sd_eta_C=0, sd_tau_C=0,
    p_dev_base=0.10, p_dev_L_coef=0.05,
    knot=3, dyn_on1=-0.22, dyn_on2=-0.10, dyn_off1=0.18, dyn_off2=0.10,
    dyn_never1=0.0, dyn_never2=0.0,
    restart_delta_on=0.0, restart_delta_wash=0.0,
    phi_U=0.70, sd_U=0.60
):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, N)
    V = rng.normal(0, 1, N)
    Z = rng.binomial(1, 0.5, N)

    eta_A = rng.normal(0, sd_eta_A, N)
    eta_Y = rng.normal(0, sd_eta_Y, N)
    eta_C = rng.normal(0, sd_eta_C, N)
    tau_A = rng.normal(0, sd_tau_A, T+1)
    tau_C = rng.normal(0, sd_tau_C, T+1)

    U_epsL = rng.normal(0, 1, (N, T))
    U_dev  = rng.uniform(0, 1, (N, T))
    U_Au   = rng.uniform(0, 1, (N, T))
    U_Yu   = rng.uniform(0, 1, (N, T))
    U_Cu   = rng.uniform(0, 1, (N, T))

    U_path = rng.normal(0, sd_U, (N, T))
    for i in range(N):
        for t in range(1, T):
            U_path[i, t] = phi_U * U_path[i, t-1] + rng.normal(0, sd_U)

    L_init_noise = rng.normal(0, 1, N)

    records = []
    for i in range(N):
        L_prev = 0.3 * X[i] + 0.3 * V[i] + L_init_noise[i]
        alive  = 1
        A_prev = 0
        ever_on, t_on, t_off, t_never = 0, 0, 0, 0
        d_on_prev, d_off_prev = 0, 0

        for t in range(1, T+1):
            if alive == 0:
                break

            U_t = U_path[i, t-1]
            eps_L = U_epsL[i, t-1]
            if t == 1:
                L = 0.35 * X[i] + 0.35 * V[i] + 0.20 * U_t + eps_L
            else:
                L = (0.45 * L_prev + 0.25 * X[i] + 0.20 * V[i] +
                     0.20 * U_t - 0.35 * A_prev + eps_L)

            if t <= grace_len:
                p_dev = p_dev_base + p_dev_L_coef * (L > 1.2)
                p_dev = float(np.minimum(np.maximum(p_dev, 0.05), 0.25))
                A = Z[i] if U_dev[i, t-1] > p_dev else (1 - Z[i])
            else:
                etaA = (base_logitA + beta_Z * Z[i] + beta_L_to_A * L +
                        beta_X_to_A * X[i] + beta_U_to_A * conf_strength * U_t +
                        rho_A * A_prev + psi_off * (1 if t_off > 0 else 0) +
                        eta_A[i] + tau_A[t])
                pA = inv_cloglog(etaA)
                A  = 1 if U_Au[i, t-1] < pA else 0

            if A == 1:
                if A_prev == 0 and ever_on == 1 and t_off > 0:
                    d_off_prev = t_off
                t_on += 1
                t_off = 0
                t_never = 0
                ever_on = 1
            else:
                if A_prev == 1 and t_on > 0:
                    d_on_prev = t_on
                t_on = 0
                if ever_on:
                    t_off += 1
                    t_never = 0
                else:
                    t_off = 0
                    t_never += 1

            s_on1  = min(t_on, knot)
            s_on2  = max(t_on - knot, 0)
            s_off1 = min(t_off, knot)
            s_off2 = max(t_off - knot, 0)
            s_nev1 = min(t_never, knot)
            s_nev2 = max(t_never - knot, 0)
            add_dyn = (A * (dyn_on1 * s_on1 + dyn_on2 * s_on2) +
                       (1 - A) * ever_on * (dyn_off1 * s_off1 + dyn_off2 * s_off2) +
                       (1 - A) * (1 - ever_on) * (dyn_never1 * s_nev1 + dyn_never2 * s_nev2))
            if restart_delta_on != 0 and A == 1 and d_off_prev > 0:
                add_dyn += restart_delta_on * d_off_prev
            if restart_delta_wash != 0 and A == 0 and ever_on == 1 and d_on_prev > 0:
                add_dyn += restart_delta_wash * d_on_prev

            etaY = (base_logitY + beta_L_to_Y * L + beta_A_to_Y * A +
                    beta_U_to_Y * conf_strength * U_t + eta_Y[i] + add_dyn)
            pY = inv_cloglog(etaY)
            Y  = 1 if U_Yu[i, t-1] < pY else 0

            eta_notC = (base_logit_notC + gamma_A * A_prev + gamma_L * L +
                        eta_C[i] + tau_C[t])
            p_notC = inv_cloglog(eta_notC)
            notC   = 1 if U_Cu[i, t-1] < p_notC else 0
            C_ltfu = 1 - notC

            alive = 0 if (Y == 1 or C_ltfu == 1) else 1
            records.append({
                "id": i, "t": t,
                "X": X[i], "V": V[i], "Z": Z[i],
                "L": L, "A": A, "Y": Y, "C_ltfu": C_ltfu, "alive": alive
            })
            L_prev = L
            A_prev = A

    df = pd.DataFrame(records)

    snap = df[df["t"] == 1].copy()
    corr_UL = np.corrcoef(U_path[:, 0], snap["L"].to_numpy())[0, 1]
    print("\nDGP checks:")
    print("  P(A=1) overall:", round(df['A'].mean(), 3))
    print("  P(A=1 | Z):", df.groupby('Z')['A'].mean().round(3).to_dict())
    print("  P(Y=1) overall:", round(df['Y'].mean(), 3))
    print("  P(LTFU=1) overall:", round(df['C_ltfu'].mean(), 3))
    print("  Corr(X,L)~", round(df[['X','L']].corr().iloc[0,1], 3))
    print("  Corr(U_t, L) at t=1 ~", round(float(corr_UL), 3))

    draws = dict(
        X=X, V=V, Z=Z,
        eta_A=eta_A, eta_Y=eta_Y, eta_C=eta_C,
        tau_A=tau_A, tau_C=tau_C,
        U_epsL=U_epsL, U_dev=U_dev, U_Au=U_Au, U_Yu=U_Yu, U_Cu=U_Cu,
        U_path=U_path,
        L_init_noise=L_init_noise
    )
    dgp = dict(
        N=N, T=T, grace_len=grace_len, conf_strength=conf_strength,
        base_logitA=base_logitA, beta_Z=beta_Z, beta_L_to_A=beta_L_to_A,
        beta_X_to_A=beta_X_to_A, beta_U_to_A=beta_U_to_A,
        rho_A=rho_A, psi_off=psi_off,
        base_logitY=base_logitY, beta_L_to_Y=beta_L_to_Y,
        beta_A_to_Y=beta_A_to_Y, beta_U_to_Y=beta_U_to_Y,
        base_logit_notC=base_logit_notC, gamma_A=gamma_A, gamma_L=gamma_L,
        p_dev_base=p_dev_base, p_dev_L_coef=p_dev_L_coef,
        knot=knot, dyn_on1=dyn_on1, dyn_on2=dyn_on2,
        dyn_off1=dyn_off1, dyn_off2=dyn_off2,
        dyn_never1=dyn_never1, dyn_never2=dyn_never2,
        restart_delta_on=restart_delta_on, restart_delta_wash=restart_delta_wash
    )
    return df, draws, dgp

def compute_true_itt(draws, dgp):
    """Twin-world ITT under fixed Z=1 versus fixed Z=0."""

    N = int(dgp["N"])
    T = int(dgp["T"])

    X            = draws["X"]
    V            = draws["V"]
    eta_A        = draws["eta_A"]
    eta_Y        = draws["eta_Y"]
    tau_A        = draws["tau_A"]
    U_epsL       = draws["U_epsL"]
    U_dev        = draws["U_dev"]
    U_Au         = draws["U_Au"]
    U_Yu         = draws["U_Yu"]
    U_path       = draws["U_path"]
    L_init_noise = draws["L_init_noise"]

    base_logitA     = dgp["base_logitA"]
    beta_Z          = dgp["beta_Z"]
    beta_L_to_A     = dgp["beta_L_to_A"]
    beta_X_to_A     = dgp["beta_X_to_A"]
    beta_U_to_A     = dgp["beta_U_to_A"]
    rho_A           = dgp["rho_A"]
    psi_off         = dgp["psi_off"]

    base_logitY     = dgp["base_logitY"]
    beta_L_to_Y     = dgp["beta_L_to_Y"]
    beta_A_to_Y     = dgp["beta_A_to_Y"]
    beta_U_to_Y     = dgp["beta_U_to_Y"]

    p_dev_base      = dgp["p_dev_base"]
    p_dev_L_coef    = dgp["p_dev_L_coef"]

    knot            = dgp["knot"]
    dyn_on1         = dgp["dyn_on1"]
    dyn_on2         = dgp["dyn_on2"]
    dyn_off1        = dgp["dyn_off1"]
    dyn_off2        = dgp["dyn_off2"]
    dyn_never1      = dgp.get("dyn_never1", 0.0)
    dyn_never2      = dgp.get("dyn_never2", 0.0)

    conf_strength   = dgp["conf_strength"]
    grace_len       = int(dgp["grace_len"])
    restart_delta_on   = dgp.get("restart_delta_on", 0.0)
    restart_delta_wash = dgp.get("restart_delta_wash", 0.0)

    def cloglog_inv(eta):
        return 1.0 - np.exp(-np.exp(eta))

    def rollout(zval):
        t_on    = np.zeros(N)
        t_off   = np.zeros(N)
        t_never = np.zeros(N)
        ever_on = np.zeros(N, dtype=int)
        d_on_prev  = np.zeros(N)
        d_off_prev = np.zeros(N)

        ever_event = np.zeros(N, dtype=int)

        L_prev = 0.3 * X + 0.3 * V + L_init_noise
        A_prev = np.zeros(N, dtype=int)

        cum_incidence = []

        for t in range(1, T + 1):
            U_t = U_path[:, t-1]
            eps_L = U_epsL[:, t-1]

            if t == 1:
                L = 0.35 * X + 0.35 * V + 0.20 * U_t + eps_L
            else:
                L = (0.45 * L_prev + 0.25 * X + 0.20 * V +
                     0.20 * U_t - 0.35 * A_prev + eps_L)

            if t <= grace_len:
                p_dev = p_dev_base + p_dev_L_coef * (L > 1.2)
                p_dev = np.clip(p_dev, 0.05, 0.25)
                A = np.where(U_dev[:, t-1] > p_dev, zval, 1 - zval).astype(int)
            else:
                etaA = (base_logitA + beta_Z * zval + beta_L_to_A * L +
                        beta_X_to_A * X + beta_U_to_A * conf_strength * U_t +
                        rho_A * A_prev + psi_off * (t_off > 0).astype(float) +
                        eta_A + tau_A[t])
                pA = cloglog_inv(etaA)
                A  = (U_Au[:, t-1] < pA).astype(int)

            new_on  = (A == 1)
            new_off = (A == 0)

            restart_mask = new_on & (A_prev == 0) & (ever_on == 1) & (t_off > 0)
            d_off_prev = np.where(restart_mask, t_off, d_off_prev)
            stop_mask = new_off & (A_prev == 1) & (t_on > 0)
            d_on_prev = np.where(stop_mask, t_on, d_on_prev)

            t_on  = np.where(new_on, t_on + 1, 0)
            t_off = np.where(new_off & (ever_on == 1), t_off + 1, np.where(new_off, t_off, 0))
            t_never = np.where(new_off & (ever_on == 0), t_never + 1, 0)
            ever_on = np.maximum(ever_on, new_on.astype(int))

            s_on1  = np.minimum(t_on, knot)
            s_on2  = np.maximum(t_on - knot, 0)
            s_off1 = np.minimum(t_off, knot)
            s_off2 = np.maximum(t_off - knot, 0)
            s_nev1 = np.minimum(t_never, knot)
            s_nev2 = np.maximum(t_never - knot, 0)
            add_dyn = (A * (dyn_on1 * s_on1 + dyn_on2 * s_on2) +
                       (1 - A) * ever_on * (dyn_off1 * s_off1 + dyn_off2 * s_off2) +
                       (1 - A) * (1 - ever_on) * (dyn_never1 * s_nev1 + dyn_never2 * s_nev2))
            if restart_delta_on != 0:
                add_dyn = add_dyn + A * (d_off_prev > 0) * restart_delta_on * d_off_prev
            if restart_delta_wash != 0:
                add_dyn = add_dyn + (1 - A) * ever_on * (d_on_prev > 0) * restart_delta_wash * d_on_prev

            etaY = (base_logitY + beta_L_to_Y * L + beta_A_to_Y * A +
                    beta_U_to_Y * conf_strength * U_t + eta_Y + add_dyn)
            pY = cloglog_inv(etaY)
            Y = ((U_Yu[:, t-1] < pY) & (ever_event == 0)).astype(int)

            ever_event = np.maximum(ever_event, Y)

            cum_incidence.append(ever_event.mean())

            L_prev = L
            A_prev = A

        return np.array(cum_incidence)

    risk0_curve = rollout(0)
    risk1_curve = rollout(1)

    true_rd = float(risk1_curve[-1] - risk0_curve[-1])

    return true_rd, risk0_curve, risk1_curve

# Standard errors
def _numerical_gradient(fn, x, eps=1e-5):
    """Central finite differences for gradient of scalar fn w.r.t. vector x."""
    x = np.asarray(x, dtype=float)
    p = len(x)
    grad = np.zeros(p)
    for j in range(p):
        x_up = x.copy(); x_up[j] += eps
        x_dn = x.copy(); x_dn[j] -= eps
        grad[j] = (fn(x_up) - fn(x_dn)) / (2 * eps)
    return grad

def _gmm_jacobian(moment_fn, beta, eps=1e-5):
    """Numerical Jacobian of mean moment conditions w.r.t. beta (q x p matrix)."""
    beta = np.asarray(beta, dtype=float)
    p = len(beta)
    g0 = moment_fn(beta).mean(axis=0)
    q = len(g0)
    D = np.zeros((q, p))
    for j in range(p):
        b_up = beta.copy(); b_up[j] += eps
        b_dn = beta.copy(); b_dn[j] -= eps
        D[:, j] = (moment_fn(b_up).mean(0) - moment_fn(b_dn).mean(0)) / (2 * eps)
    return D


def _cluster_gmm_cov(moment_fn, beta_hat, W_opt, groups):
    """Clustered GMM sandwich variance."""
    D = _gmm_jacobian(moment_fn, beta_hat)
    Gm = moment_fn(beta_hat)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    cluster_sums = np.zeros((len(uniq), Gm.shape[1]), dtype=float)
    np.add.at(cluster_sums, inv, Gm)
    n_obs = Gm.shape[0]
    S = (cluster_sums.T @ cluster_sums) / (n_obs ** 2)
    bread = D.T @ W_opt @ D
    bread_inv = np.linalg.pinv(bread)
    middle = D.T @ W_opt @ S @ W_opt @ D
    V_beta = bread_inv @ middle @ bread_inv
    if len(uniq) > 1:
        V_beta *= len(uniq) / (len(uniq) - 1)
    return V_beta

# Canonical TTE
def run_canonical_tte(df, trunc=TRUNC, cumulative_weights=False, verbose=False):
    """Discrete-time ITT model with IPCW for loss to follow-up."""
    df = df.sort_values(["id", "t"]).copy()

    df["L_s"] = _zscore(df["L"])
    df["t_s"] = _zscore(df["t"])
    df["A_lag"] = df.groupby("id")["A"].shift(1).fillna(0)

    X_ltfu = add_constant(df[["t_s", "L_s", "A_lag"]])
    y_ltfu = 1 - df["C_ltfu"]
    m_ltfu = sm.GLM(y_ltfu, _sanitize_design(X_ltfu), family=LINK_FAM).fit()
    p_notC = m_ltfu.predict(_sanitize_design(X_ltfu))
    num_notC = df.groupby("t")["C_ltfu"].transform(lambda s: 1 - s.mean())
    df["w_period"] = num_notC / p_notC
    if cumulative_weights:
        df["sw"] = df.groupby("id")["w_period"].cumprod()
        df["sw"] = _truncate_weights(df["sw"], trunc[0], trunc[1])
    else:
        df["sw"] = _truncate_weights(df["w_period"], trunc[0], trunc[1])

    if verbose:
        w = df["sw"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"[Canonical TTE Weights] mean={w.mean():.3f} sd={w.std():.3f} "
              f"min={w.min():.3f} max={w.max():.3f}")

    df["C_ltfu_prev"] = df.groupby("id")["C_ltfu"].shift(1).fillna(0)
    df["ltfu_cum"] = df.groupby("id")["C_ltfu_prev"].cumsum()
    df = df[df["ltfu_cum"] == 0].copy()

    if verbose:
        n_ids = df["id"].nunique()
        print(f"[Canonical TTE] retained {len(df)} rows after LTFU censoring; {n_ids} individuals")

    tfe = pd.get_dummies(df["t"].astype(int), prefix="t", drop_first=True, dtype=float)
    X = add_constant(pd.concat([df[["Z"]], tfe], axis=1))
    X = _sanitize_design(X)

    model = sm.GLM(df["Y"], X, family=LINK_FAM, freq_weights=df["sw"]).fit()
    df["pred_h"] = model.predict(X)

    curves = {}
    for z_val in [0, 1]:
        Xpred = X.copy()
        Xpred["Z"] = z_val
        pred_h = model.predict(Xpred)
        tmp = df[["t", "sw"]].copy()
        tmp["pred_h"] = pred_h
        gsum_z = (tmp.assign(hw=tmp["pred_h"] * tmp["sw"])
                    .groupby("t", as_index=False)
                    .agg(hazard_sum=("hw", "sum"), w_sum=("sw", "sum")))
        gsum_z["hazard"] = gsum_z["hazard_sum"] / gsum_z["w_sum"]
        gsum_z.drop(columns=["hazard_sum", "w_sum"], inplace=True)
        gsum_z["cum_event"] = 1 - np.cumprod(1 - gsum_z["hazard"].to_numpy())
        gsum_z["Z"] = z_val
        curves[z_val] = gsum_z

    gsum = pd.concat([curves[0], curves[1]], ignore_index=True)
    t_final = int(gsum["t"].max())
    final_0 = curves[0].loc[curves[0]["t"] == t_final, "cum_event"].iloc[0]
    final_1 = curves[1].loc[curves[1]["t"] == t_final, "cum_event"].iloc[0]
    est_rd = float(final_1 - final_0)
    rr = float(final_1 / final_0) if final_0 > 0 else np.nan

    X_np = X.to_numpy(dtype=float)
    Z_arr = df["Z"].to_numpy(dtype=float)
    t_arr = df["t"].to_numpy(dtype=float)
    sw_arr = df["sw"].to_numpy(dtype=float)
    t_final_val = float(t_final)

    def _rd_from_coefs(coefs):
        eta = X_np @ coefs
        h = 1.0 - np.exp(-np.exp(eta))
        cum = {}
        for z_val in [0, 1]:
            Xp = X_np.copy()
            Xp[:, 1] = z_val
            eta_z = Xp @ coefs
            h_z = 1.0 - np.exp(-np.exp(eta_z))
            surv = 1.0
            for tv in sorted(np.unique(t_arr)):
                sel = t_arr == tv
                hz = np.sum(h_z[sel] * sw_arr[sel]) / np.sum(sw_arr[sel])
                surv *= (1.0 - hz)
            cum[z_val] = 1.0 - surv
        return cum[1] - cum[0]

    try:
        V_theta = model.cov_params()
        grad_rd = _numerical_gradient(_rd_from_coefs, model.params.to_numpy())
        se_rd = float(np.sqrt(grad_rd @ V_theta @ grad_rd))
    except Exception:
        se_rd = np.nan

    if verbose:
        print(f"[Canonical TTE] RD={est_rd:.4f}, SE(RD)={se_rd:.4f}, RR={rr:.4f}")

    return {
        "label": "Canonical TTE",
        "df_analysis": df,
        "gsum": gsum,
        "joint_summary": gsum,
        "est_RD": est_rd,
        "se_rd": se_rd,
        "risk_ratio": rr,
        "model": model,
        "weights": df["sw"].to_numpy(dtype=float)
    }


# Adjusted TTE
def run_adjusted_tte(df, trunc=TRUNC, cumulative_weights=False, verbose=False):
    """Canonical TTE with baseline X added to the outcome model."""
    df = df.sort_values(["id", "t"]).copy()

    df["L_s"] = _zscore(df["L"])
    df["t_s"] = _zscore(df["t"])
    df["X_s"] = _zscore(df["X"])
    df["A_lag"] = df.groupby("id")["A"].shift(1).fillna(0)

    X_ltfu = add_constant(df[["t_s", "L_s", "A_lag"]])
    y_ltfu = 1 - df["C_ltfu"]
    m_ltfu = sm.GLM(y_ltfu, _sanitize_design(X_ltfu), family=LINK_FAM).fit()
    p_notC = m_ltfu.predict(_sanitize_design(X_ltfu))
    num_notC = df.groupby("t")["C_ltfu"].transform(lambda s: 1 - s.mean())
    df["w_period"] = num_notC / p_notC
    if cumulative_weights:
        df["sw"] = df.groupby("id")["w_period"].cumprod()
        df["sw"] = _truncate_weights(df["sw"], trunc[0], trunc[1])
    else:
        df["sw"] = _truncate_weights(df["w_period"], trunc[0], trunc[1])

    if verbose:
        w = df["sw"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"[Adjusted TTE Weights] mean={w.mean():.3f} sd={w.std():.3f} "
              f"min={w.min():.3f} max={w.max():.3f}")

    df["C_ltfu_prev"] = df.groupby("id")["C_ltfu"].shift(1).fillna(0)
    df["ltfu_cum"] = df.groupby("id")["C_ltfu_prev"].cumsum()
    df = df[df["ltfu_cum"] == 0].copy()

    if verbose:
        n_ids = df["id"].nunique()
        print(f"[Adjusted TTE] retained {len(df)} rows after LTFU censoring; {n_ids} individuals")

    tfe = pd.get_dummies(df["t"].astype(int), prefix="t", drop_first=True, dtype=float)
    X_design = add_constant(pd.concat([df[["Z", "X_s"]], tfe], axis=1))
    X_design = _sanitize_design(X_design)

    model = sm.GLM(df["Y"], X_design, family=LINK_FAM, freq_weights=df["sw"]).fit()
    curves = {}
    for z_val in [0, 1]:
        Xpred = X_design.copy()
        Xpred["Z"] = z_val
        pred_h = model.predict(Xpred)
        tmp = df[["t", "sw"]].copy()
        tmp["pred_h"] = pred_h
        gsum_z = (tmp.assign(hw=tmp["pred_h"] * tmp["sw"])
                    .groupby("t", as_index=False)
                    .agg(hazard_sum=("hw", "sum"), w_sum=("sw", "sum")))
        gsum_z["hazard"] = gsum_z["hazard_sum"] / gsum_z["w_sum"]
        gsum_z.drop(columns=["hazard_sum", "w_sum"], inplace=True)
        gsum_z["cum_event"] = 1 - np.cumprod(1 - gsum_z["hazard"].to_numpy())
        gsum_z["Z"] = z_val
        curves[z_val] = gsum_z

    gsum = pd.concat([curves[0], curves[1]], ignore_index=True)
    t_final = int(gsum["t"].max())
    final_0 = curves[0].loc[curves[0]["t"] == t_final, "cum_event"].iloc[0]
    final_1 = curves[1].loc[curves[1]["t"] == t_final, "cum_event"].iloc[0]
    est_rd = float(final_1 - final_0)
    rr = float(final_1 / final_0) if final_0 > 0 else np.nan

    X_np = X_design.to_numpy(dtype=float)
    Z_arr = df["Z"].to_numpy(dtype=float)
    t_arr = df["t"].to_numpy(dtype=float)
    sw_arr = df["sw"].to_numpy(dtype=float)

    def _rd_from_coefs(coefs):
        cum = {}
        for z_val in [0, 1]:
            Xp = X_np.copy()
            Xp[:, 1] = z_val
            eta = Xp @ coefs
            h = 1.0 - np.exp(-np.exp(eta))
            surv = 1.0
            for tv in sorted(np.unique(t_arr)):
                sel = t_arr == tv
                hz = np.sum(h[sel] * sw_arr[sel]) / np.sum(sw_arr[sel])
                surv *= (1.0 - hz)
            cum[z_val] = 1.0 - surv
        return cum[1] - cum[0]

    try:
        V_theta = model.cov_params()
        grad_rd = _numerical_gradient(_rd_from_coefs, model.params.to_numpy())
        se_rd = float(np.sqrt(grad_rd @ V_theta @ grad_rd))
    except Exception:
        se_rd = np.nan

    if verbose:
        print(f"[Adjusted TTE] RD={est_rd:.4f}, SE(RD)={se_rd:.4f}, RR={rr:.4f}")

    return {
        "label": "Adjusted TTE",
        "df_analysis": df,
        "gsum": gsum,
        "joint_summary": gsum,
        "est_RD": est_rd,
        "se_rd": se_rd,
        "risk_ratio": rr,
        "model": model,
        "weights": df["sw"].to_numpy(dtype=float)
    }


# Spline basis
def piecewise_basis(d, knots):
    """General piecewise-linear basis: m+1 segments for m knots."""
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


# Restricted cubic splines
def rcs_basis(d, knots):
    """Restricted cubic spline basis."""
    d = np.asarray(d, dtype=float)
    knots = np.sort(np.asarray(knots, dtype=float))
    k = len(knots)
    if k < 3:
        raise ValueError("RCS requires at least 3 knots")

    def _truncpow3(x):
        return np.where(x > 0, x ** 3, 0.0)

    t_last = knots[-1]
    t_pen = knots[-2]
    denom = t_last - t_pen

    cols = [d]
    for j in range(k - 2):
        tj = knots[j]
        col = (_truncpow3(d - tj)
               - _truncpow3(d - t_pen) * (t_last - tj) / denom
               + _truncpow3(d - t_last) * (t_pen - tj) / denom)
        col /= (t_last - knots[0]) ** 2
        cols.append(col)

    return np.column_stack(cols)


def make_basis(d, knots, spline_type=None):
    """Dispatch to piecewise_basis or rcs_basis based on spline_type."""
    if spline_type is None:
        spline_type = SPLINE_TYPE
    if spline_type == "rcs":
        return rcs_basis(d, knots)
    else:
        return piecewise_basis(d, knots)


# Treatment-history timers
def add_spline_timers(df, grace_len, on_q=None, wash_q=None, never_q=None,
                      fixed_knots_on=None, fixed_knots_wash=None, fixed_knots_never=None):
    """Add treatment-history timers and spline basis columns."""
    if on_q is None:
        on_q = ON_KNOT_Q
    if wash_q is None:
        wash_q = WASH_KNOT_Q
    if never_q is None:
        never_q = NEVER_KNOT_Q

    df = df.sort_values(["id", "Z_clone", "t"]).copy()

    df["ever_on"] = 0
    df["t_on"]    = 0
    df["t_off"]   = 0
    df["t_never"] = 0
    df["grace_on_dur"] = 0
    df["grace_off_dur"] = 0
    df["d_on_prev"]  = 0
    df["d_off_prev"] = 0

    for (i, zc), g in df.groupby(["id", "Z_clone"], sort=False):
        ever = 0
        ton  = 0
        toff = 0
        tnev = 0
        grace_on = 0
        grace_off = 0
        d_on_prev  = 0
        d_off_prev = 0
        prev_A = 0
        idx  = g.index.to_list()

        for k, (_, row) in enumerate(g.iterrows()):
            t_val = int(row["t"])
            A_val = int(row["A"])

            if A_val == 1:
                if prev_A == 0 and ever == 1 and toff > 0:
                    d_off_prev = toff
                ton  += 1
                toff  = 0
                tnev  = 0
                ever  = 1
            else:
                if prev_A == 1 and ton > 0:
                    d_on_prev = ton
                ton = 0
                if ever == 1:
                    toff += 1
                    tnev = 0
                else:
                    toff = 0
                    tnev += 1

            if t_val <= grace_len:
                if A_val == 1:
                    grace_on = ton
                else:
                    grace_off = toff

            prev_A = A_val

            df.loc[idx[k], ["ever_on", "t_on", "t_off", "t_never",
                            "grace_on_dur", "grace_off_dur",
                            "d_on_prev", "d_off_prev"]] = (
                ever, ton, toff, tnev, grace_on, grace_off,
                d_on_prev, d_off_prev
            )

    t_on_pos    = df.loc[df["t_on"]    > 0, "t_on"].to_numpy(dtype=float)
    t_wash_pos  = df.loc[df["t_off"]   > 0, "t_off"].to_numpy(dtype=float)
    t_never_pos = df.loc[df["t_never"] > 0, "t_never"].to_numpy(dtype=float)

    def choose_knots(d_pos, q_list):
        fallback = np.array([2.0, 4.0, 6.0], dtype=float)
        if q_list is None or len(q_list) == 0:
            q_list = [0.2, 0.4, 0.6, 0.8]
        if d_pos.size == 0 or np.max(d_pos) <= 1:
            return fallback
        q_arr = np.clip(np.asarray(q_list, dtype=float), 0.0, 1.0)
        knots = np.quantile(d_pos, q_arr)
        knots = np.unique(knots)
        if knots.size == 0:
            return fallback
        if SPLINE_TYPE == "rcs" and knots.size < 3:
            lo, hi = knots[0], knots[-1]
            if knots.size == 1:
                knots = np.array([max(lo * 0.5, 1.0), lo, lo * 1.5], dtype=float)
            elif knots.size == 2:
                knots = np.array([lo, (lo + hi) / 2, hi], dtype=float)
            knots = np.unique(knots)
        return knots.astype(float)

    knots_on    = np.asarray(fixed_knots_on, dtype=float)    if fixed_knots_on    is not None else choose_knots(t_on_pos, on_q)
    knots_wash  = np.asarray(fixed_knots_wash, dtype=float)  if fixed_knots_wash  is not None else choose_knots(t_wash_pos, wash_q)
    knots_never = np.asarray(fixed_knots_never, dtype=float) if fixed_knots_never is not None else choose_knots(t_never_pos, never_q)

    for c in list(df.columns):
        if c.startswith("A_on_") or c.startswith("wash_") or c.startswith("never_") or c.startswith("off_"):
            df.drop(columns=c, inplace=True)

    if knots_on.size > 0:
        B_on = make_basis(df["t_on"].to_numpy(dtype=float), knots_on)
        for j in range(B_on.shape[1]):
            df[f"A_on_sp{j+1}"] = B_on[:, j]

    if knots_wash.size > 0:
        B_wash = make_basis(df["t_off"].to_numpy(dtype=float), knots_wash)
        for j in range(B_wash.shape[1]):
            df[f"wash_sp{j+1}"] = B_wash[:, j]

    if knots_never.size > 0:
        B_never = piecewise_basis(df["t_never"].to_numpy(dtype=float), knots_never)
        for j in range(B_never.shape[1]):
            df[f"never_sp{j+1}"] = B_never[:, j]

    return df, knots_on, knots_wash, knots_never

def run_snmm_g(df, grace_len=3, trunc=TRUNC,
               on_q=ON_KNOT_Q, wash_q=WASH_KNOT_Q, never_q=NEVER_KNOT_Q,
               fixed_knots_on=None, fixed_knots_wash=None, fixed_knots_never=None,
               beta0_init=None, cumulative_weights=False,
               use_adherence_weights=False, restart_interactions=False,
               fixed_base_beta=None,
               verbose=False):

    clones = pd.concat([df.assign(Z_clone=0), df.assign(Z_clone=1)]) \
             .sort_values(["id","Z_clone","t"]).reset_index(drop=True)

    clones["adherent"]    = ((clones["t"] <= grace_len) & (clones["A"] == clones["Z_clone"])).astype(int)
    clones["C_adherence"] = ((clones["t"] <= grace_len) & (clones["A"] != clones["Z_clone"])).astype(int)

    work = clones.copy()
    work["L_s"] = _zscore(work["L"])
    work["X_s"] = _zscore(work["X"])
    work["t_s"] = _zscore(work["t"])
    work["A_lag"] = work.groupby(["id","Z_clone"])["A"].shift(1).fillna(0)
    mask_grace = work["t"] <= grace_len

    X_ltfu = add_constant(work[["t_s","L_s","A_lag"]])
    y_ltfu = 1 - work["C_ltfu"]
    m_ltfu = sm.GLM(y_ltfu, _sanitize_design(X_ltfu), family=LINK_FAM).fit()
    p_notC = m_ltfu.predict(_sanitize_design(X_ltfu))
    num_notC = work.groupby("t")["C_ltfu"].transform(lambda s: 1 - s.mean())
    work["w_ltfu_period"] = num_notC / p_notC
    if cumulative_weights:
        work["w_ltfu"] = work.groupby(["id","Z_clone"])["w_ltfu_period"].cumprod()
    else:
        work["w_ltfu"] = work["w_ltfu_period"]

    work["w_adherence"] = 1.0
    if use_adherence_weights and mask_grace.any():
        df_ad = work.loc[mask_grace].copy()
        X_ad  = add_constant(df_ad[["t_s","L_s","Z","Z_clone"]])
        y_ad  = 1 - df_ad["C_adherence"]
        m_ad  = sm.GLM(y_ad, _sanitize_design(X_ad), family=LINK_FAM).fit()
        p_notCad = m_ad.predict(_sanitize_design(X_ad))
        num_notCad = df_ad.groupby(["t","Z","Z_clone"])["C_adherence"].transform(lambda s: 1 - s.mean())
        work.loc[mask_grace, "w_adherence"] = (num_notCad.values / p_notCad)

    work["sw"] = _truncate_weights(work["w_ltfu"] * work["w_adherence"],
                                  trunc[0], trunc[1])

    work = work.sort_values(["id","Z_clone","t"]).copy()
    work["adh_cum"] = work.groupby(["id","Z_clone"])["C_adherence"].cumsum()

    work["C_ltfu_prev"] = work.groupby(["id","Z_clone"])["C_ltfu"].shift(1).fillna(0)
    work["ltfu_cum_prev"] = work.groupby(["id","Z_clone"])["C_ltfu_prev"].cumsum()

    work = work[(work["adh_cum"] == 0) & (work["ltfu_cum_prev"] == 0)].copy()

    if verbose:
        n_paths = work[["id","Z_clone"]].drop_duplicates().shape[0]
        print(f"[SNMM] retained {len(work)} rows after censoring; {n_paths} id x clone paths")

    work, knots_on, knots_wash, knots_never = add_spline_timers(
        work, grace_len, on_q=on_q, wash_q=wash_q, never_q=never_q,
        fixed_knots_on=fixed_knots_on, fixed_knots_wash=fixed_knots_wash,
        fixed_knots_never=fixed_knots_never
    )

    analysis = work.copy()

    analysis["A_pre"] = analysis["A"] * (analysis["t"] <= grace_len).astype(int)

    X_base = analysis[["L", "A_pre"]].copy()
    tfe = pd.get_dummies(analysis["t"].astype(int), prefix="t",
                        drop_first=True, dtype=float)

    Xnuis = add_constant(pd.concat([X_base, tfe], axis=1))
    Xnuis = _sanitize_design(Xnuis)

    y = analysis["Y"].to_numpy(dtype=float)
    nuis = sm.GLM(y, Xnuis, family=LINK_FAM,
                freq_weights=analysis["sw"]).fit()

    lin0 = Xnuis.to_numpy(dtype=float) @ nuis.params.to_numpy(dtype=float)

    analysis["Z_resid"] = analysis["Z_clone"] - 0.5

    if verbose:
        print(f"[SNMM] Instrument: Z_resid = Z_clone - 0.5")
        print(f"  Mean Z_resid: {analysis['Z_resid'].mean():.4f}, SD: {analysis['Z_resid'].std():.4f}")

    w  = analysis["sw"].to_numpy(dtype=float)
    Z_resid = analysis["Z_resid"].to_numpy(dtype=float)
    A  = analysis["A"].to_numpy(dtype=float)
    EO = analysis["ever_on"].to_numpy(dtype=float)
    N  = len(analysis)

    on_cols    = [c for c in analysis.columns if c.startswith("A_on_")]
    wash_cols  = [c for c in analysis.columns if c.startswith("wash_")]

    S_on_p    = analysis[on_cols].to_numpy(dtype=float)    if on_cols    else np.zeros((N, 0))
    S_wash_p  = analysis[wash_cols].to_numpy(dtype=float)  if wash_cols  else np.zeros((N, 0))

    p_on   = S_on_p.shape[1]
    p_wash = S_wash_p.shape[1]

    p_restart_on = 0
    p_restart_wash = 0
    restart_on_basis = np.zeros((N, 0))
    restart_wash_basis = np.zeros((N, 0))
    if restart_interactions:
        restart_on_basis = analysis["d_off_prev"].to_numpy(dtype=float).reshape(-1, 1)
        restart_wash_basis = analysis["d_on_prev"].to_numpy(dtype=float).reshape(-1, 1)
        p_restart_on = 1
        p_restart_wash = 1
        if verbose:
            n_restarters = int((analysis["d_off_prev"] > 0).sum())
            n_on = int((A > 0).sum())
            n_wash_prev = int(((1 - A) * EO * analysis["d_on_prev"].to_numpy(dtype=float) > 0).sum())
            n_wash = int(((1 - A) * EO > 0).sum())
            print(f"[SNMM] Restart interactions: {n_restarters}/{n_on} ON rows from restarters, "
                  f"{n_wash_prev}/{n_wash} WASH rows with prior spell info")

    if fixed_base_beta is not None:
        restart_interactions = True
        if p_restart_on == 0:
            restart_on_basis = analysis["d_off_prev"].to_numpy(dtype=float).reshape(-1, 1)
            restart_wash_basis = analysis["d_on_prev"].to_numpy(dtype=float).reshape(-1, 1)
            p_restart_on = 1
            p_restart_wash = 1

    p = p_on + p_restart_on + p_wash + p_restart_wash

    post_grace = (analysis["t"] > grace_len).to_numpy()

    wZ  = (w * Z_resid)[:, None]
    L_arr = analysis["L"].to_numpy(dtype=float)
    L_cent = L_arr - L_arr[post_grace].mean()
    wZL = (w * Z_resid * L_cent)[:, None]

    q = 2 * p

    idx_on_end = p_on
    idx_ron_end = idx_on_end + p_restart_on
    idx_wash_end = idx_ron_end + p_wash

    def _compute_blip(beta):
        beta = np.asarray(beta, dtype=float)
        blip = np.zeros(N)
        if p_on:
            blip += A * (S_on_p @ beta[:idx_on_end])
        if p_restart_on:
            blip += A * (restart_on_basis @ beta[idx_on_end:idx_ron_end])
        if p_wash:
            blip += (1-A) * EO * (S_wash_p @ beta[idx_ron_end:idx_wash_end])
        if p_restart_wash:
            blip += (1-A) * EO * (restart_wash_basis @ beta[idx_wash_end:])
        return blip

    def moment_matrix(beta):
        blip = _compute_blip(beta)
        mu = 1 - np.exp(-np.exp(lin0 + blip))
        resid = (y - mu).reshape(-1, 1)

        g_on_z    = wZ  * (A[:, None]          * S_on_p   * resid) if p_on   else np.zeros((N, 0))
        g_wash_z  = wZ  * (((1-A)*EO)[:, None] * S_wash_p * resid) if p_wash else np.zeros((N, 0))
        g_on_zl   = wZL * (A[:, None]          * S_on_p   * resid) if p_on   else np.zeros((N, 0))
        g_wash_zl = wZL * (((1-A)*EO)[:, None] * S_wash_p * resid) if p_wash else np.zeros((N, 0))

        g_ron_z    = wZ  * (A[:, None]          * restart_on_basis   * resid) if p_restart_on   else np.zeros((N, 0))
        g_rwash_z  = wZ  * (((1-A)*EO)[:, None] * restart_wash_basis * resid) if p_restart_wash else np.zeros((N, 0))
        g_ron_zl   = wZL * (A[:, None]          * restart_on_basis   * resid) if p_restart_on   else np.zeros((N, 0))
        g_rwash_zl = wZL * (((1-A)*EO)[:, None] * restart_wash_basis * resid) if p_restart_wash else np.zeros((N, 0))

        return np.concatenate([g_on_z, g_ron_z, g_wash_z, g_rwash_z,
                               g_on_zl, g_ron_zl, g_wash_zl, g_rwash_zl], axis=1)[post_grace]

    def gmm_objective(beta, W):
        g_bar = moment_matrix(beta).mean(axis=0)
        return float(g_bar @ W @ g_bar)

    if fixed_base_beta is not None:
        _fbb = np.asarray(fixed_base_beta, dtype=float)
        assert len(_fbb) == p_on + p_wash, \
            f"fixed_base_beta length {len(_fbb)} != p_on+p_wash={p_on+p_wash}"
        _fbb_on = _fbb[:p_on]
        _fbb_wash = _fbb[p_on:]

        p_gamma = p_restart_on + p_restart_wash
        q_gamma = 2 * p_gamma

        def _gamma_to_full(gamma):
            """Expand [gamma_on, gamma_wash] to full parameter vector with fixed base beta."""
            gamma = np.asarray(gamma, dtype=float)
            full = np.zeros(p)
            full[:idx_on_end] = _fbb_on
            full[idx_on_end:idx_ron_end] = gamma[:p_restart_on]
            full[idx_ron_end:idx_wash_end] = _fbb_wash
            full[idx_wash_end:] = gamma[p_restart_on:]
            return full

        def _restart_moment_matrix(gamma):
            """Moment conditions for restart terms only, with base beta fixed."""
            full = _gamma_to_full(gamma)
            blip = _compute_blip(full)
            mu = 1 - np.exp(-np.exp(lin0 + blip))
            resid = (y - mu).reshape(-1, 1)
            g_ron_z    = wZ  * (A[:, None]          * restart_on_basis   * resid)
            g_rwash_z  = wZ  * (((1-A)*EO)[:, None] * restart_wash_basis * resid)
            g_ron_zl   = wZL * (A[:, None]          * restart_on_basis   * resid)
            g_rwash_zl = wZL * (((1-A)*EO)[:, None] * restart_wash_basis * resid)
            return np.concatenate([g_ron_z, g_rwash_z,
                                   g_ron_zl, g_rwash_zl], axis=1)[post_grace]

        def _gamma_objective(gamma, W):
            g_bar = _restart_moment_matrix(gamma).mean(axis=0)
            return float(g_bar @ W @ g_bar)

        gamma0 = np.zeros(p_gamma)
        W1g = np.eye(q_gamma)
        sol1g = minimize(_gamma_objective, gamma0, args=(W1g,), method='L-BFGS-B',
                         options=dict(maxiter=500, ftol=1e-12))
        G1g = _restart_moment_matrix(sol1g.x)
        Omega_g = (G1g.T @ G1g) / G1g.shape[0]
        try:
            W_opt_g = np.linalg.inv(Omega_g)
        except np.linalg.LinAlgError:
            W_opt_g = np.linalg.pinv(Omega_g)
        sol2g = minimize(_gamma_objective, sol1g.x, args=(W_opt_g,), method='L-BFGS-B',
                         options=dict(maxiter=500, ftol=1e-12))
        gamma_hat = sol2g.x
        beta_hat = _gamma_to_full(gamma_hat)

        try:
            D_g = _gmm_jacobian(_restart_moment_matrix, gamma_hat)
            n_obs_g = G1g.shape[0]
            bread_g = D_g.T @ W_opt_g @ D_g
            V_gamma = np.linalg.inv(bread_g) / n_obs_g
            se_gamma = np.sqrt(np.diag(V_gamma))
        except Exception:
            V_gamma = None
            se_gamma = np.full(p_gamma, np.nan)

        G_final_g = _restart_moment_matrix(gamma_hat)
        g_bar_g = G_final_g.mean(axis=0)
        n_obs_g = G_final_g.shape[0]
        J_stat = float(n_obs_g * (g_bar_g @ W_opt_g @ g_bar_g))
        J_df = q_gamma - p_gamma
        J_pval = float(1 - chi2.cdf(J_stat, J_df))

        beta_hat_base_full = _gamma_to_full(np.zeros(p_gamma))
        W_opt = W_opt_g

        if verbose:
            print(f"[SNMM restart test] gamma_on={gamma_hat[:p_restart_on]}, "
                  f"SE={se_gamma[:p_restart_on]}")
            print(f"[SNMM restart test] gamma_wash={gamma_hat[p_restart_on:]}, "
                  f"SE={se_gamma[p_restart_on:]}")
            print(f"[SNMM restart test] J={J_stat:.3f}, df={J_df}, p={J_pval:.4f}")

        blip_base = _compute_blip(beta_hat_base_full)
        analysis["pred_h"] = 1 - np.exp(-np.exp(lin0 + blip_base))
        gsum = (analysis.assign(hw=analysis["pred_h"] * analysis["sw"])
                     .groupby(["Z_clone","t"], as_index=False)
                     .agg(hazard_sum=("hw","sum"), w_sum=("sw","sum")))
        gsum["hazard"] = gsum["hazard_sum"] / gsum["w_sum"]
        gsum["cum_event"] = gsum.groupby("Z_clone")["hazard"].transform(
            lambda h: 1 - np.cumprod(1 - h))
        final = gsum[gsum["t"] == gsum["t"].max()].set_index("Z_clone")["cum_event"]
        est_rd = float(final[1] - final[0])

        result = {
            "label": "SNMM restart test",
            "est_RD": est_rd,
            "se_rd": np.nan,
            "gamma_on": gamma_hat[:p_restart_on],
            "gamma_wash": gamma_hat[p_restart_on:],
            "se_gamma_on": se_gamma[:p_restart_on],
            "se_gamma_wash": se_gamma[p_restart_on:],
            "J_stat": J_stat, "J_df": J_df, "J_pval": J_pval,
        }
        return result

    if beta0_init is not None and len(beta0_init) == p:
        beta0 = np.asarray(beta0_init, dtype=float)
    else:
        beta0 = np.zeros(p)

    if p == 0:
        beta_hat = beta0
    else:
        W1 = np.eye(q)
        sol1 = minimize(gmm_objective, beta0, args=(W1,), method='L-BFGS-B',
                        options=dict(maxiter=500, ftol=1e-12))

        G1 = moment_matrix(sol1.x)
        Omega = (G1.T @ G1) / G1.shape[0]
        try:
            W_opt = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            W_opt = np.linalg.pinv(Omega)

        sol2 = minimize(gmm_objective, sol1.x, args=(W_opt,), method='L-BFGS-B',
                        options=dict(maxiter=500, ftol=1e-12))
        beta_hat = sol2.x

        if np.any(np.abs(beta_hat) > 10):
            msg = f"SNMM solver produced extreme coefficients (max |beta|={np.max(np.abs(beta_hat)):.2g})"
            if verbose:
                print(f"  [SNMM] {msg}")
            raise RuntimeError(msg)

    J_stat, J_df, J_pval = np.nan, 0, np.nan
    if p > 0 and q > p:
        G_final = moment_matrix(beta_hat)
        g_bar = G_final.mean(axis=0)
        n_obs = G_final.shape[0]
        J_stat = float(n_obs * (g_bar @ W_opt @ g_bar))
        J_df = q - p
        J_pval = float(1 - chi2.cdf(J_stat, J_df))
        if verbose:
            print(f"[SNMM] Hansen J-test: J={J_stat:.3f}, df={J_df}, p={J_pval:.4f}")

    V_beta = None
    se_beta = np.full(p, np.nan) if p > 0 else np.array([])
    if p > 0:
        try:
            groups_pg = analysis["id"].to_numpy()[post_grace]
            V_beta = _cluster_gmm_cov(moment_matrix, beta_hat, W_opt, groups_pg)
            se_beta = np.sqrt(np.diag(V_beta))
        except Exception:
            V_beta = None
            se_beta = np.full(p, np.nan)

    blip_all = _compute_blip(beta_hat)

    analysis["pred_h"] = 1 - np.exp(-np.exp(lin0 + blip_all))

    gsum = (analysis.assign(hw=analysis["pred_h"] * analysis["sw"])
                 .groupby(["Z_clone","t"], as_index=False)
                 .agg(hazard_sum=("hw","sum"), w_sum=("sw","sum")))
    gsum["hazard"] = gsum["hazard_sum"] / gsum["w_sum"]
    gsum["cum_event"] = gsum.groupby("Z_clone")["hazard"].transform(
        lambda h: 1 - np.cumprod(1 - h)
    )

    final = gsum[gsum["t"] == gsum["t"].max()] \
                .set_index("Z_clone")["cum_event"]

    est_rd = float(final[1] - final[0])
    rr     = float(final[1] / final[0])

    Zc_arr = analysis["Z_clone"].to_numpy(dtype=float)
    t_arr_snmm = analysis["t"].to_numpy(dtype=float)
    sw_arr_snmm = analysis["sw"].to_numpy(dtype=float)

    def _rd_from_beta_snmm(beta_v):
        blip_v = _compute_blip(beta_v)
        h_v = 1.0 - np.exp(-np.exp(lin0 + blip_v))
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

    se_rd = np.nan
    if p > 0 and V_beta is not None:
        try:
            grad_rd = _numerical_gradient(_rd_from_beta_snmm, beta_hat)
            se_rd = float(np.sqrt(grad_rd @ V_beta @ grad_rd))
        except Exception:
            se_rd = np.nan

    def basis_means(df_, dur_col, cols, a_filter, max_dur=T):
        out = {}
        df2 = df_.loc[a_filter, [dur_col] + cols].copy()
        if df2.empty:
            return {d: pd.Series(0.0, index=cols) for d in range(max_dur+1)}

        means = df2.groupby(dur_col)[cols].mean()
        idx = means.index.to_series()

        for d in range(0, max_dur + 1):
            if d == 0:
                out[d] = pd.Series(0.0, index=cols)
            elif d in means.index:
                out[d] = means.loc[d]
            else:
                nearest = int(idx.iloc[(idx - d).abs().argmin()])
                out[d] = means.loc[nearest]
        return out

    design_all = analysis.copy()

    d_grid = np.arange(0, T + 1, dtype=float)

    modeled_on = None
    if p_on:
        B_on_grid = make_basis(d_grid, knots_on)
        modeled_on = B_on_grid @ beta_hat[:idx_on_end]

    modeled_wash = None
    if p_wash:
        B_wash_grid = make_basis(d_grid, knots_wash)
        modeled_wash = B_wash_grid @ beta_hat[idx_ron_end:idx_wash_end]

    modeled_never = np.zeros(T + 1)

    if verbose:
        print("\n[SNMM beta estimates]")
        print(f"  ON spline coeffs:    {beta_hat[:idx_on_end]}")
        if p_restart_on:
            print(f"  ON restart gamma:    {beta_hat[idx_on_end:idx_ron_end]} (SE: {se_beta[idx_on_end:idx_ron_end]})")
        print(f"  WASH spline coeffs:  {beta_hat[idx_ron_end:idx_wash_end]}")
        if p_restart_wash:
            print(f"  WASH restart gamma:  {beta_hat[idx_wash_end:]} (SE: {se_beta[idx_wash_end:]})")
        print(f"  NEVER: fixed at 0 (not estimated)")
        se_str = f"{se_rd:.4f}" if not np.isnan(se_rd) else "N/A"
        print(f"[SNMM ITT] RD={est_rd:.4f} SE(RD)={se_str} RR={rr:.4f}")

    result = {
        "label": "SNMM ITT",
        "est_RD": est_rd,
        "se_rd": se_rd,
        "risk_ratio": rr,
        "joint_summary": gsum,
        "meta": dict(beta=beta_hat, se_beta=se_beta, V_beta=V_beta,
                     on_cols=on_cols, wash_cols=wash_cols,
                     modeled_on=modeled_on, modeled_wash=modeled_wash,
                     modeled_never=modeled_never,
                     knots_on=knots_on, knots_wash=knots_wash,
                     knots_never=knots_never),
        "nuisance": nuis,
        "weights": analysis["sw"].to_numpy(dtype=float),
        "J_stat": J_stat,
        "J_df": J_df,
        "J_pval": J_pval
    }

    if restart_interactions:
        result["gamma_on"] = beta_hat[idx_on_end:idx_ron_end]
        result["gamma_wash"] = beta_hat[idx_wash_end:]
        result["se_gamma_on"] = se_beta[idx_on_end:idx_ron_end]
        result["se_gamma_wash"] = se_beta[idx_wash_end:]

    return result


def _fast_snmm_bootstrap_rd(df, snmm_res, grace_len, trunc, b_boot, seed):
    """Fast SNMM bootstrap for RD."""
    beta_main = snmm_res["meta"]["beta"]
    fk_on = snmm_res["meta"]["knots_on"]
    fk_wash = snmm_res["meta"]["knots_wash"]
    fk_never = snmm_res["meta"]["knots_never"]

    ids = df["id"].unique()
    n_ids = len(ids)
    rng = np.random.default_rng(seed)

    id_groups = df.groupby("id").indices
    id_list = list(id_groups.keys())
    id_to_idx = {iid: id_groups[iid] for iid in id_list}

    boot_rds = []
    for bb in range(b_boot):
        try:
            boot_ids = rng.choice(ids, size=n_ids, replace=True)
            idx_blocks = [id_to_idx[oid] for oid in boot_ids]
            new_id_arr = np.concatenate([
                np.full(len(idx_blocks[i]), i, dtype=np.int64)
                for i in range(n_ids)
            ])
            row_idx = np.concatenate(idx_blocks)
            dfb = df.iloc[row_idx].copy()
            dfb["id"] = new_id_arr

            clones = pd.concat([dfb.assign(Z_clone=0), dfb.assign(Z_clone=1)]) \
                     .sort_values(["id","Z_clone","t"]).reset_index(drop=True)
            clones["C_adherence"] = ((clones["t"] <= grace_len) & (clones["A"] != clones["Z_clone"])).astype(int)

            work = clones.copy()
            work["L_s"] = _zscore(work["L"])
            work["X_s"] = _zscore(work["X"])
            work["t_s"] = _zscore(work["t"])
            work["A_lag"] = work.groupby(["id","Z_clone"])["A"].shift(1).fillna(0)

            X_ltfu = add_constant(work[["t_s","L_s","A_lag"]])
            y_ltfu = 1 - work["C_ltfu"]
            m_ltfu = sm.GLM(y_ltfu, _sanitize_design(X_ltfu), family=LINK_FAM).fit()
            p_notC = m_ltfu.predict(_sanitize_design(X_ltfu))
            num_notC = work.groupby("t")["C_ltfu"].transform(lambda s: 1 - s.mean())
            work["w_ltfu"] = num_notC / p_notC
            work["sw"] = _truncate_weights(work["w_ltfu"], trunc[0], trunc[1])

            work = work.sort_values(["id","Z_clone","t"]).copy()
            work["adh_cum"] = work.groupby(["id","Z_clone"])["C_adherence"].cumsum()
            work["C_ltfu_prev"] = work.groupby(["id","Z_clone"])["C_ltfu"].shift(1).fillna(0)
            work["ltfu_cum_prev"] = work.groupby(["id","Z_clone"])["C_ltfu_prev"].cumsum()
            work = work[(work["adh_cum"] == 0) & (work["ltfu_cum_prev"] == 0)].copy()

            work, _, _, _ = add_spline_timers(
                work, grace_len,
                fixed_knots_on=fk_on, fixed_knots_wash=fk_wash,
                fixed_knots_never=fk_never)

            analysis = work.copy()
            analysis["A_pre"] = analysis["A"] * (analysis["t"] <= grace_len).astype(int)
            X_base = analysis[["L", "A_pre"]].copy()
            tfe = pd.get_dummies(analysis["t"].astype(int), prefix="t",
                                drop_first=True, dtype=float)
            Xnuis = add_constant(pd.concat([X_base, tfe], axis=1))
            Xnuis = _sanitize_design(Xnuis)
            y = analysis["Y"].to_numpy(dtype=float)
            nuis_b = sm.GLM(y, Xnuis, family=LINK_FAM,
                           freq_weights=analysis["sw"]).fit()
            lin0 = Xnuis.to_numpy(dtype=float) @ nuis_b.params.to_numpy(dtype=float)

            analysis["Z_resid"] = analysis["Z_clone"] - 0.5
            w  = analysis["sw"].to_numpy(dtype=float)
            Z_resid = analysis["Z_resid"].to_numpy(dtype=float)
            A_arr  = analysis["A"].to_numpy(dtype=float)
            EO = analysis["ever_on"].to_numpy(dtype=float)
            Nb = len(analysis)

            on_cols    = [c for c in analysis.columns if c.startswith("A_on_")]
            wash_cols  = [c for c in analysis.columns if c.startswith("wash_")]
            S_on_p    = analysis[on_cols].to_numpy(dtype=float) if on_cols else np.zeros((Nb, 0))
            S_wash_p  = analysis[wash_cols].to_numpy(dtype=float) if wash_cols else np.zeros((Nb, 0))
            p_on = S_on_p.shape[1]
            p_wash = S_wash_p.shape[1]
            p_tot = p_on + p_wash

            post_grace = (analysis["t"] > grace_len).to_numpy()
            wZ  = (w * Z_resid)[:, None]
            L_arr = analysis["L"].to_numpy(dtype=float)
            L_cent = L_arr - L_arr[post_grace].mean()
            wZL = (w * Z_resid * L_cent)[:, None]
            q = 2 * p_tot

            def _blip_b(beta):
                blip = np.zeros(Nb)
                if p_on:
                    blip += A_arr * (S_on_p @ beta[:p_on])
                if p_wash:
                    blip += (1-A_arr) * EO * (S_wash_p @ beta[p_on:])
                return blip

            def _moments_b(beta):
                blip = _blip_b(beta)
                mu = 1 - np.exp(-np.exp(lin0 + blip))
                resid = (y - mu).reshape(-1, 1)
                g_on_z   = wZ  * (A_arr[:, None] * S_on_p * resid) if p_on else np.zeros((Nb, 0))
                g_wash_z = wZ  * (((1-A_arr)*EO)[:, None] * S_wash_p * resid) if p_wash else np.zeros((Nb, 0))
                g_on_zl  = wZL * (A_arr[:, None] * S_on_p * resid) if p_on else np.zeros((Nb, 0))
                g_wash_zl= wZL * (((1-A_arr)*EO)[:, None] * S_wash_p * resid) if p_wash else np.zeros((Nb, 0))
                return np.concatenate([g_on_z, g_wash_z, g_on_zl, g_wash_zl], axis=1)[post_grace]

            def _obj_b(beta, W):
                g_bar = _moments_b(beta).mean(axis=0)
                return float(g_bar @ W @ g_bar)

            beta0 = beta_main[:p_tot] if len(beta_main) >= p_tot else np.zeros(p_tot)
            W1 = np.eye(q)
            sol1 = minimize(_obj_b, beta0, args=(W1,), method='L-BFGS-B',
                           options=dict(maxiter=300, ftol=1e-10))
            G1 = _moments_b(sol1.x)
            Omega = (G1.T @ G1) / G1.shape[0]
            try:
                W_opt = np.linalg.inv(Omega)
            except np.linalg.LinAlgError:
                W_opt = np.linalg.pinv(Omega)
            sol2 = minimize(_obj_b, sol1.x, args=(W_opt,), method='L-BFGS-B',
                           options=dict(maxiter=300, ftol=1e-10))
            beta_b = sol2.x

            if np.any(np.abs(beta_b) > 10):
                continue

            blip_b = _blip_b(beta_b)
            pred_h = 1 - np.exp(-np.exp(lin0 + blip_b))
            hw = pred_h * w
            Zc = analysis["Z_clone"].to_numpy()
            t_arr = analysis["t"].to_numpy()
            cum = {}
            for z_val in [0, 1]:
                mz = Zc == z_val
                surv = 1.0
                for tv in sorted(np.unique(t_arr)):
                    sel = mz & (t_arr == tv)
                    if sel.any():
                        hz = hw[sel].sum() / w[sel].sum()
                    else:
                        hz = 0.0
                    surv *= (1.0 - hz)
                cum[z_val] = 1.0 - surv
            boot_rds.append(cum[1] - cum[0])
        except Exception:
            pass

    return boot_rds


# True and modeled duration curves
def _true_dynamic_curves(dgp, max_dur=T):
    k    = int(dgp["knot"])
    on1  = float(dgp["dyn_on1"]);   on2  = float(dgp["dyn_on2"])
    off1 = float(dgp["dyn_off1"]);  off2 = float(dgp["dyn_off2"])
    nev1 = float(dgp.get("dyn_never1", 0.0)); nev2 = float(dgp.get("dyn_never2", 0.0))
    beta_A = float(dgp.get("beta_A_to_Y", 0.0))
    d = np.arange(0, max_dur + 1)
    s1 = np.minimum(d, k); s2 = np.maximum(d - k, 0)
    on_mask = (d > 0).astype(float)
    true_on    = beta_A * on_mask + on1 * s1 + on2 * s2
    true_wash  = off1 * s1 + off2 * s2
    true_never = nev1 * s1 + nev2 * s2
    return dict(on=true_on, wash=true_wash, never=true_never)

# Bootstrap
def _single_bootstrap_iter(b, boot_ids, df, grace_len, trunc, Tt, Td,
                           fixed_knots_on=None, fixed_knots_wash=None,
                           fixed_knots_never=None, beta0_init=None):
    """
    Single bootstrap iteration. Runs Canonical TTE and SNMM on one bootstrap sample.
    """
    pieces = []
    for new_id, orig_id in enumerate(boot_ids):
        g = df.loc[df["id"] == orig_id].copy()
        g["id"] = new_id
        pieces.append(g)
    dfb = pd.concat(pieces, ignore_index=True)

    result = {
        "b": b,
        "tte_curve_0": None, "tte_curve_1": None, "tte_rd": np.nan,
        "snmm_curve_0": None, "snmm_curve_1": None, "snmm_rd": np.nan,
        "snmm_on": None, "snmm_wash": None, "snmm_never": None,
        "error": None
    }

    try:
        tte_res = run_canonical_tte(dfb, trunc=trunc, verbose=False)
        tte = tte_res["joint_summary"]

        tmp0 = tte[tte["Z"] == 0].set_index("t")["cum_event"]
        tmp0 = tmp0.reindex(range(1, Tt+1)).ffill().fillna(0.0)
        result["tte_curve_0"] = tmp0.to_numpy()

        tmp1 = tte[tte["Z"] == 1].set_index("t")["cum_event"]
        tmp1 = tmp1.reindex(range(1, Tt+1)).ffill().fillna(0.0)
        result["tte_curve_1"] = tmp1.to_numpy()

        result["tte_rd"] = tte_res["est_RD"]

    except Exception as e:
        result["error"] = f"[Bootstrap {b}] Canonical TTE failed: {e}"
        result["tte_curve_0"] = np.full(Tt, np.nan)
        result["tte_curve_1"] = np.full(Tt, np.nan)

    try:
        snmm_res = run_snmm_g(dfb, grace_len=grace_len, trunc=trunc,
                              fixed_knots_on=fixed_knots_on,
                              fixed_knots_wash=fixed_knots_wash,
                              fixed_knots_never=fixed_knots_never,
                              beta0_init=beta0_init,
                              verbose=False)
        ssum = snmm_res["joint_summary"]

        tmp0 = ssum[ssum["Z_clone"] == 0].set_index("t")["cum_event"]
        tmp0 = tmp0.reindex(range(1, Tt+1)).ffill().fillna(0.0)
        result["snmm_curve_0"] = tmp0.to_numpy()

        tmp1 = ssum[ssum["Z_clone"] == 1].set_index("t")["cum_event"]
        tmp1 = tmp1.reindex(range(1, Tt+1)).ffill().fillna(0.0)
        result["snmm_curve_1"] = tmp1.to_numpy()

        result["snmm_rd"] = snmm_res["est_RD"]

        m_on    = snmm_res["meta"]["modeled_on"]
        m_wash  = snmm_res["meta"]["modeled_wash"]
        m_never = snmm_res["meta"]["modeled_never"]
        if m_on is None or not isinstance(m_on, np.ndarray):
            m_on = np.zeros(Td + 1)
        if m_wash is None or not isinstance(m_wash, np.ndarray):
            m_wash = np.zeros(Td + 1)
        if m_never is None or not isinstance(m_never, np.ndarray):
            m_never = np.zeros(Td + 1)
        result["snmm_on"]    = m_on
        result["snmm_wash"]  = m_wash
        result["snmm_never"] = m_never

    except Exception as e:
        if result["error"]:
            result["error"] += f"; [Bootstrap {b}] SNMM failed: {e}"
        else:
            result["error"] = f"[Bootstrap {b}] SNMM failed: {e}"
        result["snmm_curve_0"] = np.full(Tt, np.nan)
        result["snmm_curve_1"] = np.full(Tt, np.nan)
        result["snmm_on"]    = np.full(Td + 1, np.nan)
        result["snmm_wash"]  = np.full(Td + 1, np.nan)
        result["snmm_never"] = np.full(Td + 1, np.nan)

    return result


def bootstrap_all(df, draws, dgp, B=B_BOOT, seed=2025, grace_len=GRACE, trunc=TRUNC,
                  n_jobs=-1):
    """Cluster bootstrap for the single-dataset run."""
    rng = np.random.default_rng(seed)
    ids = df["id"].unique()
    Tt = int(df["t"].max())
    Td = int(dgp["T"])

    true_rd, true0, true1 = compute_true_itt(draws, dgp)

    base_tte  = run_canonical_tte(df, trunc=trunc, verbose=False)
    base_snmm = run_snmm_g(df, grace_len=grace_len, trunc=trunc, verbose=False)

    fk_on    = base_snmm["meta"]["knots_on"]
    fk_wash  = base_snmm["meta"]["knots_wash"]
    fk_never = base_snmm["meta"]["knots_never"]
    base_beta = base_snmm["meta"]["beta"]
    base_tte_rd  = base_tte["est_RD"]
    base_snmm_rd = base_snmm["est_RD"]

    print("\n[BOOTSTRAP DEBUG] Baseline RDs on full data:")
    print(f"  True ITT RD            : {true_rd:+.4f}")
    print(f"  Single-run TTE RD      : {base_tte_rd:+.4f}")
    print(f"  Single-run SNMM RD     : {base_snmm_rd:+.4f}")
    print(f"  #clusters (ids)        : {len(ids)}")
    print(f"  #rows in df            : {len(df)}")

    boot_indices = [rng.choice(ids, size=len(ids), replace=True) for _ in range(B)]

    print(f"\n[BOOTSTRAP] Running {B} iterations with n_jobs={n_jobs}...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_single_bootstrap_iter)(b, boot_indices[b], df, grace_len, trunc, Tt, Td,
                                        fixed_knots_on=fk_on, fixed_knots_wash=fk_wash,
                                        fixed_knots_never=fk_never,
                                        beta0_init=base_beta)
        for b in range(B)
    )

    for r in results:
        if r["error"]:
            print(r["error"])

    def bands():
        return {0: np.zeros((B, Tt)), 1: np.zeros((B, Tt))}

    c_true, c_tte, c_snmm = bands(), bands(), bands()
    rds_true = np.zeros(B)
    rds_tte  = np.zeros(B)
    rds_snmm = np.zeros(B)
    snmm_on_curves    = np.zeros((B, Td + 1))
    snmm_wash_curves  = np.zeros((B, Td + 1))
    snmm_never_curves = np.zeros((B, Td + 1))

    for r in results:
        b = r["b"]
        c_true[0][b, :] = true0
        c_true[1][b, :] = true1
        rds_true[b]     = true_rd

        c_tte[0][b, :] = r["tte_curve_0"]
        c_tte[1][b, :] = r["tte_curve_1"]
        rds_tte[b]     = r["tte_rd"]

        c_snmm[0][b, :] = r["snmm_curve_0"]
        c_snmm[1][b, :] = r["snmm_curve_1"]
        rds_snmm[b]     = r["snmm_rd"]
        snmm_on_curves[b, :]    = r["snmm_on"]
        snmm_wash_curves[b, :]  = r["snmm_wash"]
        snmm_never_curves[b, :] = r["snmm_never"]

    def ci_band(arr):
        arr = np.asarray(arr, dtype=float)
        mean = np.nanmean(arr, axis=0)
        se   = np.nanstd(arr, axis=0, ddof=1)
        lo   = mean - 1.96 * se
        hi   = mean + 1.96 * se
        return mean, lo, hi

    bands_out = dict(
        true_itt={z: ci_band(c_true[z]) for z in [0, 1]},
        tte     ={z: ci_band(c_tte[z])  for z in [0, 1]},
        snmm    ={z: ci_band(c_snmm[z]) for z in [0, 1]}
    )

    def rd_ci(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return (np.nan, np.nan, np.nan)
        m  = float(np.mean(x))
        se = float(np.std(x, ddof=1))
        return (m, m - 1.96 * se, m + 1.96 * se)

    rds_tte_ci  = rd_ci(rds_tte)
    rds_snmm_ci = rd_ci(rds_snmm)

    def ci_band_se(arr):
        arr = np.asarray(arr, dtype=float)
        med  = np.nanmedian(arr, axis=0)
        mad  = np.nanmedian(np.abs(arr - med[None, :]), axis=0)
        se   = mad * 1.4826
        lo   = med - 1.96 * se
        hi   = med + 1.96 * se
        return med, lo, hi

    snmm_on_med,    snmm_on_lo,    snmm_on_hi    = ci_band_se(snmm_on_curves)
    snmm_wash_med,  snmm_wash_lo,  snmm_wash_hi  = ci_band_se(snmm_wash_curves)
    snmm_never_med, snmm_never_lo, snmm_never_hi = ci_band_se(snmm_never_curves)

    dyn_curves = dict(
        snmm_on    = dict(median=snmm_on_med,    lo=snmm_on_lo,    hi=snmm_on_hi),
        snmm_wash  = dict(median=snmm_wash_med,  lo=snmm_wash_lo,  hi=snmm_wash_hi),
        snmm_never = dict(median=snmm_never_med, lo=snmm_never_lo, hi=snmm_never_hi)
    )

    print("\n[BOOTSTRAP DEBUG] RD summary:")
    print(f"  True ITT RD (fixed)     : {true_rd:+.4f}")
    print(f"  Single-run TTE RD       : {base_tte_rd:+.4f}")
    print(f"  Single-run SNMM RD      : {base_snmm_rd:+.4f}")
    print(f"  Bootstrap mean TTE RD   : {rds_tte_ci[0]:+.4f}")
    print(f"  Bootstrap mean SNMM RD  : {rds_snmm_ci[0]:+.4f}")

    return dict(
        bands      = bands_out,
        rds        = dict(true_itt=(true_rd,true_rd,true_rd),
                          tte=rds_tte_ci,
                          snmm=rds_snmm_ci),
        rd_draws   = dict(true_itt=rds_true, tte=rds_tte, snmm=rds_snmm),
        dyn_curves = dyn_curves
    )


# Monte Carlo

def _run_single_mc_replication(sim_id, N, T, grace_len, conf_strength, base_seed,
                                trunc, on_q, wash_q, never_q,
                                fixed_knots_on=None, fixed_knots_wash=None,
                                fixed_knots_never=None, b_mc_boot=0,
                                **dgp_kwargs):
    """
    Run a single Monte Carlo replication.

    Uses analytical sandwich SEs (no inner bootstrap) for CI computation.

    Returns dict with:
      - true_rd: true ITT risk difference
      - tte_rd: Canonical TTE point estimate
      - adj_tte_rd: Adjusted TTE point estimate (includes baseline X)
      - snmm_rd: SNMM point estimate
      - *_se: analytical standard errors
      - *_ci_lo/hi: analytical 95% CIs
      - *_covers: coverage indicators
    """
    seed = base_seed + sim_id * 1000

    result = {
        "sim_id": sim_id,
        "true_rd": np.nan,
        "tte_rd": np.nan,
        "adj_tte_rd": np.nan,
        "snmm_rd": np.nan,
        "tte_se": np.nan,
        "adj_tte_se": np.nan,
        "snmm_se": np.nan,
        "tte_ci_lo": np.nan,
        "tte_ci_hi": np.nan,
        "adj_tte_ci_lo": np.nan,
        "adj_tte_ci_hi": np.nan,
        "snmm_ci_lo": np.nan,
        "snmm_ci_hi": np.nan,
        "tte_covers": np.nan,
        "adj_tte_covers": np.nan,
        "snmm_covers": np.nan,
        "snmm_boot_se": np.nan,
        "snmm_boot_ci_lo": np.nan,
        "snmm_boot_ci_hi": np.nan,
        "snmm_boot_covers": np.nan,
        "error": None
    }

    try:
        df, draws, dgp = simulate_trial_data(
            N=N, T=T, grace_len=grace_len,
            conf_strength=conf_strength, seed=seed,
            **dgp_kwargs
        )

        true_rd, true_c0, true_c1 = compute_true_itt(draws, dgp)
        result["true_rd"] = true_rd

        tte_res = run_canonical_tte(df, trunc=trunc, verbose=False)
        result["tte_rd"] = tte_res["est_RD"]

        adj_tte_res = run_adjusted_tte(df, trunc=trunc, verbose=False)
        result["adj_tte_rd"] = adj_tte_res["est_RD"]

        snmm_res = run_snmm_g(df, grace_len=grace_len, trunc=trunc,
                              on_q=on_q, wash_q=wash_q, never_q=never_q,
                              fixed_knots_on=fixed_knots_on,
                              fixed_knots_wash=fixed_knots_wash,
                              fixed_knots_never=fixed_knots_never,
                              verbose=False)
        result["snmm_rd"] = snmm_res["est_RD"]

        tte_gs = tte_res["joint_summary"]
        snmm_gs = snmm_res["joint_summary"]
        mc_data = {}
        mc_data["true_c0"] = np.asarray(true_c0, dtype=float)
        mc_data["true_c1"] = np.asarray(true_c1, dtype=float)
        for z in [0, 1]:
            mc_data[f"tte_c{z}"] = tte_gs[tte_gs["Z"]==z].sort_values("t")["cum_event"].values.astype(float)
            mc_data[f"snmm_c{z}"] = snmm_gs[snmm_gs["Z_clone"]==z].sort_values("t")["cum_event"].values.astype(float)

        meta = snmm_res["meta"]
        mc_data["on_eff"] = np.asarray(meta["modeled_on"]) if meta["modeled_on"] is not None else np.zeros(T + 1)
        mc_data["wash_eff"] = np.asarray(meta["modeled_wash"]) if meta["modeled_wash"] is not None else np.zeros(T + 1)
        mc_data["never_eff"] = np.asarray(meta["modeled_never"])

        result["J_stat"] = snmm_res.get("J_stat", np.nan)
        result["J_pval"] = snmm_res.get("J_pval", np.nan)

        result["_mc_data"] = mc_data

        tte_se = tte_res.get("se_rd", np.nan)
        result["tte_se"] = tte_se
        if not np.isnan(tte_se):
            result["tte_ci_lo"] = tte_res["est_RD"] - 1.96 * tte_se
            result["tte_ci_hi"] = tte_res["est_RD"] + 1.96 * tte_se
            result["tte_covers"] = (result["tte_ci_lo"] <= true_rd <= result["tte_ci_hi"])

        adj_tte_se = adj_tte_res.get("se_rd", np.nan)
        result["adj_tte_se"] = adj_tte_se
        if not np.isnan(adj_tte_se):
            result["adj_tte_ci_lo"] = adj_tte_res["est_RD"] - 1.96 * adj_tte_se
            result["adj_tte_ci_hi"] = adj_tte_res["est_RD"] + 1.96 * adj_tte_se
            result["adj_tte_covers"] = (result["adj_tte_ci_lo"] <= true_rd <= result["adj_tte_ci_hi"])

        snmm_se = snmm_res.get("se_rd", np.nan)
        result["snmm_se"] = snmm_se
        if not np.isnan(snmm_se):
            result["snmm_ci_lo"] = snmm_res["est_RD"] - 1.96 * snmm_se
            result["snmm_ci_hi"] = snmm_res["est_RD"] + 1.96 * snmm_se
            result["snmm_covers"] = (result["snmm_ci_lo"] <= true_rd <= result["snmm_ci_hi"])

        if b_mc_boot > 0 and not np.isnan(result["snmm_rd"]):
            boot_rds = _fast_snmm_bootstrap_rd(
                df, snmm_res, grace_len=grace_len, trunc=trunc,
                b_boot=b_mc_boot, seed=seed + 777)

            if len(boot_rds) >= 10:
                boot_se = float(np.std(boot_rds, ddof=1))
                result["snmm_boot_se"] = boot_se
                result["snmm_boot_ci_lo"] = result["snmm_rd"] - 1.96 * boot_se
                result["snmm_boot_ci_hi"] = result["snmm_rd"] + 1.96 * boot_se
                result["snmm_boot_covers"] = (
                    result["snmm_boot_ci_lo"] <= true_rd <= result["snmm_boot_ci_hi"])

    except Exception as e:
        result["error"] = str(e)

    return result


def run_monte_carlo(M=M_SIMS, N=N, T=T, grace_len=GRACE, conf_strength=CONF_STRENGTH,
                    base_seed=SEED, trunc=TRUNC,
                    on_q=ON_KNOT_Q, wash_q=WASH_KNOT_Q, never_q=NEVER_KNOT_Q,
                    n_jobs=-1, verbose=True, b_mc_boot=B_MC_BOOT, **dgp_kwargs):
    """Run the Monte Carlo simulation."""
    if verbose:
        print(f"\n[MONTE CARLO] Running {M} replications (analytical + bootstrap SEs)...")
        print(f"  N={N}, T={T}, grace={grace_len}, conf_strength={conf_strength}")
        print(f"  B_MC_BOOT={B_MC_BOOT}, FIX_KNOTS_IN_MC={FIX_KNOTS_IN_MC}")
        if dgp_kwargs:
            print(f"  Extra DGP kwargs: {dgp_kwargs}")

    ref_knots_on = ref_knots_wash = ref_knots_never = None
    if FIX_KNOTS_IN_MC:
        ref_df, _, _ = simulate_trial_data(
            N=N, T=T, grace_len=grace_len,
            conf_strength=conf_strength, seed=base_seed, **dgp_kwargs)
        ref_df["Z_clone"] = ref_df["Z"]
        _, ref_knots_on, ref_knots_wash, ref_knots_never = add_spline_timers(
            ref_df, grace_len, on_q=on_q, wash_q=wash_q, never_q=never_q)
        if verbose:
            print(f"  Fixed ON knots:   {ref_knots_on}")
            print(f"  Fixed WASH knots: {ref_knots_wash}")

    results = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
        delayed(_run_single_mc_replication)(
            sim_id=m, N=N, T=T, grace_len=grace_len, conf_strength=conf_strength,
            base_seed=base_seed, trunc=trunc,
            on_q=on_q, wash_q=wash_q, never_q=never_q,
            fixed_knots_on=ref_knots_on, fixed_knots_wash=ref_knots_wash,
            fixed_knots_never=ref_knots_never,
            b_mc_boot=b_mc_boot,
            **dgp_kwargs
        )
        for m in range(M)
    )

    mc_data_list = []
    for r in results:
        mc_data_list.append(r.pop("_mc_data", None))

    results_df = pd.DataFrame(results)

    errors = results_df[results_df["error"].notna()]
    if len(errors) > 0 and verbose:
        print(f"\n[MC WARNING] {len(errors)} replications had errors")

    valid = results_df[results_df["error"].isna()].copy()

    def summarize_estimator(est_col, true_col="true_rd", ci_lo_col=None, ci_hi_col=None, covers_col=None):
        est = valid[est_col].to_numpy()
        true = valid[true_col].to_numpy()

        bias = np.mean(est - true)
        emp_se = np.std(est, ddof=1)
        rmse = np.sqrt(np.mean((est - true) ** 2))
        mean_est = np.mean(est)
        mean_true = np.mean(true)

        summary = {
            "mean_true": mean_true,
            "mean_est": mean_est,
            "bias": bias,
            "emp_se": emp_se,
            "rmse": rmse,
            "n_valid": len(est)
        }

        if covers_col and covers_col in valid.columns:
            covers = valid[covers_col].dropna()
            if len(covers) > 0:
                summary["coverage"] = covers.mean()
                summary["n_coverage"] = len(covers)
            else:
                summary["coverage"] = np.nan
                summary["n_coverage"] = 0

        return summary

    summary = {
        "tte": summarize_estimator("tte_rd", covers_col="tte_covers"),
        "adj_tte": summarize_estimator("adj_tte_rd", covers_col="adj_tte_covers"),
        "snmm": summarize_estimator("snmm_rd", covers_col="snmm_covers")
    }

    for est_key, se_col in [("tte", "tte_se"), ("adj_tte", "adj_tte_se"), ("snmm", "snmm_se")]:
        if se_col in valid.columns:
            se_vals = valid[se_col].dropna()
            if len(se_vals) > 0:
                mean_analytic = float(se_vals.mean())
                summary[est_key]["mean_analytic_se"] = mean_analytic
                emp_se = summary[est_key]["emp_se"]
                if emp_se > 0:
                    summary[est_key]["se_ratio"] = mean_analytic / emp_se

    if "snmm_boot_se" in valid.columns:
        boot_se_vals = valid["snmm_boot_se"].dropna()
        if len(boot_se_vals) > 0:
            summary["snmm"]["mean_boot_se"] = float(boot_se_vals.mean())
            emp_se = summary["snmm"]["emp_se"]
            if emp_se > 0:
                summary["snmm"]["boot_se_ratio"] = float(boot_se_vals.mean()) / emp_se
        boot_covers = valid["snmm_boot_covers"].dropna()
        if len(boot_covers) > 0:
            summary["snmm"]["boot_coverage"] = float(boot_covers.mean())
            summary["snmm"]["n_boot_coverage"] = len(boot_covers)

    for est_key in ("tte", "adj_tte", "snmm"):
        s = summary[est_key]
        emp_se = s.get("emp_se", np.nan)
        bias = s.get("bias", np.nan)
        if emp_se and emp_se > 0 and np.isfinite(bias):
            r = bias / emp_se
            ceil = float(norm.cdf(1.96 - r) - norm.cdf(-1.96 - r))
            s["cov_ceiling"] = ceil
            s["bias_coverage_loss"] = 0.95 - ceil
            if "coverage" in s and np.isfinite(s.get("coverage", np.nan)):
                s["var_coverage_loss_analytic"] = ceil - s["coverage"]
            if "boot_coverage" in s and np.isfinite(s.get("boot_coverage", np.nan)):
                s["var_coverage_loss_boot"] = ceil - s["boot_coverage"]

    if verbose:
        print(f"\n[MONTE CARLO RESULTS] ({len(valid)} valid replications)")
        print(f"  Mean True ITT RD: {summary['tte']['mean_true']:.4f}")
        for est_label, est_key in [("Canonical TTE", "tte"), ("Adjusted TTE", "adj_tte"), ("SNMM", "snmm")]:
            s = summary[est_key]
            print(f"\n  {est_label}:")
            print(f"    Mean estimate:    {s['mean_est']:.4f}")
            print(f"    Bias:             {s['bias']:+.4f}")
            print(f"    Empirical SE:     {s['emp_se']:.4f}")
            if "mean_analytic_se" in s:
                print(f"    Mean analytic SE: {s['mean_analytic_se']:.4f}")
            if "se_ratio" in s:
                print(f"    SE ratio (A/E):   {s['se_ratio']:.3f}")
            if "mean_boot_se" in s:
                print(f"    Mean boot SE:     {s['mean_boot_se']:.4f}")
            if "boot_se_ratio" in s:
                print(f"    Boot SE ratio:    {s['boot_se_ratio']:.3f}")
            print(f"    RMSE:             {s['rmse']:.4f}")
            if "coverage" in s:
                print(f"    Coverage (ana):   {s['coverage']:.1%} (n={s['n_coverage']})")
            if "boot_coverage" in s:
                print(f"    Coverage (boot):  {s['boot_coverage']:.1%} (n={s['n_boot_coverage']})")
            if "cov_ceiling" in s:
                print(f"    Coverage ceiling: {s['cov_ceiling']:.1%} (max achievable given bias)")
                if "var_coverage_loss_analytic" in s:
                    print(f"      - loss to bias:        {s['bias_coverage_loss']:+.1%}")
                    print(f"      - loss to analytic SE: {s['var_coverage_loss_analytic']:+.1%}")
                if "var_coverage_loss_boot" in s:
                    print(f"      - loss to bootstrap SE:{s['var_coverage_loss_boot']:+.1%}")

    valid_mc_data = [d for d in mc_data_list if d is not None]
    mc_curves = {}
    mc_dyn = {}
    if valid_mc_data:
        for key in ["true_c0", "true_c1", "tte_c0", "tte_c1", "snmm_c0", "snmm_c1"]:
            arr = np.array([d[key] for d in valid_mc_data])
            m = np.nanmean(arr, axis=0)
            s = np.nanstd(arr, axis=0, ddof=1)
            mc_curves[key] = (m, m - 1.96 * s, m + 1.96 * s)

        for key in ["on_eff", "wash_eff", "never_eff"]:
            arr = np.array([d[key] for d in valid_mc_data])
            m = np.nanmean(arr, axis=0)
            s = np.nanstd(arr, axis=0, ddof=1)
            mc_dyn[key] = {"mean": m, "lo": m - 1.96 * s, "hi": m + 1.96 * s}

    return {"results_df": results_df, "summary": summary,
            "mc_curves": mc_curves, "mc_dyn": mc_dyn}


def run_scenario_sensitivity(scenarios, base_seed=SEED, M_per_scenario=50,
                              n_jobs=-1, verbose=True, b_mc_boot=0):
    """Run the scenario grid."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCENARIO SENSITIVITY ANALYSIS")
        print(f"{'='*70}")
        print(f"  {len(scenarios)} scenarios x {M_per_scenario} replications each")
        print(f"  Using analytical sandwich SEs for coverage")
        print()

    all_results = []

    _reserved_keys = {"label", "N", "T", "grace_len", "conf_strength"}

    for i, scenario in enumerate(scenarios):
        label = scenario.get("label", f"Scenario {i+1}")
        N_s = scenario.get("N", N)
        T_s = scenario.get("T", T)
        grace_s = scenario.get("grace_len", GRACE)
        conf_s = scenario.get("conf_strength", CONF_STRENGTH)

        extra_dgp = {k: v for k, v in scenario.items() if k not in _reserved_keys}

        if verbose:
            extra_str = f", {extra_dgp}" if extra_dgp else ""
            print(f"\n[{i+1}/{len(scenarios)}] {label}: N={N_s}, grace={grace_s}, conf={conf_s}{extra_str}")

        mc_result = run_monte_carlo(
            M=M_per_scenario,
            N=N_s,
            T=T_s,
            grace_len=grace_s,
            conf_strength=conf_s,
            base_seed=base_seed + i * 10000,
            n_jobs=n_jobs,
            verbose=False,
            b_mc_boot=b_mc_boot,
            **extra_dgp
        )

        summary = mc_result["summary"]

        est_labels = {
            "tte": "Canonical TTE",
            "adj_tte": "Adjusted TTE",
            "snmm": "SNMM"
        }
        for est_name, est_summary in summary.items():
            row = {
                "Scenario": label,
                "N": N_s,
                "conf_strength": conf_s,
                "Estimator": est_labels.get(est_name, est_name),
                "True RD": est_summary["mean_true"],
                "Mean Est": est_summary["mean_est"],
                "Bias": est_summary["bias"],
                "Emp SE": est_summary["emp_se"],
                "RMSE": est_summary["rmse"],
                "n_valid": est_summary["n_valid"]
            }
            if "coverage" in est_summary:
                row["Coverage"] = est_summary["coverage"]
            if "boot_coverage" in est_summary:
                row["Boot Cov"] = est_summary["boot_coverage"]
            if "cov_ceiling" in est_summary:
                row["Coverage ceiling"] = est_summary["cov_ceiling"]
            if "se_ratio" in est_summary:
                row["SE ratio"] = est_summary["se_ratio"]
            all_results.append(row)

    results_df = pd.DataFrame(all_results)

    if verbose:
        print(f"\n{'='*70}")
        print("SCENARIO SENSITIVITY RESULTS")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))

    return results_df


def build_table_3_mc_performance(mc_result, scenario_results=None):
    """Table 3: Monte Carlo performance."""
    rows = []

    summary = mc_result["summary"]
    est_labels = {
        "tte": "Canonical TTE",
        "adj_tte": "Adjusted TTE",
        "snmm": "SNMM"
    }
    for est_name in ["tte", "adj_tte", "snmm"]:
        if est_name not in summary:
            continue
        s = summary[est_name]
        row = {
            "Scenario": "Baseline",
            "Estimator": est_labels[est_name],
            "True RD": f"{s['mean_true']:.4f}",
            "Bias": f"{s['bias']:+.5f}",
            "Emp SE": f"{s['emp_se']:.4f}",
            "RMSE": f"{s['rmse']:.4f}",
        }
        if "coverage" in s and not np.isnan(s.get("coverage", np.nan)):
            row["Coverage"] = f"{s['coverage']:.1%}"
        else:
            row["Coverage"] = "-"
        if est_name == "snmm" and "boot_coverage" in s:
            row["Boot Cov"] = f"{s['boot_coverage']:.1%}"
        else:
            row["Boot Cov"] = "-"
        rows.append(row)

    if isinstance(scenario_results, pd.DataFrame) and not scenario_results.empty:
        for _, sr in scenario_results.iterrows():
            row = {
                "Scenario": sr["Scenario"],
                "Estimator": sr["Estimator"],
                "True RD": f"{sr['True RD']:.4f}",
                "Bias": f"{sr['Bias']:+.5f}",
                "Emp SE": f"{sr['Emp SE']:.4f}",
                "RMSE": f"{sr['RMSE']:.4f}",
            }
            if "Coverage" in sr and not pd.isna(sr.get("Coverage")):
                row["Coverage"] = f"{sr['Coverage']:.1%}" if isinstance(sr["Coverage"], float) else sr["Coverage"]
            else:
                row["Coverage"] = "-"
            if "Boot Cov" in sr and not pd.isna(sr.get("Boot Cov")):
                row["Boot Cov"] = f"{sr['Boot Cov']:.1%}" if isinstance(sr["Boot Cov"], float) else sr["Boot Cov"]
            else:
                row["Boot Cov"] = "-"
            rows.append(row)

    return pd.DataFrame(rows)


# Main tables

def build_table_1_dgp_summary(df, draws, dgp):
    """
    Table 1: Consolidated DGP summary for main manuscript.
    Combines key simulation parameters with empirical summary statistics.
    """
    pA_z = df.groupby("Z")["A"].mean().to_dict()
    pY = df["Y"].mean()
    pC = df["C_ltfu"].mean()

    rows = [
        {"Parameter": "Sample size (N)", "Value": f"{int(dgp['N']):,}"},
        {"Parameter": "Follow-up periods (T)", "Value": str(int(dgp['T']))},
        {"Parameter": "Grace period", "Value": str(int(dgp['grace_len']))},
        {"Parameter": "Dynamic effect knot", "Value": str(int(dgp['knot']))},
        {"Parameter": "P(A=1 | Z=0)", "Value": f"{pA_z.get(0, np.nan):.3f}"},
        {"Parameter": "P(A=1 | Z=1)", "Value": f"{pA_z.get(1, np.nan):.3f}"},
        {"Parameter": "Instrument strength", "Value": f"{pA_z.get(1, 0) - pA_z.get(0, 0):.3f}"},
        {"Parameter": "P(Y=1) per period", "Value": f"{pY:.3f}"},
        {"Parameter": "P(LTFU=1) per period", "Value": f"{pC:.3f}"},
        {"Parameter": "Unmeasured confounding strength", "Value": str(dgp['conf_strength'])},
        {"Parameter": "ON effect (before knot)", "Value": f"{dgp['dyn_on1']:.2f}"},
        {"Parameter": "ON effect (after knot)", "Value": f"{dgp['dyn_on2']:.2f}"},
        {"Parameter": "Washout effect (before knot)", "Value": f"{dgp['dyn_off1']:.2f}"},
        {"Parameter": "Washout effect (after knot)", "Value": f"{dgp['dyn_off2']:.2f}"},
        {"Parameter": "Never-treated effect (before knot)", "Value": f"{dgp.get('dyn_never1', 0.0):.2f}"},
        {"Parameter": "Never-treated effect (after knot)", "Value": f"{dgp.get('dyn_never2', 0.0):.2f}"},
    ]
    return pd.DataFrame(rows)


def build_table_2_performance(boot, true_itt_rd, snmm_result=None):
    """Table 2: bootstrap estimator performance."""
    rows = []
    labels = {
        "tte": "Canonical TTE",
        "snmm": "SNMM (g-estimation)"
    }

    for key in ["tte", "snmm"]:
        rd_draws = np.asarray(boot["rd_draws"][key], float)
        rd_draws = rd_draws[~np.isnan(rd_draws)]

        if rd_draws.size == 0:
            mean = sd = bias = rmse = coverage = np.nan
        else:
            mean = rd_draws.mean()
            sd = rd_draws.std(ddof=1)
            bias = mean - true_itt_rd
            rmse = np.sqrt(np.mean((rd_draws - true_itt_rd) ** 2))

            ci_lo = mean - 1.96 * sd
            ci_hi = mean + 1.96 * sd
            covered = (ci_lo <= true_itt_rd <= ci_hi)
            coverage = "Yes" if covered else "No"

        row = {
            "Estimator": labels[key],
            "True RD": f"{true_itt_rd:.4f}",
            "Mean RD": f"{mean:.4f}",
            "SD": f"{sd:.4f}",
            "Bias": f"{bias:+.4f}",
            "RMSE": f"{rmse:.4f}",
            "95% CI": f"({ci_lo:.4f}, {ci_hi:.4f})",
            "Covers True": coverage
        }
        if key == "snmm" and snmm_result is not None:
            J = snmm_result.get("J_stat", np.nan)
            Jp = snmm_result.get("J_pval", np.nan)
            row["J-stat"] = f"{J:.3f}" if not np.isnan(J) else "-"
            row["J p-value"] = f"{Jp:.4f}" if not np.isnan(Jp) else "-"
        else:
            row["J-stat"] = "-"
            row["J p-value"] = "-"
        rows.append(row)

    return pd.DataFrame(rows)


# Appendix tables

def build_table_A1_dgp(dgp):
    """
    Table A1: Full DGP parameters (for appendix).
    One row per parameter with a stub description you can edit.
    """
    desc_map = {
        "N": "Number of individuals",
        "T": "Max follow-up periods",
        "grace_len": "Length of grace period",
        "conf_strength": "Strength multiplier on U_t in A and Y models",
        "base_logitA": "Baseline logit for treatment (post-grace)",
        "beta_Z": "Effect of instrument Z on treatment",
        "beta_L_to_A": "Effect of L on treatment",
        "beta_X_to_A": "Effect of baseline X on treatment",
        "beta_U_to_A": "Effect of U_t on treatment (before conf_strength)",
        "rho_A": "Autoregressive effect of prior A on current A",
        "psi_off": "Penalty for being OFF after ever having been ON",
        "base_logitY": "Baseline cloglog linear predictor for outcome",
        "beta_L_to_Y": "Effect of L on outcome",
        "beta_A_to_Y": "Effect of A on outcome (baseline, not dynamic)",
        "beta_U_to_Y": "Effect of U_t on outcome (before conf_strength)",
        "base_logit_notC": "Baseline cloglog linear predictor for not censored",
        "gamma_A": "Effect of prior A on not being censored",
        "gamma_L": "Effect of L on not being censored",
        "p_dev_base": "Baseline grace-period deviation probability",
        "p_dev_L_coef": "Increment in deviation probability when L>threshold",
        "knot": "Change point for dynamic duration effects",
        "dyn_on1": "ON dynamic effect before knot",
        "dyn_on2": "ON dynamic effect after knot",
        "dyn_off1": "Washout dynamic effect before knot",
        "dyn_off2": "Washout dynamic effect after knot",
        "dyn_never1": "Never-treated dynamic effect before knot",
        "dyn_never2": "Never-treated dynamic effect after knot",
    }
    rows = []
    for k, v in dgp.items():
        rows.append({
            "parameter": k,
            "value": v,
            "description": desc_map.get(k, "")
        })
    dfA1 = pd.DataFrame(rows).sort_values("parameter").reset_index(drop=True)
    return dfA1


def build_table_A2_dgp_summary(df, draws, dgp):
    """
    Table A2: Detailed empirical summary of the simulated world (for appendix).
    """
    pA = df["A"].mean()
    pA_z = df.groupby("Z")["A"].mean().to_dict()
    pY = df["Y"].mean()
    pC = df["C_ltfu"].mean()

    corr_XL = df[["X", "L"]].corr().iloc[0, 1]
    snap = df[df["t"] == 1].copy()
    U_t1 = draws["U_path"][:, 0]
    corr_UL1 = np.corrcoef(U_t1, snap["L"].to_numpy())[0, 1]

    df_tmp = df.copy()
    df_tmp["Z_clone"] = df_tmp["Z"]
    df_timers, _, _, _ = add_spline_timers(df_tmp, grace_len=dgp["grace_len"])
    post = df_timers["t"] > dgp["grace_len"]
    t_on    = df_timers.loc[post, "t_on"]
    t_wash  = df_timers.loc[post & (df_timers["ever_on"] == 1), "t_off"]
    t_never = df_timers.loc[post & (df_timers["ever_on"] == 0), "t_never"]
    mean_t_on    = t_on[t_on > 0].mean()
    mean_t_wash  = t_wash[t_wash > 0].mean() if (t_wash > 0).any() else np.nan
    mean_t_never = t_never[t_never > 0].mean() if (t_never > 0).any() else np.nan

    row = {
        "P(A=1)": pA,
        "P(A=1|Z=0)": pA_z.get(0, np.nan),
        "P(A=1|Z=1)": pA_z.get(1, np.nan),
        "P(Y=1)": pY,
        "P(LTFU=1)": pC,
        "Corr(X,L)": corr_XL,
        "Corr(U_t,L) at t=1": corr_UL1,
        "Mean ON duration (t_on>0)": mean_t_on,
        "Mean washout duration (ever-on, t_off>0)": mean_t_wash,
        "Mean never-treated duration (t_never>0)": mean_t_never,
        "N_ids": df["id"].nunique(),
        "N_rows": len(df)
    }
    return pd.DataFrame([row])


def build_table_A4_iv_and_weights(df, est_tte, est_snmm):
    """
    Table A4: instrument strength + weight diagnostics.
    """
    pA_z = df.groupby("Z")["A"].mean()
    diff = pA_z[1] - pA_z[0]

    def summarize_weights(w):
        w = np.asarray(w, float)
        w = w[np.isfinite(w)]
        return {
            "mean": w.mean(),
            "sd": w.std(ddof=1),
            "min": w.min(),
            "p1": np.quantile(w, 0.01),
            "p99": np.quantile(w, 0.99),
            "max": w.max()
        }

    ws_tte = summarize_weights(est_tte["weights"])
    ws_snmm = summarize_weights(est_snmm["weights"])

    rows = [
        {
            "Quantity": "P(A=1 | Z=0)",
            "Canonical TTE": pA_z[0],
            "SNMM": np.nan
        },
        {
            "Quantity": "P(A=1 | Z=1)",
            "Canonical TTE": pA_z[1],
            "SNMM": np.nan
        },
        {
            "Quantity": "Difference P(A=1|Z=1) - P(A=1|Z=0)",
            "Canonical TTE": diff,
            "SNMM": np.nan
        },
        {
            "Quantity": "Mean stabilized weight",
            "Canonical TTE": ws_tte["mean"],
            "SNMM": ws_snmm["mean"]
        },
        {
            "Quantity": "SD stabilized weight",
            "Canonical TTE": ws_tte["sd"],
            "SNMM": ws_snmm["sd"]
        },
        {
            "Quantity": "Min stabilized weight",
            "Canonical TTE": ws_tte["min"],
            "SNMM": ws_snmm["min"]
        },
        {
            "Quantity": "1st percentile stabilized weight",
            "Canonical TTE": ws_tte["p1"],
            "SNMM": ws_snmm["p1"]
        },
        {
            "Quantity": "99th percentile stabilized weight",
            "Canonical TTE": ws_tte["p99"],
            "SNMM": ws_snmm["p99"]
        },
        {
            "Quantity": "Max stabilized weight",
            "Canonical TTE": ws_tte["max"],
            "SNMM": ws_snmm["max"]
        },
    ]
    return pd.DataFrame(rows)



# Appendix figures

def plot_weight_distributions(est_tte, est_snmm, bins=50, save_path=None):
    """Stabilized weight distributions."""
    w_tte = np.asarray(est_tte["weights"], float)
    w_snmm = np.asarray(est_snmm["weights"], float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    ax = axes[0]
    ax.hist(w_tte[np.isfinite(w_tte)], bins=bins, density=False)
    ax.set_title("Canonical TTE stabilized weights")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(w_snmm[np.isfinite(w_snmm)], bins=bins, density=False)
    ax.set_title("SNMM stabilized weights")
    ax.set_xlabel("Weight")
    ax.grid(alpha=0.3)

    fig.suptitle("Distributions of stabilized weights", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    plt.show()


def plot_adherence_patterns(df, grace_len=GRACE, save_path=None):
    """Grace-period adherence plot."""
    grace_df = df[df["t"] <= grace_len].copy()

    adherence = grace_df.groupby(["t", "Z"])["A"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7, 5))

    times = adherence.index.values
    width = 0.35
    x = np.arange(len(times))

    ax.bar(x - width/2, adherence[0], width, label="Z=0 (control)", color="C0", alpha=0.7)
    ax.bar(x + width/2, adherence[1], width, label="Z=1 (encouraged)", color="C1", alpha=0.7)

    ax.set_xlabel("Time (period)", fontsize=11)
    ax.set_ylabel("P(A=1 | Z)", fontsize=11)
    ax.set_title("Adherence Patterns During Grace Period", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={int(t)}" for t in times])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    for i, t in enumerate(times):
        ax.text(i - width/2, adherence.loc[t, 0] + 0.02, f"{adherence.loc[t, 0]:.2f}",
                ha="center", fontsize=9)
        ax.text(i + width/2, adherence.loc[t, 1] + 0.02, f"{adherence.loc[t, 1]:.2f}",
                ha="center", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    plt.show()

    return adherence


def build_table_attrition(df, grace_len=GRACE):
    """
    Appendix Table: Attrition/censoring analysis.
    Shows % lost to LTFU and % censored for non-adherence by time.
    """
    rows = []

    n_start = df["id"].nunique()

    for t in range(1, int(df["t"].max()) + 1):
        df_t = df[df["t"] == t]
        n_at_risk = len(df_t)

        n_ltfu = df_t["C_ltfu"].sum()
        pct_ltfu = 100 * n_ltfu / n_at_risk if n_at_risk > 0 else 0

        if t <= grace_len:
            n_nonadhere = ((df_t["A"] != df_t["Z"])).sum()
            pct_nonadhere = 100 * n_nonadhere / n_at_risk if n_at_risk > 0 else 0
        else:
            n_nonadhere = np.nan
            pct_nonadhere = np.nan

        n_events = df_t["Y"].sum()
        pct_events = 100 * n_events / n_at_risk if n_at_risk > 0 else 0

        rows.append({
            "Time": t,
            "N at risk": n_at_risk,
            "N events": int(n_events),
            "% events": round(pct_events, 2),
            "N LTFU": int(n_ltfu),
            "% LTFU": round(pct_ltfu, 2),
            "N non-adherent": int(n_nonadhere) if not np.isnan(n_nonadhere) else "-",
            "% non-adherent": round(pct_nonadhere, 2) if not np.isnan(pct_nonadhere) else "-"
        })

    total_ltfu = df.groupby("id")["C_ltfu"].max().sum()
    total_events = df.groupby("id")["Y"].max().sum()

    rows.append({
        "Time": "Total",
        "N at risk": n_start,
        "N events": int(total_events),
        "% events": round(100 * total_events / n_start, 2),
        "N LTFU": int(total_ltfu),
        "% LTFU": round(100 * total_ltfu / n_start, 2),
        "N non-adherent": "-",
        "% non-adherent": "-"
    })

    return pd.DataFrame(rows)


def build_table_covariate_balance(df):
    """
    Appendix Table: Covariate balance by Z at baseline.
    Shows that randomization is working.
    """
    baseline = df[df["t"] == 1].copy()

    rows = []

    for var in ["X", "L"]:
        z0 = baseline.loc[baseline["Z"] == 0, var]
        z1 = baseline.loc[baseline["Z"] == 1, var]

        rows.append({
            "Covariate": var,
            "Z=0 mean": round(z0.mean(), 3),
            "Z=0 SD": round(z0.std(), 3),
            "Z=1 mean": round(z1.mean(), 3),
            "Z=1 SD": round(z1.std(), 3),
            "Std. Diff": round((z1.mean() - z0.mean()) / np.sqrt((z0.var() + z1.var()) / 2), 3)
        })

    n0 = (baseline["Z"] == 0).sum()
    n1 = (baseline["Z"] == 1).sum()
    rows.append({
        "Covariate": "N",
        "Z=0 mean": n0,
        "Z=0 SD": "-",
        "Z=1 mean": n1,
        "Z=1 SD": "-",
        "Std. Diff": "-"
    })

    return pd.DataFrame(rows)


def plot_final_RD_bar(boot, true_itt_rd, save_path=None):
    """Final-time RD bar plot."""
    rds = boot["rds"]
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["tte", "snmm"]
    labels = ["Canonical TTE", "SNMM"]
    xpos = np.arange(len(names))
    means = [rds[k][0] for k in names]
    ax.bar(xpos, means, width=0.5, color=["C0", "C1"], alpha=0.7)

    for i, k in enumerate(names):
        mean, lo, hi = rds[k]
        ax.plot([i, i], [lo, hi], color="k", lw=2)
        ax.plot([i - 0.08, i + 0.08], [lo, lo], color="k", lw=1.5)
        ax.plot([i - 0.08, i + 0.08], [hi, hi], color="k", lw=1.5)

    ax.axhline(true_itt_rd, color="red", ls="--", lw=2,
               label=f"True ITT RD = {true_itt_rd:+.3f}")

    ax.set_xticks(xpos, labels, fontsize=12)
    ax.set_ylabel("Risk Difference at Final Time", fontsize=11)
    ax.set_title("Estimated vs True ITT Risk Difference", fontsize=12)
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    plt.show()


# Duration support plot
def plot_duration_support(df, grace_len=GRACE, max_dur=T, save_path=None):
    """
    Figure 2: Duration support / positivity diagnostics.
    Shows how many person-periods are observed at each ON and OFF duration.
    """
    df_tmp = df.copy()
    df_tmp["Z_clone"] = df_tmp["Z"]
    df_timers, _, _, _ = add_spline_timers(df_tmp, grace_len=grace_len)

    post = df_timers["t"] > grace_len
    t_on    = df_timers.loc[post, "t_on"]
    t_wash  = df_timers.loc[post & (df_timers["ever_on"] == 1), "t_off"]
    t_never = df_timers.loc[post & (df_timers["ever_on"] == 0), "t_never"]

    t_on    = t_on[t_on > 0]
    t_wash  = t_wash[t_wash > 0]
    t_never = t_never[t_never > 0]

    max_on    = int(min(max_dur, t_on.max()    if not t_on.empty    else 0))
    max_wash  = int(min(max_dur, t_wash.max()  if not t_wash.empty  else 0))
    max_never = int(min(max_dur, t_never.max() if not t_never.empty else 0))
    max_all = max(max_on, max_wash, max_never, 1)
    bins = np.arange(1, max_all + 1)

    on_counts    = t_on.value_counts().reindex(bins, fill_value=0).sort_index()
    wash_counts  = t_wash.value_counts().reindex(bins, fill_value=0).sort_index()
    never_counts = t_never.value_counts().reindex(bins, fill_value=0).sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.bar(on_counts.index, on_counts.values, width=0.8, color="C0")
    ax.set_xlabel("ON duration (t_on)")
    ax.set_ylabel("Number of person-periods")
    ax.set_title("ON durations (post-grace)")
    ax.set_xticks(bins)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(wash_counts.index, wash_counts.values, width=0.8, color="C1")
    ax.set_xlabel("Washout duration (t_off)")
    ax.set_title("Washout durations (post-grace, ever-on)")
    ax.set_xticks(bins)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(never_counts.index, never_counts.values, width=0.8, color="C2")
    ax.set_xlabel("Never-treated duration (t_never)")
    ax.set_title("Never-treated durations (post-grace)")
    ax.set_xticks(bins)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Duration support / positivity diagnostics", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_true_vs_snmm_dynamic_boot(boot, dgp, max_dur=T, save_path=None):
    """
    Figure 2: TRUE dynamic curves vs SNMM dynamic curves with bootstrap 95% CIs.
    Three components: ON, WASHOUT, NEVER-TREATED.
    """
    tru = _true_dynamic_curves(dgp, max_dur=max_dur)
    x   = np.arange(0, max_dur + 1)

    dyn   = boot["dyn_curves"]
    on    = dyn["snmm_on"]
    wash  = dyn["snmm_wash"]
    never = dyn["snmm_never"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    ax = axes[0]
    ax.plot(x, tru["on"], "k--", lw=2.5, label="True")
    ax.plot(x, on["median"], "C0-", lw=2.0, label="SNMM")
    ax.fill_between(x, on["lo"], on["hi"], color="C0", alpha=0.2, linewidth=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(int(dgp["knot"]), color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Duration (periods)", fontsize=11)
    ax.set_ylabel("Dynamic effect (cloglog scale)", fontsize=11)
    ax.set_title("ON (treatment)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, tru["wash"], "k--", lw=2.5, label="True")
    ax.plot(x, wash["median"], "C1-", lw=2.0, label="SNMM")
    ax.fill_between(x, wash["lo"], wash["hi"], color="C1", alpha=0.2, linewidth=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(int(dgp["knot"]), color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Duration (periods)", fontsize=11)
    ax.set_title("WASHOUT (post-discontinuation)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, tru["never"], "k--", lw=2.5, label="True")
    ax.plot(x, never["median"], "C2-", lw=2.0, label="SNMM")
    ax.fill_between(x, never["lo"], never["hi"], color="C2", alpha=0.2, linewidth=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(int(dgp["knot"]), color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Duration (periods)", fontsize=11)
    ax.set_title("NEVER-TREATED", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Figure 2: True vs SNMM-Estimated Dynamic Treatment Effects", fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# MC plots and tables

def plot_figure1_mc(mc_curves, T_val, grace_len=GRACE, save_path=None):
    """Figure 1 from MC: ITT cumulative incidence by encouragement arm."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)
    rows = [
        ("True ITT (twin-world)", "true"),
        ("Canonical TTE", "tte"),
        ("SNMM (g-estimation)", "snmm"),
    ]
    colors = {0: "C0", 1: "C1"}
    tvals = np.arange(1, T_val + 1)

    for ax, (title, key) in zip(axes, rows):
        for z in [0, 1]:
            mean, lo, hi = mc_curves[f"{key}_c{z}"]
            ax.plot(tvals, mean, color=colors[z], label=f"Z={z}")
            ax.fill_between(tvals, lo, hi, color=colors[z], alpha=0.25, linewidth=0)
        ax.axvline(x=grace_len, color="gray", ls="--", lw=1)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (periods)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Figure 1: ITT Cumulative Incidence by Encouragement Arm\n(MC mean with 95% CI)", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.show()


def plot_figure2_mc(mc_dyn, dgp, max_dur=T, save_path=None):
    """Figure 2 from MC draws."""
    tru = _true_dynamic_curves(dgp, max_dur=max_dur)
    x = np.arange(0, max_dur + 1)

    on = mc_dyn["on_eff"]
    wash = mc_dyn["wash_eff"]
    never = mc_dyn["never_eff"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    ax = axes[0]
    ax.plot(x, tru["on"], "k--", lw=2.5, label="True")
    ax.plot(x, on["mean"], "C0-", lw=2.0, label="SNMM (MC mean)")
    ax.fill_between(x, on["lo"], on["hi"], color="C0", alpha=0.2, linewidth=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(int(dgp["knot"]), color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Duration (periods)", fontsize=11)
    ax.set_ylabel("Dynamic effect (cloglog scale)", fontsize=11)
    ax.set_title("ON (treatment)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, tru["wash"], "k--", lw=2.5, label="True")
    ax.plot(x, wash["mean"], "C1-", lw=2.0, label="SNMM (MC mean)")
    ax.fill_between(x, wash["lo"], wash["hi"], color="C1", alpha=0.2, linewidth=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(int(dgp["knot"]), color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Duration (periods)", fontsize=11)
    ax.set_title("WASHOUT (post-discontinuation)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, tru["never"], "k--", lw=2.5, label="True")
    ax.plot(x, never["mean"], "C2-", lw=2.0, label="SNMM (MC mean)")
    ax.fill_between(x, never["lo"], never["hi"], color="C2", alpha=0.2, linewidth=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(int(dgp["knot"]), color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Duration (periods)", fontsize=11)
    ax.set_title("NEVER-TREATED", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Figure 2: True vs SNMM-Estimated Dynamic Treatment Effects\n(MC mean with 95% CI)", fontsize=13, y=1.04)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_figure3_mc(mc_result, save_path=None):
    """Figure 3 from MC: RD comparison using MC distribution."""
    rdf = mc_result["results_df"]
    valid = rdf[rdf["error"].isna()]
    summary = mc_result["summary"]
    true_rd = summary["tte"]["mean_true"]

    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["tte", "snmm"]
    labels = ["Canonical TTE", "SNMM"]
    xpos = np.arange(len(names))

    means = [summary[k]["mean_est"] for k in names]
    ses = [summary[k]["emp_se"] for k in names]

    ax.bar(xpos, means, width=0.5, color=["C0", "C1"], alpha=0.7)
    for i, (m, se) in enumerate(zip(means, ses)):
        lo, hi = m - 1.96 * se, m + 1.96 * se
        ax.plot([i, i], [lo, hi], color="k", lw=2)
        ax.plot([i - 0.08, i + 0.08], [lo, lo], color="k", lw=1.5)
        ax.plot([i - 0.08, i + 0.08], [hi, hi], color="k", lw=1.5)

    ax.axhline(true_rd, color="red", ls="--", lw=2,
               label=f"Mean True ITT RD = {true_rd:+.3f}")

    ax.set_xticks(xpos, labels, fontsize=12)
    ax.set_ylabel("Risk Difference at Final Time", fontsize=11)
    ax.set_title("MC Mean Estimated vs True ITT Risk Difference", fontsize=12)
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.show()


def build_table_2_mc_performance(mc_summary, mc_result_df=None):
    """Table 2 from MC replications."""
    rows = []
    labels = {
        "tte": "Canonical TTE",
        "adj_tte": "Adjusted TTE",
        "snmm": "SNMM (g-estimation)"
    }

    for key in ["tte", "adj_tte", "snmm"]:
        if key not in mc_summary:
            continue
        s = mc_summary[key]
        row = {
            "Estimator": labels[key],
            "True RD": f"{s['mean_true']:.4f}",
            "Mean RD": f"{s['mean_est']:.4f}",
            "Emp SE": f"{s['emp_se']:.4f}",
            "Bias": f"{s['bias']:+.4f}",
            "RMSE": f"{s['rmse']:.4f}",
        }
        def _pct(x):
            return f"{x:.1%}" if (x is not None and np.isfinite(x)) else "-"
        def _r3(x):
            return f"{x:.3f}" if (x is not None and np.isfinite(x)) else "-"

        ana_cov = s.get("coverage", np.nan)
        boot_cov = s.get("boot_coverage", np.nan)
        if key == "snmm" and np.isfinite(boot_cov):
            row["Coverage"] = _pct(boot_cov)
            row["Coverage basis"] = "bootstrap"
        else:
            row["Coverage"] = _pct(ana_cov)
            row["Coverage basis"] = "analytic"
        row["Coverage (analytic)"] = _pct(ana_cov)
        row["Coverage ceiling"] = _pct(s.get("cov_ceiling", np.nan))
        row["SE Ratio (ana/emp)"] = _r3(s.get("se_ratio", np.nan))
        row["Boot SE Ratio"] = _r3(s.get("boot_se_ratio", np.nan)) if key == "snmm" else "-"

        if key == "snmm" and mc_result_df is not None:
            valid = mc_result_df[mc_result_df["error"].isna()]
            if "J_stat" in valid.columns:
                j_vals = valid["J_stat"].dropna()
                jp_vals = valid["J_pval"].dropna()
                if len(j_vals) > 0:
                    row["J-stat (mean)"] = f"{j_vals.mean():.3f}"
                    row["J p-value (mean)"] = f"{jp_vals.mean():.4f}"
                else:
                    row["J-stat (mean)"] = "-"
                    row["J p-value (mean)"] = "-"
            else:
                row["J-stat (mean)"] = "-"
                row["J p-value (mean)"] = "-"
        else:
            row["J-stat (mean)"] = "-"
            row["J p-value (mean)"] = "-"

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"\n=== Dynamic scenario === N={N}, T={T}, GRACE={GRACE}, conf={CONF_STRENGTH} ===")
    df, draws, dgp = simulate_trial_data(
        N=N,
        T=T,
        grace_len=GRACE,
        conf_strength=CONF_STRENGTH,
        seed=SEED
    )

    true_itt_rd, _, _ = compute_true_itt(draws, dgp)
    print(f"[True ITT RD] {true_itt_rd:.4f}")

    tte = run_canonical_tte(df, trunc=TRUNC, verbose=True)
    adj_tte = run_adjusted_tte(df, trunc=TRUNC, verbose=True)
    snmm = run_snmm_g(
        df,
        grace_len=GRACE,
        trunc=TRUNC,
        verbose=True
    )

    print(f"\n[J-test] J={snmm['J_stat']:.3f}, df={snmm['J_df']}, p={snmm['J_pval']:.4f}")

    print("Single-run RDs (with analytical SEs):")
    print(f"  Canonical TTE : {tte['est_RD']:.4f}  SE={tte.get('se_rd', np.nan):.4f}")
    print(f"  Adjusted TTE  : {adj_tte['est_RD']:.4f}  SE={adj_tte.get('se_rd', np.nan):.4f}")
    print(f"  SNMM          : {snmm['est_RD']:.4f}  SE={snmm.get('se_rd', np.nan):.4f}")
    print(f"True ITT RD: {true_itt_rd:.4f}")

    table_1 = build_table_1_dgp_summary(df, draws, dgp)
    table_1.to_csv(os.path.join(BASE_OUTDIR, "Table1_DGP_Summary.csv"), index=False)
    print("\n" + "="*60)
    print("TABLE 1: Simulation Design and DGP Summary")
    print("="*60)
    print(table_1.to_string(index=False))

    table_A1 = build_table_A1_dgp(dgp)
    table_A2 = build_table_A2_dgp_summary(df, draws, dgp)
    table_A3 = build_table_A4_iv_and_weights(df, tte, snmm)
    table_A1.to_csv(os.path.join(APPENDIX_OUTDIR, "TableA1_Full_DGP_Parameters.csv"), index=False)
    table_A2.to_csv(os.path.join(APPENDIX_OUTDIR, "TableA2_Detailed_Summary.csv"), index=False)
    table_A3.to_csv(os.path.join(APPENDIX_OUTDIR, "TableA3_IPCW_Diagnostics.csv"), index=False)

    plot_duration_support(
        df, grace_len=GRACE, max_dur=T,
        save_path=os.path.join(APPENDIX_OUTDIR, "FigureA1_Duration_Distributions.png")
    )
    plot_weight_distributions(
        tte, snmm,
        save_path=os.path.join(APPENDIX_OUTDIR, "FigureA2_IPCW_Weight_Distributions.png")
    )

    table_A4 = build_table_attrition(df, grace_len=GRACE)
    table_A4.to_csv(os.path.join(APPENDIX_OUTDIR, "TableA4_Attrition_Patterns.csv"), index=False)
    table_A5 = build_table_covariate_balance(df)
    table_A5.to_csv(os.path.join(APPENDIX_OUTDIR, "TableA5_Covariate_Balance.csv"), index=False)

    if RUN_MC:
        print("\n" + "="*70)
        print("MONTE CARLO SIMULATION")
        print("="*70)

        mc = run_monte_carlo(
            M=M_SIMS, N=N, T=T, grace_len=GRACE,
            conf_strength=CONF_STRENGTH, base_seed=SEED,
            n_jobs=-1, verbose=True
        )
        print(f"[CHECKPOINT] MC baseline done. {len(mc['results_df'])} replications.", flush=True)

        print("[CHECKPOINT] Starting scenario sensitivity...", flush=True)
        scenarios = [
            {"label": "N=5000", "N": 5000, "conf_strength": CONF_STRENGTH},
            {"label": "N=20000", "N": 20000, "conf_strength": CONF_STRENGTH},
            {"label": "No confounding", "N": N, "conf_strength": 0},
            {"label": "Strong confounding", "N": N, "conf_strength": 2},
            {"label": "G=1", "N": N, "conf_strength": CONF_STRENGTH, "grace_len": 1},
            {"label": "G=5", "N": N, "conf_strength": CONF_STRENGTH, "grace_len": 5},
            {"label": "rho_A=0.30", "N": N, "conf_strength": CONF_STRENGTH, "rho_A": 0.30},
            {"label": "rho_A=0.70", "N": N, "conf_strength": CONF_STRENGTH, "rho_A": 0.70},
            {"label": "Null duration effects",
             "N": N, "conf_strength": CONF_STRENGTH,
             "dyn_on1": 0.0, "dyn_on2": 0.0,
             "dyn_off1": 0.0, "dyn_off2": 0.0},
            {"label": "Weak duration effects",
             "N": N, "conf_strength": CONF_STRENGTH,
             "dyn_on1": -0.11, "dyn_on2": -0.05,
             "dyn_off1": 0.09, "dyn_off2": 0.05},
            {"label": "Strong duration effects",
             "N": N, "conf_strength": CONF_STRENGTH,
             "dyn_on1": -0.33, "dyn_on2": -0.15,
             "dyn_off1": 0.27, "dyn_off2": 0.15},
        ]
        try:
            scenario_results = run_scenario_sensitivity(
                scenarios=scenarios, base_seed=SEED + 50000,
                M_per_scenario=M_SIMS,
                n_jobs=-1, verbose=True
            )
            print(f"[CHECKPOINT] Scenario sensitivity done. {len(scenario_results)} scenarios.", flush=True)
        except Exception as e:
            print(f"[ERROR] Scenario sensitivity failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            scenario_results = pd.DataFrame()

        print("[CHECKPOINT] Generating Figure 1...", flush=True)
        try:
            plot_figure1_mc(
                mc["mc_curves"], T_val=T, grace_len=GRACE,
                save_path=os.path.join(BASE_OUTDIR, "Figure1_ITT_Cumulative_Incidence.png")
            )
            print("[CHECKPOINT] Figure 1 saved.", flush=True)
        except Exception as e:
            print(f"[ERROR] Figure 1 failed: {e}", flush=True)
            import traceback; traceback.print_exc()

        print("[CHECKPOINT] Generating Figure 2...", flush=True)
        try:
            plot_figure2_mc(
                mc["mc_dyn"], dgp, max_dur=T,
                save_path=os.path.join(BASE_OUTDIR, "Figure2_Duration_Effects_True_vs_SNMM.png")
            )
            print("[CHECKPOINT] Figure 2 saved.", flush=True)
        except Exception as e:
            print(f"[ERROR] Figure 2 failed: {e}", flush=True)
            import traceback; traceback.print_exc()

        print("[CHECKPOINT] Generating Figure 3...", flush=True)
        try:
            plot_figure3_mc(
                mc,
                save_path=os.path.join(BASE_OUTDIR, "Figure3_Final_RiskDifference.png")
            )
            print("[CHECKPOINT] Figure 3 saved.", flush=True)
        except Exception as e:
            print(f"[ERROR] Figure 3 failed: {e}", flush=True)
            import traceback; traceback.print_exc()

        print("[CHECKPOINT] Building Table 2...", flush=True)
        table_2 = build_table_2_mc_performance(
            mc["summary"], mc_result_df=mc["results_df"]
        )
        table_2.to_csv(os.path.join(BASE_OUTDIR, "Table2_Estimator_Performance.csv"), index=False)
        print("\n" + "="*60)
        print("TABLE 2: Estimator Performance (MC)")
        print("="*60)
        print(table_2.to_string(index=False))
        print("="*60)

        print("[CHECKPOINT] Building Table 3...", flush=True)
        table_3 = build_table_3_mc_performance(mc, scenario_results)
        table_3.to_csv(os.path.join(BASE_OUTDIR, "Table3_MonteCarlo_Performance.csv"), index=False)
        print("\n" + "="*70)
        print("TABLE 3: Monte Carlo Performance Summary")
        print("="*70)
        print(table_3.to_string(index=False))
        print("="*70)

        print("[CHECKPOINT] Saving raw MC results...", flush=True)
        mc["results_df"].to_csv(
            os.path.join(APPENDIX_OUTDIR, "MC_Raw_Results.csv"), index=False
        )
        if isinstance(scenario_results, pd.DataFrame) and len(scenario_results) > 0:
            scenario_results.to_csv(
                os.path.join(APPENDIX_OUTDIR, "MC_Scenario_Sensitivity.csv"), index=False
            )
        print("[CHECKPOINT] All done.", flush=True)

        try:
            n_scenarios = scenario_results[scenario_results['Scenario'].str.contains('N=')]
            if len(n_scenarios) > 0:
                fig_a3, axes = plt.subplots(1, 2, figsize=(12, 5))
                for est_name, marker, color in [('Canonical TTE', 'o', 'steelblue'), ('SNMM', 's', 'forestgreen')]:
                    est_data = n_scenarios[n_scenarios['Estimator'] == est_name]
                    if len(est_data) > 0:
                        n_vals = est_data['Scenario'].str.extract(r'N=(\d+)')[0].astype(int)
                        biases = est_data['Bias'].values
                        rmses = est_data['RMSE'].values
                        axes[0].plot(n_vals, biases, marker=marker, color=color, label=est_name, linewidth=2, markersize=8)
                        axes[1].plot(n_vals, rmses, marker=marker, color=color, label=est_name, linewidth=2, markersize=8)
                axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[0].set_xlabel('Sample Size (N)'); axes[0].set_ylabel('Bias')
                axes[0].set_title('Bias vs Sample Size'); axes[0].legend(); axes[0].set_xscale('log')
                axes[1].set_xlabel('Sample Size (N)'); axes[1].set_ylabel('RMSE')
                axes[1].set_title('RMSE vs Sample Size'); axes[1].legend(); axes[1].set_xscale('log')
                fig_a3.suptitle('Figure A3: Sample Size Sensitivity Analysis', y=1.02)
                fig_a3.tight_layout()
                fig_a3.savefig(os.path.join(APPENDIX_OUTDIR, "FigureA3_SampleSize_Sensitivity.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_a3)
        except Exception as e:
            print(f"Warning: Could not generate Figure A3: {e}")

    else:
        print("\n[RUN_MC=False] Using single-dataset bootstrap for figures and tables...")
        boot = bootstrap_all(
            df, draws, dgp, B=B_BOOT, seed=2025,
            grace_len=GRACE, trunc=TRUNC
        )
        print("Bootstrap RDs:")
        for k, v in boot["rds"].items():
            print(f"  {k}: mean={v[0]:.4f}, 95% CI=({v[1]:.4f}, {v[2]:.4f})")

        curve_bands = boot["bands"]
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)
        plot_rows = [
            ("True ITT (twin-world)", "true_itt"),
            ("Canonical TTE", "tte"),
            ("SNMM (g-estimation)", "snmm")
        ]
        colors = {0: "C0", 1: "C1"}
        tvals = np.arange(1, T + 1)
        for ax, (title, key) in zip(axes, plot_rows):
            for z in [0, 1]:
                mean, lo, hi = curve_bands[key][z]
                ax.plot(tvals, mean, color=colors[z], label=f"Z={z}")
                ax.fill_between(tvals, lo, hi, color=colors[z], alpha=0.25, linewidth=0)
            ax.axvline(x=GRACE, color="gray", ls="--", lw=1)
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Time (periods)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Figure 1: ITT Cumulative Incidence by Encouragement Arm", y=0.98)
        fig.tight_layout(rect=[0, 0, 0.98, 0.95])
        fig.savefig(os.path.join(BASE_OUTDIR, "Figure1_ITT_Cumulative_Incidence.png"), dpi=300)
        plt.show()

        plot_true_vs_snmm_dynamic_boot(
            boot, dgp, max_dur=T,
            save_path=os.path.join(BASE_OUTDIR, "Figure2_Duration_Effects_True_vs_SNMM.png")
        )

        plot_final_RD_bar(
            boot, true_itt_rd,
            save_path=os.path.join(BASE_OUTDIR, "Figure3_Final_RiskDifference.png")
        )

        table_2 = build_table_2_performance(boot, true_itt_rd, snmm_result=snmm)
        table_2.to_csv(os.path.join(BASE_OUTDIR, "Table2_Estimator_Performance.csv"), index=False)
        print("\n" + "="*60)
        print("TABLE 2: Estimator Performance (Single Dataset)")
        print("="*60)
        print(table_2.to_string(index=False))
        print("="*60)

    print("\n" + "="*70)
    print("OUTPUT FILES SAVED:")
    print("="*70)
    print("\nMain Manuscript (3 figures + 3 tables):")
    print(f"  - {os.path.join(BASE_OUTDIR, 'Figure1_ITT_Cumulative_Incidence.png')}")
    print(f"  - {os.path.join(BASE_OUTDIR, 'Figure2_Duration_Effects_True_vs_SNMM.png')}")
    print(f"  - {os.path.join(BASE_OUTDIR, 'Figure3_Final_RiskDifference.png')}")
    print(f"  - {os.path.join(BASE_OUTDIR, 'Table1_DGP_Summary.csv')}")
    print(f"  - {os.path.join(BASE_OUTDIR, 'Table2_Estimator_Performance.csv')}")
    print(f"  - {os.path.join(BASE_OUTDIR, 'Table3_MonteCarlo_Performance.csv')}")
    print("\nAppendix (5 tables + 3 figures):")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'TableA1_Full_DGP_Parameters.csv')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'TableA2_Detailed_Summary.csv')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'TableA3_IPCW_Diagnostics.csv')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'TableA4_Attrition_Patterns.csv')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'TableA5_Covariate_Balance.csv')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'FigureA1_Duration_Distributions.png')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'FigureA2_IPCW_Weight_Distributions.png')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'FigureA3_SampleSize_Sensitivity.png')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'MC_Raw_Results.csv')}")
    print(f"  - {os.path.join(APPENDIX_OUTDIR, 'MC_Scenario_Sensitivity.csv')}")
