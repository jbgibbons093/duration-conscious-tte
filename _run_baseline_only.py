"""Baseline-only Monte Carlo run."""
import os, importlib.util, numpy as np

HERE = os.getcwd()
spec = importlib.util.spec_from_file_location('sdt', os.path.join(HERE, 'run_simulation.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE, APP = m.BASE_OUTDIR, m.APPENDIX_OUTDIR

print(f"[BASELINE] N={m.N} T={m.T} G={m.GRACE} M={m.M_SIMS} B_MC_BOOT={m.B_MC_BOOT}", flush=True)
df, draws, dgp = m.simulate_trial_data(N=m.N, T=m.T, grace_len=m.GRACE,
                                       conf_strength=m.CONF_STRENGTH, seed=m.SEED)
true_rd, _, _ = m.compute_true_itt(draws, dgp)
print(f"[BASELINE] True ITT RD = {true_rd:.4f}", flush=True)

m.build_table_1_dgp_summary(df, draws, dgp).to_csv(
    os.path.join(BASE, "Table1_DGP_Summary.csv"), index=False)
print("[CHECKPOINT] Table 1 written.", flush=True)

print("[CHECKPOINT] Starting baseline MC...", flush=True)
mc = m.run_monte_carlo(M=m.M_SIMS, N=m.N, T=m.T, grace_len=m.GRACE,
                       conf_strength=m.CONF_STRENGTH, base_seed=m.SEED,
                       n_jobs=-1, verbose=True)
print(f"[CHECKPOINT] Baseline MC done: {len(mc['results_df'])} reps.", flush=True)

m.plot_figure1_mc(mc["mc_curves"], T_val=m.T, grace_len=m.GRACE,
                  save_path=os.path.join(BASE, "Figure1_ITT_Cumulative_Incidence.png"))
m.plot_figure2_mc(mc["mc_dyn"], dgp, max_dur=m.T,
                  save_path=os.path.join(BASE, "Figure2_Duration_Effects_True_vs_SNMM.png"))
m.plot_figure3_mc(mc, save_path=os.path.join(BASE, "Figure3_Final_RiskDifference.png"))
print("[CHECKPOINT] Figures 1-3 written.", flush=True)

t2 = m.build_table_2_mc_performance(mc["summary"], mc_result_df=mc["results_df"])
t2.to_csv(os.path.join(BASE, "Table2_Estimator_Performance.csv"), index=False)
m.build_table_3_mc_performance(mc, None).to_csv(
    os.path.join(BASE, "Table3_MonteCarlo_Performance.csv"), index=False)
mc["results_df"].to_csv(os.path.join(APP, "MC_Raw_Results.csv"), index=False)
print("[CHECKPOINT] Tables 2-3 + raw results written.", flush=True)

# Compact summary
s = mc["summary"]
lines = ["=== BASELINE MC SUMMARY ===", f"True ITT RD = {true_rd:.4f}", ""]
for key, lab in [("tte", "Canonical TTE"), ("adj_tte", "Adjusted TTE"), ("snmm", "SNMM")]:
    d = s[key]
    lines.append(f"{lab}:")
    lines.append(f"  bias={d.get('bias'):+.4f}  emp_se={d.get('emp_se'):.4f}  rmse={d.get('rmse'):.4f}")
    lines.append(f"  coverage(analytic)={d.get('coverage', float('nan')):.1%}  se_ratio={d.get('se_ratio', float('nan')):.3f}")
    if 'boot_coverage' in d:
        lines.append(f"  coverage(bootstrap)={d.get('boot_coverage'):.1%}  boot_se_ratio={d.get('boot_se_ratio', float('nan')):.3f}")
    if 'cov_ceiling' in d:
        lines.append(f"  cov_ceiling={d.get('cov_ceiling'):.1%}  bias_loss={d.get('bias_coverage_loss', float('nan')):+.1%}  var_loss(ana)={d.get('var_coverage_loss_analytic', float('nan')):+.1%}")
    lines.append("")
lines.append("=== Table 2 ===")
lines.append(t2.to_string(index=False))
out = "\n".join(lines)
with open(os.path.join(HERE, "_baseline_summary.txt"), "w", encoding="utf-8") as f:
    f.write(out + "\n")
print(out, flush=True)
print("[CHECKPOINT] All baseline outputs done.", flush=True)
