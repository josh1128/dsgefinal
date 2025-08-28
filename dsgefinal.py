# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Real rate uses lag 1: RR_{t-1} = i_{t-1} − π_{t-1}
#      - Taylor uses inflation gap (π_t − π*)
#      - Shocks: IS, Phillips, Taylor + (NEW) RR, Foreign Demand, Energy, Non-Energy
#      - Policy shock behavior selector
#      - LaTeX equations shown below charts
#      - Monetary transmission floor on IS real-rate coefficient (optional)
#      - (NEW) Neutral rate slider used for Original long-run target & NK level plotting
#      - (NEW) Export plotted data to Excel on desktop
#   2) Simple NK (built-in)
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ===== Desktop export folder =====
file_path = r'C:\Users\AC03537\Desktop
# =========================
# Page setup
# =========================
st.set_page_config(page_title="DSGE IRF Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — IS, Phillips, Taylor")

st.markdown(
    "- **Units (Original model plots):** Real GDP growth **DlogGDP in %**, Inflation **DlogCPI in %**, "
    "and the **Taylor policy rate in decimals**.\n"
    "- **Taylor** uses the **inflation gap** \\((\\pi_t - \\pi^*)\\)."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s/100.0 if np.nanmedian(np.abs(s.values)) > 1.0 else s

def fmt_coef(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"; return f"+{s}" if x >= 0 else s

def build_latex_equation(const_val: float, terms: List[tuple], lhs: str, eps_symbol: str) -> str:
    rhs_terms = "" if not terms else " ".join([f"{fmt_coef(c)}\\,{sym}" for (c, sym) in terms])
    return rf"""\begin{{aligned}} {lhs} &= {const_val:.3f} {rhs_terms} + {eps_symbol} \end{{aligned}}"""

def row_from_params(params_index: pd.Index, values: Dict[str, float]) -> pd.DataFrame:
    cols = list(params_index)
    row = {c: (1.0 if c == "const" else float(values.get(c, 0.0))) for c in cols}
    return pd.DataFrame([row], columns=cols)

def predict_with_params(row_df: pd.DataFrame, beta: pd.Series) -> float:
    x = row_df[beta.index].iloc[0].values.astype(float)
    b = beta.values.astype(float)
    return float(np.dot(x, b))

# =========================
# Simple NK (built-in)
# =========================
@dataclass
class NKParamsSimple:
    sigma: float = 1.00
    kappa: float = 0.10
    phi_pi: float = 1.50
    phi_x: float = 0.125
    rho_i: float = 0.80
    rho_x: float = 0.50
    rho_r: float = 0.80
    rho_u: float = 0.50
    gamma_pi: float = 0.50

class SimpleNK3EqBuiltIn:
    def __init__(self, params: Optional[NKParamsSimple] = None):
        self.p = params or NKParamsSimple()

    def irf(self, shock="demand", T=24, size_pp=1.0, t0=0, rho_override=None):
        p = self.p
        x = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
        r_nat = np.zeros(T); u = np.zeros(T); e_i = np.zeros(T)

        if shock == "demand":
            r_nat[t0] = size_pp; rho_sh = p.rho_r if rho_override is None else rho_override
        elif shock == "cost":
            u[t0] = size_pp; rho_sh = p.rho_u if rho_override is None else rho_override
        elif shock == "policy":
            e_i[t0] = size_pp; rho_sh = None
        else:
            raise ValueError("shock must be 'demand','cost','policy'")

        for t in range(T):
            if t > t0:
                if shock == "demand": r_nat[t] += (rho_sh or 0.0) * r_nat[t-1]
                elif shock == "cost": u[t] += (rho_sh or 0.0) * u[t-1]

            x_lag = x[t-1] if t>0 else 0.0
            pi_lag = pi[t-1] if t>0 else 0.0
            i_lag = i[t-1] if t>0 else 0.0

            A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
            B_const = (
                p.rho_i * i_lag
                + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                + e_i[t]
            )
            denom = 1.0 + (A_x / p.sigma)
            num = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t] = num / max(denom, 1e-8)
            pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

        return np.arange(T), x, pi, i

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox("Choose model version", ["Original (DSGE.xlsx)", "Simple NK (built-in)"], index=0)

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    neutral_rate_pct = st.number_input(
        "Neutral nominal policy rate — % annual",
        value=2.00, step=0.25, format="%.2f",
        help="Used for Original (long-run target) and NK (level plotting)."
    )

    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original",
                               help="If omitted, the app looks for 'DSGE.xlsx' next to this script.")
        fallback = Path(__file__).parent / "DSGE.xlsx"

        rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

        st.header("Inflation target for Taylor")
        use_sample_mean = st.checkbox("Use sample mean of DlogCPI as target π*", value=False)
        if use_sample_mean:
            target_annual_pct = None
            st.caption("π* will be set to sample mean (quarterly) after data loads.")
        else:
            target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)
        st.divider()

        st.header("Core shocks")
        shock_target = st.selectbox(
            "Apply shock to",
            ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
            index=0,
        )
        is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f")
        policy_shock_bp_abs = st.number_input("Policy shock size (absolute bp)", value=100, step=5, format="%d")
        shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1, 1)
        shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

        st.subheader("Additional shocks (optional)")
        rr_shock_bp_abs = st.number_input("Real interest rate shock (bp, applied to RR_{t-1})", value=0, step=5, format="%d")
        fd_shock_pp     = st.number_input("Foreign demand shock (pp, enters Dlog FD_Lag1 with lag)", value=0.00, step=0.05, format="%.2f")
        energy_shock_pp = st.number_input("Energy shock (pp)", value=0.00, step=0.05, format="%.2f")
        nonenergy_shock_pp = st.number_input("Non-energy shock (pp)", value=0.00, step=0.05, format="%.2f")
        st.caption("Note: RR shock affects GDP at t+1 since IS uses RR_{t-1}. FD shock also appears with a lag.")

        st.header("Policy shock behavior")
        policy_mode = st.radio(
            "How should the policy shock enter?",
            ["Add after smoothing (standard)", "Add to target (inside 1−ρ)", "Force local jump (override)"],
            index=0
        )

        st.divider()
        st.header("Variable selection (include/exclude)")

        IS_ALL = ["DlogGDP_L1", "Real_Rate_L1_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"]
        PC_ALL = ["Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]
        TR_ALL = ["Nominal_Rate_L1", "Inflation_Gap", "DlogGDP"]

        with st.expander("IS Curve regressors", expanded=True):
            is_selected = st.multiselect("Use these variables in the IS regression:", IS_ALL, default=IS_ALL, key="is_vars")
        with st.expander("Phillips Curve regressors", expanded=True):
            pc_selected = st.multiselect("Use these variables in the Phillips regression:", PC_ALL, default=PC_ALL, key="pc_vars")
        with st.expander("Taylor Rule regressors", expanded=True):
            tr_selected = st.multiselect("Use these variables in the Taylor (partial adjustment) regression:", TR_ALL, default=TR_ALL, key="tr_vars")

        st.divider()
        mon_pass_floor_pp = st.slider(
            "Min Δg from −100 bp (pp)",
            0.00, 0.50, 0.05, 0.01,
            help="Floor on policy→activity: a 100 bp rate cut must raise quarterly GDP growth by at least this many **percentage points** (shows mainly at t+1 because IS uses RR_{t-1})."
        )

        st.divider()
        st.header("Export")
        export_filename = st.text_input("Excel filename (saved to Desktop)", value="dsge_export.xlsx")

    else:
        st.info("**Mapping:** IS→(σ, ρx, ρr) • Phillips→(κ, γπ, ρu) • Taylor→(φπ, φx, ρi)")
        st.header("Simple NK parameters (pp units)")
        st.subheader("IS")
        sigma = st.slider("σ", 0.2, 5.0, 1.00, 0.05)
        rho_x = st.slider("ρx", 0.0, 0.98, 0.50, 0.02)
        rho_r = st.slider("ρr", 0.0, 0.98, 0.80, 0.02)
        st.subheader("Phillips")
        kappa = st.slider("κ", 0.01, 0.50, 0.10, 0.01)
        gamma_pi = st.slider("γπ", 0.0, 0.95, 0.50, 0.05)
        rho_u = st.slider("ρu", 0.0, 0.98, 0.50, 0.02)
        st.subheader("Taylor")
        phi_pi = st.slider("φπ", 1.0, 3.0, 1.50, 0.05)
        phi_x = st.slider("φx", 0.00, 1.00, 0.125, 0.005)
        rho_i = st.slider("ρi", 0.0, 0.98, 0.80, 0.02)
        st.divider()
        st.header("Shock (IRF mode)")
        shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
        shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
        shock_quarter_nk = st.slider("Shock timing t", 1, T-1, 1, 1)
        shock_persist_nk = st.slider("Shock persistence ρ_shock", 0.0, 0.98, 0.80, 0.02)
        units_mode = st.radio("NK policy rate units", ["Deviation (pp)", "Level (decimal)"], index=1)
        st.divider()
        export_filename = st.text_input("Excel filename (saved to Desktop)", value="nk_export.xlsx")

# =========================
# ORIGINAL MODEL (DSGE.xlsx)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE.xlsx or place it beside this script.")
    excel_src = file_like_or_path
    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    for d in (is_df, pc_df, tr_df):
        d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m", errors="raise")

    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])
    df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"] = df["Nominal Rate"].shift(1)
    # ---- RR lag 1 (not lag 2) ----
    df["Real_Rate_L1_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(1)

    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L1_data",
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing: raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty: raise ValueError("No rows remain after dropping NA for required columns. Check your data.")
    return df, df_est

def _clip_and_recentre(beta_raw: pd.Series, means: Dict[str, float], rules: Dict[str, str]) -> pd.Series:
    beta_new = beta_raw.copy()
    for var, rule in rules.items():
        if var in beta_new.index:
            if rule == "nonpos": beta_new[var] = min(float(beta_new[var]), 0.0)
            elif rule == "nonneg": beta_new[var] = max(float(beta_new[var]), 0.0)
    const = float(beta_raw.get("const", 0.0))
    shift = 0.0
    for var, rule in rules.items():
        if var in beta_raw.index:
            shift += (float(beta_raw[var]) - float(beta_new[var])) * float(means.get(var, 0.0))
    beta_new["const"] = const + shift
    return beta_new

def fit_models_original(df_est: pd.DataFrame, pi_star_quarterly: float,
                        is_selected: List[str], pc_selected: List[str], tr_selected: List[str],
                        mon_pass_floor_pp: float):
    # ---------- IS ----------
    if not is_selected:
        raise ValueError("Select at least one regressor for IS (besides constant).")
    X_is = sm.add_constant(df_est[is_selected], has_constant="add")
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()
    X_is_means = {c: float(X_is[c].mean()) for c in X_is.columns if c != "const"}
    beta_is_sim = _clip_and_recentre(model_is.params, X_is_means, rules={"Real_Rate_L1_data": "nonpos"})

    # Pass-through floor: 100bp cut → ≥ mon_pass_floor_pp pp ↑ in growth
    floor_abs = (mon_pass_floor_pp / 100.0) / 0.01  # pp→decimal / 0.01
    if "Real_Rate_L1_data" in beta_is_sim.index:
        current = float(beta_is_sim["Real_Rate_L1_data"])
        desired = -max(abs(current), floor_abs)
        if desired != current:
            mean_rr = float(X_is_means.get("Real_Rate_L1_data", 0.0))
            beta_is_sim["const"] = float(beta_is_sim.get("const", 0.0)) + (current - desired) * mean_rr
            beta_is_sim["Real_Rate_L1_data"] = desired

    # ---------- Phillips ----------
    if not pc_selected:
        raise ValueError("Select at least one regressor for Phillips (besides constant).")
    X_pc = sm.add_constant(df_est[pc_selected], has_constant="add")
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()
    X_pc_means = {c: float(X_pc[c].mean()) for c in X_pc.columns if c != "const"}
    beta_pc_sim = _clip_and_recentre(model_pc.params, X_pc_means, rules={"DlogGDP_L1": "nonneg"})

    # ---------- Taylor (partial adjustment with inflation gap) ----------
    infl_gap_full = df_est["Dlog_CPI"] - pi_star_quarterly
    df_tr = pd.DataFrame(index=df_est.index)
    if "Nominal_Rate_L1" in tr_selected: df_tr["Nominal_Rate_L1"] = df_est["Nominal_Rate_L1"]
    if "Inflation_Gap" in tr_selected:   df_tr["Inflation_Gap"]   = infl_gap_full
    if "DlogGDP" in tr_selected:         df_tr["DlogGDP"]         = df_est["DlogGDP"]
    if df_tr.empty: raise ValueError("Select at least one regressor for Taylor (besides constant).")

    X_tr = sm.add_constant(df_tr, has_constant="add")
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    b0 = float(model_tr.params.get("const", 0.0))
    rhoh = min(max(float(model_tr.params.get("Nominal_Rate_L1", 0.0)), 0.0), 0.99)
    def safe_div(num, den): return num / den if abs(den) > 1e-8 else np.nan
    alpha_star = safe_div(b0, (1 - rhoh))
    bpi = float(model_tr.params.get("Inflation_Gap", 0.0))
    bg  = float(model_tr.params.get("DlogGDP", 0.0))
    phi_pi_star_raw = safe_div(bpi, (1 - rhoh)) if "Inflation_Gap" in model_tr.params.index else np.nan
    phi_g_star_raw  = safe_div(bg,  (1 - rhoh)) if "DlogGDP"      in model_tr.params.index else np.nan
    phi_pi_star_sim = np.nan if np.isnan(phi_pi_star_raw) else max(0.0, phi_pi_star_raw)
    phi_g_star_sim  = np.nan if np.isnan(phi_g_star_raw)  else max(0.0,  phi_g_star_raw)

    p_ss = float(df_est["Dlog_CPI"].mean())
    g_ss = float(df_est["DlogGDP"].mean())

    return {
        "model_is": model_is, "model_pc": model_pc, "model_tr": model_tr,
        "beta_is_sim": beta_is_sim, "beta_pc_sim": beta_pc_sim,
        "alpha_star": alpha_star, "rho_hat": rhoh,
        "phi_pi_star_sim": phi_pi_star_sim, "phi_g_star_sim": phi_g_star_sim,
        "pi_star_quarterly": float(pi_star_quarterly),
        "p_ss": p_ss, "g_ss": g_ss
    }

def build_shocks_original(
    T: int, target: str,
    is_size_pp: float, pc_size_pp: float, policy_bp_abs: int,
    rr_bp_abs: int, fd_pp: float, energy_pp: float, nonenergy_pp: float,
    t0: int, rho: float
):
    """Return arrays for all shocks. Units: pp inputs are converted to decimals for model internals; bp to decimal."""
    is_arr = np.zeros(T); pc_arr = np.zeros(T); pol_arr = np.zeros(T)
    rr_arr = np.zeros(T); fd_arr = np.zeros(T); en_arr = np.zeros(T); nen_arr = np.zeros(T)

    # Core shocks
    target_norm = (target or "None").strip().lower()
    if target_norm == "is (demand)".lower():
        is_arr[t0] = is_size_pp / 100.0
    elif target_norm == "phillips (supply)".lower():
        pc_arr[t0] = pc_size_pp / 100.0
    elif target_norm == "taylor (policy tightening)".lower():
        pol_arr[t0] =  (policy_bp_abs / 10000.0)
    elif target_norm == "taylor (policy easing)".lower():
        pol_arr[t0] = -(policy_bp_abs / 10000.0)

    # Additional shocks
    if rr_bp_abs != 0:
        rr_arr[t0] = rr_bp_abs / 10000.0  # decimal
    if fd_pp != 0.0:
        fd_arr[t0] = fd_pp / 100.0        # decimal
    if energy_pp != 0.0:
        en_arr[t0] = energy_pp / 100.0    # decimal
    if nonenergy_pp != 0.0:
        nen_arr[t0] = nonenergy_pp / 100.0

    # Persistence
    for arr in (is_arr, pc_arr, pol_arr, rr_arr, fd_arr, en_arr, nen_arr):
        for k in range(t0 + 1, T):
            arr[k] = rho * arr[k - 1]

    return is_arr, pc_arr, pol_arr, rr_arr, fd_arr, en_arr, nen_arr

def simulate_original(
    T: int, rho_sim: float, df_est: pd.DataFrame,
    models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float], i_mean_dec: float, real_rate_mean_dec: float,
    pi_star_quarterly: float,
    is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None,
    rr_shock_arr=None, fd_shock_arr=None, energy_shock_arr=None, nonenergy_shock_arr=None,
    policy_mode: str = "Add after smoothing (standard)", neutral_dec: float = 0.02
):
    """Forward-simulate g (DlogGDP), p (DlogCPI), i (policy rate). Internals: g,p in decimals; i in decimal."""
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]
    beta_is_sim = models["beta_is_sim"]; beta_pc_sim = models["beta_pc_sim"]
    phi_pi_star_sim = models.get("phi_pi_star_sim", np.nan)
    phi_g_star_sim  = models.get("phi_g_star_sim",  np.nan)
    p_ss = models["p_ss"]; g_ss = models["g_ss"]

    # Default zeros
    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)
    if rr_shock_arr is None: rr_shock_arr = np.zeros(T)
    if fd_shock_arr is None: fd_shock_arr = np.zeros(T)
    if energy_shock_arr is None: energy_shock_arr = np.zeros(T)
    if nonenergy_shock_arr is None: nonenergy_shock_arr = np.zeros(T)

    # Intercept so long-run equals chosen neutral rate
    alpha_star_sim = neutral_dec - (0.0 if np.isnan(phi_pi_star_sim) else phi_pi_star_sim) * (p_ss - pi_star_quarterly)

    for t in range(1, T):
        # Real rate with 1-quarter lag (decimal) + optional shock (applied to RR_{t-1})
        rr_extra = rr_shock_arr[t - 1] if t >= 1 else 0.0
        rr_lag1 = (i[t - 1] - p[t - 1]) + rr_extra

        # Build exogenous terms (means + shocks; respect lags where applicable)
        fd_val   = means["Dlog FD_Lag1"] + (fd_shock_arr[t - 1] if t >= 1 else 0.0)     # enters as _Lag1
        enr_val  = means["Dlog_Energy"] + energy_shock_arr[t]                            # contemporaneous
        nren_val = means["Dlog_NonEnergy"] + nonenergy_shock_arr[t]                      # contemporaneous

        enr_l1_val  = means["Dlog_Energy_L1"] + (energy_shock_arr[t - 1] if t >= 1 else 0.0)
        nren_l1_val = means["Dlog_Non_Energy_L1"] + (nonenergy_shock_arr[t - 1] if t >= 1 else 0.0)

        # IS (g_t)
        vals_is = {
            "DlogGDP_L1": g[t - 1],
            "Real_Rate_L1_data": rr_lag1,
            "Dlog FD_Lag1": fd_val,
            "Dlog_REER": means["Dlog_REER"],
            "Dlog_Energy": enr_val,
            "Dlog_NonEnergy": nren_val,
        }
        Xis = row_from_params(model_is.params.index, vals_is)
        g[t] = predict_with_params(Xis, beta_is_sim) + is_shock_arr[t]

        # Phillips (p_t)
        vals_pc = {
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": enr_l1_val,
            "Dlog_Non_Energy_L1": nren_l1_val,
        }
        Xpc = row_from_params(model_pc.params.index, vals_pc)
        p[t] = predict_with_params(Xpc, beta_pc_sim) + pc_shock_arr[t]

        # Taylor target and partial adjustment (i_t)
        pi_gap_t = p[t] - pi_star_quarterly
        g_dev_t  = g[t] - g_ss
        i_star = (alpha_star_sim
                  + (0.0 if np.isnan(phi_pi_star_sim) else phi_pi_star_sim) * pi_gap_t
                  + (0.0 if np.isnan(phi_g_star_sim)  else phi_g_star_sim)  * g_dev_t)

        eps = policy_shock_arr[t]  # decimal (e.g., 0.01 = 100 bp)
        if policy_mode.startswith("Add after"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * (i_star + eps)
        else:  # local-jump override (guard a minimum jump)
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
            if eps > 0: i_raw = max(i_raw, i[t - 1] + abs(eps))
            elif eps < 0: i_raw = min(i_raw, i[t - 1] - abs(eps))
        i[t] = float(i_raw)

    return g, p, i

# =========================
# Run selected model
# =========================
try:
    export_ready = False
    export_df = None
    shocks_df = None

    if model_choice == "Original (DSGE.xlsx)":
        file_source = xlf if 'xlf' in locals() and xlf is not None else (fallback if 'fallback' in locals() else None)
        df_all, df_est = load_and_prepare_original(file_source)

        # Determine π* (quarterly decimal)
        if 'use_sample_mean' in locals() and use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* set to sample mean of DlogCPI: {pi_star_quarterly:.4f} (quarterly decimal)")
        else:
            annual_pct = target_annual_pct if 'target_annual_pct' in locals() and target_annual_pct is not None else 2.0
            pi_star_quarterly = (annual_pct / 100.0) / 4.0
            st.info(f"π* set to {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

        # Fit with selected regressors (with pass-through floor)
        models_o = fit_models_original(df_est, pi_star_quarterly, is_selected, pc_selected, tr_selected, mon_pass_floor_pp)

        # Warn if policy shock but Real_Rate_L1_data is excluded (kills transmission)
        if isinstance(shock_target, str) and "taylor" in shock_target.lower() and "Real_Rate_L1_data" not in is_selected:
            st.warning("Policy shock selected but **Real_Rate_L1_data** is not in the IS regressors — "
                       "add it back to allow policy to affect GDP.")

        # Anchors & means
        i_mean_dec = float(df_est["Nominal Rate"].mean())
        real_rate_mean_dec = float(df_est["Real_Rate_L1_data"].mean())
        means_o = {
            "Dlog FD_Lag1": float(df_est["Dlog FD_Lag1"].mean()),
            "Dlog_REER": float(df_est["Dlog_REER"].mean()),
            "Dlog_Energy": float(df_est["Dlog_Energy"].mean()),
            "Dlog_NonEnergy": float(df_est["Dlog_NonEnergy"].mean()),
            "Dlog_Reer_L2": float(df_est["Dlog_Reer_L2"].mean()),
            "Dlog_Energy_L1": float(df_est["Dlog_Energy_L1"].mean()),
            "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()),
        }

        # Build shocks & simulate
        t0 = shock_quarter  # same indexing as your plotting line
        is_arr, pc_arr, pol_arr, rr_arr, fd_arr, en_arr, nen_arr = build_shocks_original(
            T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs,
            rr_shock_bp_abs, fd_shock_pp, energy_shock_pp, nonenergy_shock_pp,
            t0, shock_persist
        )
        neutral_dec = neutral_rate_pct / 100.0

        g0, p0, i0 = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            policy_mode=policy_mode, neutral_dec=neutral_dec
        )
        gS, pS, iS = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr,
            rr_shock_arr=rr_arr, fd_shock_arr=fd_arr, energy_shock_arr=en_arr, nonenergy_shock_arr=nen_arr,
            policy_mode=policy_mode, neutral_dec=neutral_dec
        )

        # ===== Plot (LEVELS): g & p in %, i in decimal =====
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        quarters = np.arange(T); vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(quarters, g0*100, label="Baseline", linewidth=2)
        axes[0].plot(quarters, gS*100, label="Shock", linewidth=2)
        axes[0].axvline(shock_quarter, **vline_kwargs)
        axes[0].set_title("Real GDP Growth (DlogGDP, %)"); axes[0].set_ylabel("%")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(quarters, p0*100, label="Baseline", linewidth=2)
        axes[1].plot(quarters, pS*100, label="Shock", linewidth=2)
        axes[1].axvline(shock_quarter, **vline_kwargs)
        axes[1].set_title("Inflation (DlogCPI, %)"); axes[1].set_ylabel("%")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, iS, label="Shock", linewidth=2)
        axes[2].axvline(shock_quarter, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (decimal)")
        axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("decimal")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Readout
        if isinstance(shock_target, str) and "taylor" in shock_target.lower():
            delta_i_bp = (iS - i0)[shock_quarter] * 10000.0
            st.info(f"Δ policy rate at t={shock_quarter}: {delta_i_bp:.1f} bp  |  mode: {policy_mode}  |  ρ={rho_sim:.2f}")
            st.caption("Policy affects GDP at **t+1** (IS uses RR_{t-1}).")

        # ===== Prepare export data =====
        export_df = pd.DataFrame({
            "quarter": quarters,
            "g_baseline_pct": g0 * 100.0,
            "g_shock_pct":    gS * 100.0,
            "p_baseline_pct": p0 * 100.0,
            "p_shock_pct":    pS * 100.0,
            "i_baseline_dec": i0,
            "i_shock_dec":    iS,
        })
        shocks_df = pd.DataFrame({
            "quarter": quarters,
            "is_shock_dec": is_arr,
            "pc_shock_dec": pc_arr,
            "policy_shock_dec": pol_arr,
            "rr_shock_dec": rr_arr,
            "fd_shock_dec": fd_arr,
            "energy_shock_dec": en_arr,
            "nonenergy_shock_dec": nen_arr,
        })
        export_ready = True

        # ===== LaTeX equations (display raw OLS) =====
        st.subheader("Estimated Equations (Original model)")
        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]; rho_hat = models_o["rho_hat"]
        phi_pi_star = models_o["phi_pi_star_sim"]; phi_g_star = models_o["phi_g_star_sim"]

        # IS
        is_terms = []
        pretty_map_is = {
            "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
            "Real_Rate_L1_data": r"RR_{t-1}",
            "Dlog FD_Lag1": r"\Delta \log FD_{t-1}",
            "Dlog_REER": r"\Delta \log REER_t",
            "Dlog_Energy": r"\Delta \log Energy_t",
            "Dlog_NonEnergy": r"\Delta \log NonEnergy_t",
        }
        for k, v in m_is.params.items():
            if k == "const": continue
            is_terms.append((float(v), pretty_map_is.get(k, k)))
        st.markdown("**IS Curve (\\(\\Delta \\log GDP_t\\))**")
        st.latex(build_latex_equation(float(m_is.params.get("const", 0.0)), is_terms, r"\Delta \log GDP_t", r"\varepsilon_t"))

        # Phillips
        pc_terms = []
        pretty_map_pc = {
            "Dlog_CPI_L1": r"\Delta \log CPI_{t-1}",
            "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
            "Dlog_Reer_L2": r"\Delta \log REER_{t-2}",
            "Dlog_Energy_L1": r"\Delta \log Energy_{t-1}",
            "Dlog_Non_Energy_L1": r"\Delta \log NonEnergy_{t-1}",
        }
        for k, v in m_pc.params.items():
            if k == "const": continue
            pc_terms.append((float(v), pretty_map_pc.get(k, k)))
        st.markdown("**Phillips Curve (\\(\\Delta \\log CPI_t\\))**")
        st.latex(build_latex_equation(float(m_pc.params.get("const", 0.0)), pc_terms, r"\Delta \log CPI_t", r"u_t"))

        # Taylor rule
        st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
        if policy_mode.startswith("Add after"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t")
        elif policy_mode.startswith("Add to target"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\,\big(i_t^\* + \varepsilon^{\text{pol}}_t\big)")
        else:
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t \quad (\text{with local-jump override})")

        st.latex(r"i_t^\* \;=\; \alpha^\*_{\text{sim}} \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,\big(g_t - \bar g\big)")
        st.caption("Clips enforce: IS real-rate ≤ 0 (with floor), Phillips activity ≥ 0, Taylor φ’s ≥ 0; α* set so long-run equals the neutral rate.")

    else:
        # =========================
        # Simple NK (built-in)
        # =========================
        P = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x, rho_i=rho_i,
            rho_x=rho_x, rho_r=rho_r, rho_u=rho_u, gamma_pi=gamma_pi
        )
        model = SimpleNK3EqBuiltIn(P)
        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))
        rho_for_shock = shock_persist_nk

        st.subheader("Impulse responses")
        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, rho_for_shock)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, rho_for_shock)

        # Plot: keep x, π in pp; show rate in **decimal** when in level mode
        neutral_dec = neutral_rate_pct / 100.0
        if units_mode == "Level (decimal)":
            i0_plot = neutral_dec + (i0 / 100.0)     # convert pp to decimal
            iS_plot = neutral_dec + (iS / 100.0)
            i_ylabel = "decimal"
            i_title = "Nominal Policy Rate (level, decimal)"
        else:
            i0_plot = i0
            iS_plot = iS
            i_ylabel = "pp"
            i_title = "Nominal Policy Rate (deviation, pp)"

        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(h, x0, linewidth=2, label="Baseline")
        axes[0].plot(h, xS, linewidth=2, label="Shock")
        axes[0].axvline(t0, **vline_kwargs); axes[0].set_title("Output Gap (x_t, pp)"); axes[0].set_ylabel("pp")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(h, pi0, linewidth=2, label="Baseline")
        axes[1].plot(h, piS, linewidth=2, label="Shock")
        axes[1].axvline(t0, **vline_kwargs); axes[1].set_title("Inflation (π_t, pp)"); axes[1].set_ylabel("pp")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(h, i0_plot, linewidth=2, label="Baseline")
        axes[2].plot(h, iS_plot, linewidth=2, label="Shock")
        axes[2].axvline(t0, **vline_kwargs)
        axes[2].set_title(i_title)
        axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel(i_ylabel)
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Export for NK
        export_df = pd.DataFrame({
            "quarter": h,
            "x_baseline_pp": x0, "x_shock_pp": xS,
            "pi_baseline_pp": pi0, "pi_shock_pp": piS,
            "i_baseline_plot": i0_plot, "i_shock_plot": iS_plot,
            "i_baseline_dev_pp": i0, "i_shock_dev_pp": iS,
        })
        shocks_df = pd.DataFrame({
            "shock_type": [shock_type_nk],
            "shock_size_pp": [shock_size_pp_nk],
            "shock_t0": [t0],
            "shock_persistence": [rho_for_shock],
        })
        export_ready = True

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()

# =========================
# Export button
# =========================
if export_ready and export_df is not None and export_filename:
    export_fullpath = os.path.join(file_path, export_filename)
    if st.button("Export current chart data to Excel"):
        try:
            with pd.ExcelWriter(export_fullpath, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, sheet_name="series", index=False)
                if shocks_df is not None:
                    shocks_df.to_excel(writer, sheet_name="shocks", index=False)
            st.success(f"Exported to: {export_fullpath}")
        except Exception as ex:
            st.error(f"Failed to export to Excel: {ex}")

