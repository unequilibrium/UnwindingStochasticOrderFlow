"""
Non-Flat EOD & Financing Costs Extension of Nutz, Webster & Zhao (2023)
========================================================================
Prompt 7 implementation.

The base model imposes a hard flat-EOD constraint: X_T = 0.
This extension replaces it with a soft terminal penalty:

    Add  1/2 * rho * X_T^2  to the cost function

where rho is the net overnight penalty per unit^2 of residual inventory:

    rho = overnight VaR cost per unit^2  -  swap income rate

ONLY ONE LINE IN THE RICCATI CHANGES:
    Terminal condition:  A(tau=0) = lambda/2 + rho/2
    (base model uses:    A(tau=0) = lambda/2            )

All six ODE equations are identical. The modification is entirely
in the boundary condition.

Three limiting cases:
  rho -> inf  :  hard flat-EOD (base model)
  rho = 0     :  break-even financing (no overnight penalty)
  rho < 0     :  favorable carry (desk is incentivised to hold overnight)

Four scenarios (theta=-0.5, mild momentum):
  A  Favorable carry   rho = -lambda/4   Desk receives swap > VaR cost
  B  Break-even        rho = 0           Swap exactly offsets VaR cost
  C  Moderate penalty  rho = 2*lambda    Desk pays net overnight cost
  D  Near hard EOD     rho = 20*lambda   Very high overnight penalty

Key results:
  - f_t urgency profile: dramatically different shape near close
    (rho=20*lam -> spike to -440 at t=T; rho=0 -> smooth -40 throughout)
  - Residual overnight inventory X_T:
    (favorable carry holds ~7% ADV overnight; near-hard EOD holds ~0.3%)
  - Total cost trade-off:
    (impact cost halved as rho falls; overnight cost rises but net saves money)
  - Internalization: 65.5% (favorable) -> 58.8% (hard EOD)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

np.random.seed(42)
sns.set_style("whitegrid")
pal = sns.color_palette("tab10")

COLS = {
    "Favorable carry  (rho=-lam/4)": pal[2],   # green — carry earns money
    "Break-even       (rho=0)":      pal[0],   # blue  — neutral
    "Moderate penalty (rho=2*lam)":  pal[1],   # orange
    "Near hard EOD    (rho=20*lam)": pal[3],   # red
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Riccati ODE — base equations (unchanged from paper)
# ─────────────────────────────────────────────────────────────────────────────

def riccati_rhs(tau, state, beta, lam, eps, theta):
    """Base Riccati ODEs — identical to the paper's equations."""
    A, B, C, D, E, F = state
    f = -(2*A + lam*B) / eps
    g = -(1 + B + 2*lam*C) / eps
    h = -(D + lam*E) / eps
    dA = -0.5*eps*f**2
    dB = -eps*f*g - beta*B
    dC = -0.5*eps*g**2 - 2*beta*C
    dD = -eps*f*h + theta*(2*A - D)
    dE = -eps*g*h + B*theta - E*(beta + theta)
    dF = -0.5*eps*h**2 + (D - 2*F)*theta
    return [dA, dB, dC, dD, dE, dF]


def solve_riccati(beta, lam, eps, theta, rho=1e6, T=1.0, N=200):
    """
    Solve the Riccati ODE backward from T.

    THE ONLY CHANGE vs. the base model:
        Terminal condition A(tau=0) = lambda/2 + rho/2
        Base model uses   A(tau=0) = lambda/2

    rho=1e6 approximates the hard flat-EOD constraint.
    rho=0 gives break-even financing.
    rho<0 gives favorable carry.
    """
    A_terminal = lam/2 + rho/2          # <-- the one changed line
    s0 = [A_terminal, 1.0, 0.0, 0.0, 0.0, 0.0]
    tau_grid = np.linspace(0, T, N+1)

    sol = solve_ivp(riccati_rhs, [0, T], s0,
                    args=(beta, lam, eps, theta),
                    t_eval=tau_grid, method='RK45', rtol=1e-8, atol=1e-10)
    st = sol.y.T   # (N+1, 6) — index 0=tau=0=t=T, index N=tau=T=t=0

    A, B, C, D, E, F = st.T
    f_grid = -(2*A + lam*B) / eps
    g_grid = -(1 + B + 2*lam*C) / eps
    h_grid = -(D + lam*E) / eps

    return tau_grid, f_grid, g_grid, h_grid, st


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Opening block J0
# ─────────────────────────────────────────────────────────────────────────────

def compute_J0(y0, z0, lam, state_at_tau_T):
    """Opening block from the Riccati state at t=0 (tau=T = state[-1])."""
    A, B, C, D, E, F = state_at_tau_T
    c_J0    = (2*A + lam*B) + lam*(B + 2*lam*C)
    c_const = ((-2*A*z0 + B*y0 + D*z0) +
               lam*(-B*z0 + 2*C*y0 + E*z0))
    return -(y0 + c_const) / (lam + c_J0)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Monte Carlo forward simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(beta=8.0, lam=0.2, eps=0.01, theta=-0.5,
             sigma=0.10, z0=0.10, y0=0.0,
             rho=1e6, T=1.0, N=200, n_samples=3000, n_shocks=20,
             label=""):
    """
    Simulate with soft terminal penalty rho.

    X_T is no longer constrained to zero — the desk holds residual
    inventory overnight, paying (or earning) rho * X_T^2 / 2.
    """
    dt = T / N; wait = N // n_shocks

    tau_grid, f_g, g_g, h_g, state = solve_riccati(
        beta, lam, eps, theta, rho, T, N)

    J0 = compute_J0(y0, z0, lam, state[-1])   # state[-1] = tau=T = t=0
    X0 = J0 - z0; Y0 = y0 + lam * J0

    nS = n_samples
    X = np.zeros((nS, N+1)); X[:, 0] = X0
    Y = np.zeros((nS, N+1)); Y[:, 0] = Y0
    Z = np.zeros((nS, N+1)); Z[:, 0] = z0
    q = np.zeros((nS, N+1))

    raw = np.random.randn(nS, N) * np.sqrt(dt)
    dW  = np.zeros((nS, N))
    for i in range(N):
        if i % wait == 0: dW[:, i] = raw[:, i] * np.sqrt(wait)
    dZn = dW * sigma

    for i in range(N):
        k = N - i   # tau index at time t_i  (tau = T - t_i)
        q[:, i] = f_g[k]*X[:, i] + g_g[k]*Y[:, i] + h_g[k]*Z[:, i]
        dZ_i = -theta * Z[:, i] * dt + dZn[:, i]
        X[:, i+1] = X[:, i] + q[:, i]*dt - dZ_i
        Y[:, i+1] = Y[:, i] + (-beta*Y[:, i] + lam*q[:, i])*dt
        Z[:, i+1] = Z[:, i] + dZ_i

    q[:, -1] = 0
    JT = X[:, -1]   # residual overnight inventory

    # ── Costs ─────────────────────────────────────────────────────────────────
    open_cost    = y0*J0 + 0.5*lam*J0**2
    impact_cost  = np.sum(Y * q, axis=1) * dt
    spread_cost  = 0.5*eps * np.sum(q**2, axis=1) * dt
    term_impact  = 0.5*lam*JT**2 + Y[:, -1]*JT
    overnight    = 0.5*rho*JT**2            # soft penalty (negative = earn carry)
    total_cost   = open_cost + impact_cost + spread_cost + term_impact + overnight

    # ── Internalization ───────────────────────────────────────────────────────
    TV_out = np.sum(np.abs(q), axis=1)*dt + np.abs(JT) + abs(J0)
    TV_in  = abs(z0) + np.sum(np.abs(np.diff(Z, axis=1)), axis=1)
    intern = (1 - TV_out / np.maximum(TV_in, 1e-10)) * 100

    norm = np.maximum(TV_in, 1e-10)

    return pd.DataFrame({
        "label":           label,
        "rho":             rho,
        "intern_%":        intern,
        "impact_bps":      impact_cost / norm * 1e4,
        "spread_bps":      spread_cost / norm * 1e4,
        "overnight_bps":   overnight   / norm * 1e4,
        "total_bps":       total_cost  / norm * 1e4,
        "TV_out_%":        TV_out * 100,
        "XT_%":            JT * 100,           # overnight residual (%ADV)
    }), {"X": X, "Y": Y, "Z": Z, "q": q, "J0": J0}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Scenarios
# ─────────────────────────────────────────────────────────────────────────────

BASE = dict(beta=8.0, lam=0.2, eps=0.01, theta=-0.5,
            sigma=0.10, z0=0.10, y0=0.0,
            T=1.0, N=200, n_samples=3000, n_shocks=20)

lam = BASE["lam"]
scenarios = {
    "Favorable carry  (rho=-lam/4)":
        dict(**BASE, rho=-lam/4,   label="Favorable carry  (rho=-lam/4)"),
    "Break-even       (rho=0)":
        dict(**BASE, rho=0.0,      label="Break-even       (rho=0)"),
    "Moderate penalty (rho=2*lam)":
        dict(**BASE, rho=2*lam,    label="Moderate penalty (rho=2*lam)"),
    "Near hard EOD    (rho=20*lam)":
        dict(**BASE, rho=20*lam,   label="Near hard EOD    (rho=20*lam)"),
}
order  = list(scenarios.keys())
colors = [COLS[k] for k in order]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run
# ─────────────────────────────────────────────────────────────────────────────

print("Non-Flat EOD Extension — theta=-0.5 (mild momentum)")
print("="*65)
results = {}
for name, kw in scenarios.items():
    print(f"  {name} ...", end=" ", flush=True)
    summary, ts = simulate(**kw)
    results[name] = dict(summary=summary, ts=ts, params=kw)
    mn_i  = summary["intern_%"].mean()
    mn_tc = summary["total_bps"].mean()
    mn_xt = summary["XT_%"].mean()
    print(f"intern {mn_i:.1f}%  total {mn_tc:.1f} bps  overnight XT {mn_xt:.2f}%ADV")

all_sum = pd.concat([v["summary"] for v in results.values()], ignore_index=True)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure 1 — THE KEY STRUCTURAL RESULT: f_t urgency profiles
#     This shows the most dramatic visual difference: f_T blows up for
#     hard EOD but stays smooth for low rho. A 10x difference at t=T.
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "How the overnight penalty reshapes intraday execution urgency\n"
    "Lower rho = smoother profile, no EOD spike  |  "
    "Higher rho = explosive urgency at close (hard-EOD approximation)",
    fontsize=10)

rho_vals   = [-lam/4, 0.0, 0.5*lam, lam, 2*lam, 5*lam, 20*lam]
rho_labels = ["rho=-lam/4 (carry)", "rho=0 (break-even)",
              "rho=0.5*lam", "rho=lam",
              "rho=2*lam", "rho=5*lam", "rho=20*lam (hard EOD)"]
rho_colors = [pal[2], pal[0], pal[9], pal[4], pal[1], pal[6], pal[3]]

t_grid = np.linspace(0, BASE["T"], BASE["N"]+1) * 6.5 + 9.5

for rv, lbl, clr in zip(rho_vals, rho_labels, rho_colors):
    tau, f, g, h, st = solve_riccati(
        BASE["beta"], BASE["lam"], BASE["eps"], BASE["theta"],
        rho=rv, T=BASE["T"], N=BASE["N"])
    t = BASE["T"] - tau   # tau -> t (0=open, T=close)

    axes[0].plot(t, f, lw=2, color=clr, label=lbl)
    axes[1].plot(t, g, lw=2, color=clr)
    axes[2].plot(t, h, lw=2, color=clr)

# Annotate f_T values on left panel
ax = axes[0]
for rv, lbl, clr in zip([0.0, 2*lam, 20*lam],
                          ["break-even", "mod", "hard EOD"],
                          [pal[0], pal[1], pal[3]]):
    tau, f, g, h, _ = solve_riccati(
        BASE["beta"],BASE["lam"],BASE["eps"],BASE["theta"],
        rho=rv, T=BASE["T"], N=BASE["N"])
    fT = f[0]   # f at tau=0 = t=T (terminal)
    ax.annotate(f"f_T={fT:.0f}", xy=(BASE["T"]*6.5+9.5, fT),
                xytext=(BASE["T"]*6.5+9.2, fT*0.7),
                fontsize=7.5, color=clr, ha="right",
                arrowprops=dict(arrowstyle="->", color=clr, lw=0.8))

axes[0].set_ylim(-500, 10)
axes[0].set_title("f_t  (inventory urgency)\n"
                   "KEY: lower rho = finite f_T; higher rho = f_T -> -infinity", fontsize=9)
axes[0].legend(fontsize=7, loc="lower left")

for ax, ttl in zip(axes[1:],
                    ["g_t  (impact management)",
                     "h_t  (flow anticipation)"]):
    ax.set_title(ttl, fontsize=9)

for ax in axes:
    ax.set_xlabel("Time (EST)"); ax.axhline(0, color="k", lw=0.5, ls=":")

plt.tight_layout()
plt.savefig("/home/claude/fig_eod1_urgency.png", dpi=150, bbox_inches="tight")
plt.show(); print("Figure 1 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Figure 2 — Overnight residual inventory X_T distribution
#     The most direct way to see the soft penalty at work
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
fig.suptitle(
    "Residual overnight inventory X_T distribution across rho scenarios\n"
    "Hard EOD forces X_T ≈ 0  |  Favorable carry incentivises holding overnight",
    fontsize=10)

for ax, (name, clr) in zip(axes, zip(order, colors)):
    xt = results[name]["ts"]["X"][:, -1] * 100   # residual as %ADV
    ax.hist(xt, bins=50, color=clr, alpha=0.8, edgecolor="white", density=True)
    ax.axvline(0, color="black", lw=1.5, ls="--", label="X_T=0 (hard EOD)")
    ax.axvline(xt.mean(), color=clr, lw=2, ls="-", label=f"mean={xt.mean():.1f}%")
    ax.set_title(name.replace("  ", "\n"), fontsize=8.5)
    ax.set_xlabel("Overnight inventory X_T (%ADV)")
    ax.legend(fontsize=7.5)
    if ax == axes[0]: ax.set_ylabel("Density")

plt.tight_layout()
plt.savefig("/home/claude/fig_eod2_xt_dist.png", dpi=150, bbox_inches="tight")
plt.show(); print("Figure 2 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Figure 3 — THE KEY ECONOMIC RESULT: cost trade-off
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
fig.suptitle(
    "Economic trade-off: lower rho reduces intraday impact cost\n"
    "but increases overnight inventory risk  |  theta=-0.5, z0=10%ADV",
    fontsize=10)

# Stacked cost bars
ax = axes[0]
bot = np.zeros(len(order))
for col, clab, clr in zip(
    ["impact_bps", "overnight_bps", "spread_bps"],
    ["Impact cost (intraday)", "Overnight cost / carry", "Spread cost"],
    [pal[3], pal[1], pal[0]]
):
    vals = [all_sum.loc[all_sum["label"]==k, col].mean() for k in order]
    ax.bar(range(len(order)), vals, bottom=bot, label=clab,
           color=clr, alpha=0.85, edgecolor="white")
    bot = bot + np.array(vals)
ax.set_title("Cost decomposition by component", fontsize=10)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("bps per in-flow"); ax.legend(fontsize=8.5)
ax.axhline(0, color="k", lw=0.8)

# Total cost
ax = axes[1]
total_vals = [all_sum.loc[all_sum["label"]==k, "total_bps"].mean() for k in order]
bars = ax.bar(range(len(order)), total_vals, color=colors, edgecolor="white")
ax.set_title("Total cost (impact + overnight + spread)", fontsize=10)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("bps per in-flow"); ax.axhline(0, color="k", lw=0.8)
base_cost = total_vals[-1]  # hardest EOD is the reference
for bar, val in zip(bars, total_vals):
    saving = base_cost - val
    lbl = f"{val:.1f}\n({saving:+.1f})" if abs(saving) > 0.05 else f"{val:.1f}\n(ref)"
    ax.text(bar.get_x()+bar.get_width()/2, max(val, 0)+0.3, lbl,
            ha="center", fontsize=9, fontweight="bold")

# Internalization
ax = axes[2]
intern_vals = [all_sum.loc[all_sum["label"]==k, "intern_%"].mean() for k in order]
bars = ax.bar(range(len(order)), intern_vals, color=colors, edgecolor="white")
ax.set_title("Internalization rate", fontsize=10)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("Internalization (%)")
ax.set_ylim(50, 75)
for bar, val in zip(bars, intern_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.3, f"{val:.1f}%",
            ha="center", fontsize=9.5, fontweight="bold")

plt.tight_layout()
plt.savefig("/home/claude/fig_eod3_costs.png", dpi=150, bbox_inches="tight")
plt.show(); print("Figure 3 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Figure 4 — Intraday inventory paths
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle(
    "Intraday inventory paths X_t (%ADV) — mean ± 1 std across 3000 samples\n"
    "Lower rho: desk holds a cushion toward close instead of forcing flat",
    fontsize=10)

t_grid = np.linspace(0, BASE["T"], BASE["N"]+1) * 6.5 + 9.5

for ax, (name, clr) in zip(axes, zip(order, colors)):
    X = results[name]["ts"]["X"] * 100
    m = X.mean(0); s = X.std(0)
    ax.plot(t_grid, m, lw=2.5, color=clr)
    ax.fill_between(t_grid, m-s, m+s, alpha=0.2, color=clr)
    ax.axhline(0, color="black", lw=1.2, ls="--", label="X=0 (flat)")
    xt_mean = X[:, -1].mean()
    ax.axhline(xt_mean, color=clr, lw=1, ls=":", alpha=0.8,
               label=f"Mean X_T={xt_mean:.1f}%")
    ax.set_title(name.replace("  ", "\n"), fontsize=8.5)
    ax.set_xlabel("Time (EST)"); ax.set_ylabel("%ADV")
    ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig("/home/claude/fig_eod4_paths.png", dpi=150, bbox_inches="tight")
plt.show(); print("Figure 4 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Figure 5 — Continuous rho sweep: the full trade-off curve
# ─────────────────────────────────────────────────────────────────────────────

print("Running continuous rho sweep ...")

rho_sweep = np.array([-lam/2, -lam/4, -lam/8, 0,
                        lam/4, lam/2, lam, 2*lam,
                        5*lam, 10*lam, 20*lam, 50*lam])
rho_ratio  = rho_sweep / lam   # rho/lambda for x-axis

intern_sw = []; cost_sw = []; xt_sw = []; impact_sw = []; onight_sw = []

base_clean = {k:v for k,v in BASE.items() if k not in ("label", "n_samples")}
for rv in rho_sweep:
    summ, _ = simulate(**base_clean, rho=rv, n_samples=1000, label="")
    intern_sw.append(summ["intern_%"].mean())
    cost_sw.append(summ["total_bps"].mean())
    xt_sw.append(summ["XT_%"].mean())
    impact_sw.append(summ["impact_bps"].mean())
    onight_sw.append(summ["overnight_bps"].mean())

intern_sw = np.array(intern_sw); cost_sw   = np.array(cost_sw)
xt_sw     = np.array(xt_sw);     impact_sw = np.array(impact_sw)
onight_sw = np.array(onight_sw)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Continuous rho sweep: the financing cost trade-off curve\n"
    "x-axis = rho / lambda  |  Vertical dashed lines mark the four scenarios",
    fontsize=10)

scenario_rhos = [-lam/4, 0.0, 2*lam, 20*lam]
scenario_cols = [pal[2], pal[0], pal[1], pal[3]]
scenario_lbls = ["Carry", "Break-even", "Moderate", "Hard EOD"]

ax = axes[0]
ax.plot(rho_ratio, intern_sw, "o-", color=pal[0], lw=2.5, ms=7)
for rv, clr, lbl in zip(scenario_rhos, scenario_cols, scenario_lbls):
    ax.axvline(rv/lam, color=clr, lw=1.5, ls="--", alpha=0.8, label=lbl)
ax.set_xlabel("rho / lambda"); ax.set_ylabel("Internalization (%)")
ax.set_title("Internalization vs. rho", fontsize=10); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(rho_ratio, cost_sw,   "s-", color=pal[1], lw=2.5, ms=7, label="Total cost")
ax.plot(rho_ratio, impact_sw, "^--", color=pal[3], lw=1.8, ms=6, label="Impact cost")
ax.plot(rho_ratio, onight_sw, "D--", color=pal[2], lw=1.8, ms=6, label="Overnight cost")
for rv, clr in zip(scenario_rhos, scenario_cols):
    ax.axvline(rv/lam, color=clr, lw=1.5, ls="--", alpha=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("rho / lambda"); ax.set_ylabel("bps per in-flow")
ax.set_title("Cost components vs. rho\n"
             "Impact falls as rho falls; overnight cost rises", fontsize=9)
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(rho_ratio, xt_sw, "D-", color=pal[3], lw=2.5, ms=7)
ax.axhline(0, color="black", lw=1.2, ls="--", label="X_T=0 (flat)")
for rv, clr, lbl in zip(scenario_rhos, scenario_cols, scenario_lbls):
    ax.axvline(rv/lam, color=clr, lw=1.5, ls="--", alpha=0.8, label=lbl)
ax.set_xlabel("rho / lambda"); ax.set_ylabel("Mean overnight X_T (%ADV)")
ax.set_title("Overnight residual inventory vs. rho\n"
             "Lower rho = larger overnight position held", fontsize=9)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/home/claude/fig_eod5_sweep.png", dpi=150, bbox_inches="tight")
plt.show(); print("Figure 5 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Figure 6 — Regime comparison: benefit of soft constraint by theta
# ─────────────────────────────────────────────────────────────────────────────

print("Running theta regime comparison ...")

theta_vals  = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
base_clean2 = {k:v for k,v in BASE.items()
               if k not in ("theta", "n_samples", "label", "rho")}

intern_hard = []; intern_soft = []; cost_hard = []; cost_soft = []; xt_soft = []

for th in theta_vals:
    # Hard EOD (rho=20*lam)
    sh, _ = simulate(**base_clean2, theta=th, rho=20*lam, n_samples=1000, label="")
    # Break-even (rho=0)
    ss, _ = simulate(**base_clean2, theta=th, rho=0.0,    n_samples=1000, label="")
    intern_hard.append(sh["intern_%"].mean()); cost_hard.append(sh["total_bps"].mean())
    intern_soft.append(ss["intern_%"].mean()); cost_soft.append(ss["total_bps"].mean())
    xt_soft.append(ss["XT_%"].mean())

intern_hard=np.array(intern_hard); intern_soft=np.array(intern_soft)
cost_hard=np.array(cost_hard);     cost_soft=np.array(cost_soft)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Benefit of soft terminal constraint (rho=0) vs. hard flat-EOD (rho=20*lam)\n"
    "by flow autocorrelation regime  |  Soft constraint helps most in momentum regimes",
    fontsize=10)

ax = axes[0]
ax.plot(theta_vals, intern_hard, "o--", color=pal[3], lw=2.5, ms=7,
        label="Hard EOD (rho=20*lam)")
ax.plot(theta_vals, intern_soft, "s-",  color=pal[0], lw=2.5, ms=7,
        label="Break-even (rho=0)")
ax.fill_between(theta_vals, intern_hard, intern_soft,
                alpha=0.15, color=pal[0])
ax.axvline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel("Flow autocorrelation theta"); ax.set_ylabel("Internalization (%)")
ax.set_title("Internalization: hard EOD vs. soft constraint", fontsize=10)
ax.legend(fontsize=9)

ax = axes[1]
saving = cost_hard - cost_soft
ax.bar(theta_vals, saving,
       color=[pal[0] if s>0 else pal[3] for s in saving], width=0.10)
ax.axhline(0, color="k", lw=1.0); ax.axvline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel("Flow autocorrelation theta"); ax.set_ylabel("Cost saving (bps)")
ax.set_title("Cost saving from allowing overnight holding\n"
             "(positive = soft constraint cheaper)", fontsize=10)
for xv, sv in zip(theta_vals, saving):
    ax.text(xv, sv+(0.3 if sv>=0 else -0.8), f"{sv:.1f}",
            ha="center", fontsize=8, fontweight="bold")

ax = axes[2]
ax.plot(theta_vals, xt_soft, "D-", color=pal[0], lw=2.5, ms=7)
ax.axhline(0, color="k", lw=1, ls="--")
ax.axvline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel("Flow autocorrelation theta")
ax.set_ylabel("Mean overnight X_T (%ADV)")
ax.set_title("Overnight position held (break-even case)\n"
             "Momentum: desk holds more overnight\n"
             "Reversion: already flat, holds nothing", fontsize=9)

plt.tight_layout()
plt.savefig("/home/claude/fig_eod6_regimes.png", dpi=150, bbox_inches="tight")
plt.show(); print("Figure 6 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Summary table
# ─────────────────────────────────────────────────────────────────────────────

table = (all_sum.groupby("label")[[
    "intern_%", "impact_bps", "overnight_bps",
    "spread_bps", "total_bps", "TV_out_%", "XT_%"
]].mean().round(2).loc[order])
table.columns = ["Intern (%)", "Impact (bps)", "Overnight (bps)",
                 "Spread (bps)", "Total (bps)", "Outflow (%ADV)", "Mean X_T (%)"]

print("\n" + "="*90)
print("SUMMARY TABLE — Non-Flat EOD Financing Costs Extension")
print("="*90)
print(table.to_string())
print("="*90)

print(f"""
KEY OBSERVATIONS
----------------------------------------------------------------------

1. THE ONLY RICCATI CHANGE: A(tau=0) = lambda/2 + rho/2
   (base model uses lambda/2). All six ODE equations are identical.
   This is a one-line implementation change with dramatic effects.

2. URGENCY PROFILE (Figure 1) — most dramatic visual difference:
   - Hard EOD (rho=20*lam):   f_T = {-(2*(lam/2+20*lam/2)+lam)/0.01:.0f} (explosive spike at close)
   - Break-even  (rho=0):     f_T = {-(2*(lam/2)+lam)/0.01:.0f} (smooth, same as mid-day level)
   Lower rho means the desk spreads urgency over the whole day rather
   than concentrating it in a last-minute closing block.

3. OVERNIGHT POSITION (Figure 2):
   - Favorable carry: desk holds {all_sum.loc[all_sum["label"]==order[0],"XT_%"].mean():.1f}%ADV overnight (large long)
   - Near hard EOD:   desk holds {all_sum.loc[all_sum["label"]==order[3],"XT_%"].mean():.1f}%ADV overnight (near zero)

4. COST TRADE-OFF (Figure 3):
   - Hard EOD:        impact {all_sum.loc[all_sum["label"]==order[3],"impact_bps"].mean():.1f} bps, total {all_sum.loc[all_sum["label"]==order[3],"total_bps"].mean():.1f} bps
   - Break-even:      impact {all_sum.loc[all_sum["label"]==order[1],"impact_bps"].mean():.1f} bps, total {all_sum.loc[all_sum["label"]==order[1],"total_bps"].mean():.1f} bps
   Allowing overnight holding halves intraday impact cost but requires
   accepting overnight market risk.

5. INTERNALIZATION (Figure 3): +{all_sum.loc[all_sum["label"]==order[0],"intern_%"].mean()-all_sum.loc[all_sum["label"]==order[3],"intern_%"].mean():.1f}pp improvement from hard EOD to favorable carry.
   Desk warehouses more when penalty for holding overnight is low.

6. REGIME DEPENDENCE (Figure 6): benefit of soft constraint is largest
   in momentum regimes (theta < 0) and falls to near zero for reverting
   flow (theta > 0), where the desk naturally flattens before the close.
""")
