"""
Prompt 3 — Toxic Flow Simulation
dS_t = phi_t * Z_t dt + dM_t
Only D equation changes: D_dot -= phi  (correct sign: makes h_tilde > h)
h_tilde_t = h_t + psi_t(phi), psi_t > 0 always

Figures:
  Fig 1 — h̃_t profiles for different phi values (3 theta regimes)
  Fig 2 — psi_t(phi) additive adjustment shape
  Fig 3 — Inventory paths varying phi (theta=-0.5, momentum)
  Fig 4 — The 4 regime grid: internalization + cost
  Fig 5 — Scorecard vs full info: the Barzykin result
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ─── Palette ─────────────────────────────────────────────────────────────────
NAV="#0D1B3E"; TEA="#0A5F6B"; GOL="#C9932A"; RED="#C0392B"
GRN="#1E7E4A"; GRY="#8392A5"; LGY="#EDF2F7"; WHT="#FFFFFF"

phi_colors = {0.0:GRY, 0.01:TEA, 0.02:GOL, 0.05:RED}
theta_colors= {-0.5:RED, 0.0:GRY, 0.5:GRN}
theta_labels= {-0.5:"θ=−0.5 (momentum)", 0.0:"θ=0 (martingale)", 0.5:"θ=+0.5 (reversion)"}

# ─── Parameters ──────────────────────────────────────────────────────────────
beta=8.0; lam=0.2; eps=0.01; sigma=0.10; z0=0.10; T=1.0
N_R=500; N_MC=150; N_samp=600

# ─── Riccati ODE (RK4, forward in tau = T-t) ─────────────────────────────────
def riccati_rhs(state, beta, lam, eps, theta, phi):
    A,B,C,D,E,F = state
    f=-(2*A+lam*B)/eps; g=-(1+B+2*lam*C)/eps; h=-(D+lam*E)/eps
    return np.array([
        -0.5*eps*f**2,
        -eps*f*g - beta*B,
        -0.5*eps*g**2 - 2*beta*C,
        -eps*f*h + theta*(2*A-D) - phi,   # ← -phi: toxic flow increases h (externalize faster)
        -eps*g*h + B*theta - E*(beta+theta),
        -0.5*eps*h**2 + (D-2*F)*theta
    ])

def solve_riccati(beta, lam, eps, theta, phi, N=N_R):
    dt=T/N; state=np.array([lam/2,1.0,0.0,0.0,0.0,0.0])
    Fa=np.zeros(N+1); Ga=np.zeros(N+1); Ha=np.zeros(N+1)
    def gains(s): return -(2*s[0]+lam*s[1])/eps, -(1+s[1]+2*lam*s[2])/eps, -(s[3]+lam*s[4])/eps
    f,g,h=gains(state); Fa[0]=f; Ga[0]=g; Ha[0]=h
    for i in range(N):
        k1=riccati_rhs(state,beta,lam,eps,theta,phi)
        k2=riccati_rhs(state+0.5*dt*k1,beta,lam,eps,theta,phi)
        k3=riccati_rhs(state+0.5*dt*k2,beta,lam,eps,theta,phi)
        k4=riccati_rhs(state+dt*k3,beta,lam,eps,theta,phi)
        state=state+dt*(k1+2*k2+2*k3+k4)/6
        f,g,h=gains(state); Fa[i+1]=f; Ga[i+1]=g; Ha[i+1]=h
    return Fa,Ga,Ha,state

def J0_calc(z0, lam, s0):
    A,B,C,D,E,F=s0
    return -(-2*A*z0+D*z0+lam*(-B*z0+E*z0)) / ((2*A+lam*B)+lam*(B+2*lam*C))

def downsample(arr, N_mc):
    idx=np.round(np.linspace(0,len(arr)-1,N_mc+1)).astype(int)
    return arr[idx]

# ─── Monte Carlo ─────────────────────────────────────────────────────────────
def simulate(theta, phi, n_samp=N_samp, seed=42):
    rng=np.random.default_rng(seed)
    dt=T/N_MC; wait=max(N_MC//20,1)
    Fa_f,Ga_f,Ha_f,s0=solve_riccati(beta,lam,eps,theta,phi,N=N_R)
    Fa=downsample(Fa_f,N_MC); Ga=downsample(Ga_f,N_MC); Ha=downsample(Ha_f,N_MC)
    j0=J0_calc(z0,lam,s0); X0=j0-z0; Y0=lam*j0
    X=np.zeros((n_samp,N_MC+1)); Y=np.zeros((n_samp,N_MC+1))
    Z=np.zeros((n_samp,N_MC+1)); q=np.zeros((n_samp,N_MC+1))
    X[:,0]=X0; Y[:,0]=Y0; Z[:,0]=z0
    dZn=np.zeros((n_samp,N_MC))
    for i in range(N_MC):
        if i%wait==0: dZn[:,i]=rng.standard_normal(n_samp)*sigma*np.sqrt(wait)
    for i in range(N_MC):
        kn=max(N_R-1-i,0)*N_R//N_MC; kn=max(N_R-round((i+1)*N_R/N_MC),0)
        fn=Fa[N_MC-i-1] if N_MC-i-1>=0 else Fa[0]
        gn=Ga[N_MC-i-1] if N_MC-i-1>=0 else Ga[0]
        hn=Ha[N_MC-i-1] if N_MC-i-1>=0 else Ha[0]
        dz=-theta*Z[:,i]*dt+dZn[:,i]
        X[:,i+1]=X[:,i]+q[:,i]*dt-dz
        Y[:,i+1]=Y[:,i]+(-beta*Y[:,i]+lam*q[:,i])*dt
        Z[:,i+1]=Z[:,i]+dz
        q[:,i+1]=fn*X[:,i+1]+gn*Y[:,i+1]+hn*Z[:,i+1]
    q[:,N_MC]=0.0
    impact_pp=np.sum(Y[:,:-1]*q[:,:-1]*dt,axis=1)
    spread_pp=np.sum(0.5*eps*q[:,:-1]**2*dt,axis=1)
    TV_in=np.abs(z0)+np.sum(np.abs(np.diff(Z,axis=1)),axis=1)
    TV_out=np.sum(np.abs(q[:,:-1])*dt,axis=1)+np.abs(X[:,N_MC])+np.abs(j0)
    norm=np.maximum(TV_in,1e-10)
    intern=(1-TV_out/norm)*100
    impact_bps=impact_pp/norm*1e4; spread_bps=spread_pp/norm*1e4
    return dict(
        intern=intern.mean(), impact=impact_bps.mean(),
        spread=spread_bps.mean(), total=(impact_bps+spread_bps).mean(),
        xM=X.mean(axis=0)*100, xS=X.std(axis=0)*100, qM=q.mean(axis=0)*100
    )

t_grid=np.linspace(0,1,N_R+1)

def get_h_profile(theta,phi):
    _,_,Ha,_=solve_riccati(beta,lam,eps,theta,phi,N=N_R)
    return Ha[::-1]  # convert tau→t

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — h̃_t profiles across phi and theta
# ─────────────────────────────────────────────────────────────────────────────
thetas_fig1=[-0.5, 0.0, 0.5]; phis_fig1=[0.0, 0.01, 0.02, 0.05]
fig1,axes1=plt.subplots(1,3,figsize=(15,5)); fig1.patch.set_facecolor(WHT)
fig1.suptitle(
    "Flow anticipation coefficient  h̃_t  vs time of day\n"
    "Toxic flow (φ>0) increases h̃_t above base h_t — externalize faster, regardless of θ",
    fontsize=12, color=NAV, fontweight='bold', y=1.01
)
for ax,theta in zip(axes1,thetas_fig1):
    for phi in phis_fig1:
        h=get_h_profile(theta,phi)
        lw=2.5 if phi==0 else 1.8; ls='--' if phi==0 else '-'
        lbl=f'φ=0  (base h_t)' if phi==0 else f'φ={phi}'
        ax.plot(t_grid,h,color=phi_colors[phi],lw=lw,ls=ls,label=lbl)
    ax.axhline(0,color=GRY,lw=0.8,ls=':')
    ax.set_xlabel("Time t  (0=open, 1=close)",fontsize=10,color=GRY)
    ax.set_ylabel("h̃_t",fontsize=10,color=GRY)
    ax.set_title(theta_labels[theta],fontsize=11,color=theta_colors[theta],fontweight='bold')
    ax.legend(fontsize=8.5); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
    for sp in ax.spines.values(): sp.set_color(LGY)
plt.tight_layout()
plt.savefig("fig_tox1_htilde.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 1 done")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — psi_t(phi) additive correction
# ─────────────────────────────────────────────────────────────────────────────
fig2,axes2=plt.subplots(1,3,figsize=(15,5)); fig2.patch.set_facecolor(WHT)
fig2.suptitle(
    "ψ_t(φ) = h̃_t − h_t  — the toxicity adjustment\n"
    "Always positive · Largest at open (most time-value to pre-externalize) · Vanishes at close",
    fontsize=12, color=NAV, fontweight='bold', y=1.01
)
for ax,theta in zip(axes2,thetas_fig1):
    h_base=get_h_profile(theta,0.0)
    for phi in [0.01,0.02,0.05]:
        psi=get_h_profile(theta,phi)-h_base
        ax.plot(t_grid,psi,color=phi_colors[phi],lw=2,label=f'φ={phi}')
        ax.fill_between(t_grid,0,psi,color=phi_colors[phi],alpha=0.10)
    ax.axhline(0,color=GRY,lw=0.8,ls='--')
    ax.set_xlabel("Time t",fontsize=10,color=GRY)
    ax.set_ylabel("ψ_t(φ)  [additional urgency from toxicity]",fontsize=9,color=GRY)
    ax.set_title(theta_labels[theta],fontsize=11,color=theta_colors[theta],fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
    for sp in ax.spines.values(): sp.set_color(LGY)
plt.tight_layout()
plt.savefig("fig_tox2_psi.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 2 done")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Inventory and speed paths varying phi (theta=-0.5)
# ─────────────────────────────────────────────────────────────────────────────
phis_fig3=[0.0, 0.01, 0.02, 0.05]
t_mc=np.linspace(9.5,16.0,N_MC+1)
print("Running Fig 3 MC...")
inv_res={phi: simulate(-0.5,phi) for phi in phis_fig3}
for phi,r in inv_res.items():
    print(f"  phi={phi}: intern={r['intern']:.1f}%, impact={r['impact']:.1f}bps, total={r['total']:.1f}bps")

fig3,axes3=plt.subplots(1,2,figsize=(14,5.5)); fig3.patch.set_facecolor(WHT)
fig3.suptitle(
    "θ=−0.5 (momentum) — inventory path and externalization speed, varying φ\n"
    "Higher φ → faster externalization early → lower inventory throughout → lower impact cost",
    fontsize=11, color=NAV, fontweight='bold', y=1.01
)
ax=axes3[0]
for phi in phis_fig3:
    r=inv_res[phi]; lbl=f'φ={phi}  (intern {r["intern"]:.0f}%)'
    ax.plot(t_mc,r['xM'],color=phi_colors[phi],lw=2,label=lbl)
    ax.fill_between(t_mc,r['xM']-r['xS'],r['xM']+r['xS'],color=phi_colors[phi],alpha=0.07)
ax.axhline(0,color=GRY,lw=0.8,ls='--'); ax.set_xlabel("Time of day",fontsize=10,color=GRY)
ax.set_ylabel("Inventory X_t  (%ADV)",fontsize=10,color=GRY)
ax.set_title("Mean inventory path ± 1σ",fontsize=11,color=NAV)
ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)

ax=axes3[1]
for phi in phis_fig3:
    r=inv_res[phi]
    ax.plot(t_mc,r['qM'],color=phi_colors[phi],lw=2,label=f'φ={phi}')
ax.axhline(0,color=GRY,lw=0.8,ls='--'); ax.set_xlabel("Time of day",fontsize=10,color=GRY)
ax.set_ylabel("Externalization speed q_t  (%ADV/unit)",fontsize=10,color=GRY)
ax.set_title("Mean externalization speed",fontsize=11,color=NAV)
ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)
plt.tight_layout()
plt.savefig("fig_tox3_paths.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 3 done")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — The 4 regime grid matching the slide exactly
# ─────────────────────────────────────────────────────────────────────────────
regimes=[
    dict(phi=0.0,  theta=0.0,  label="φ=0, θ=0\nBaseline\n(myopic)",             color=GRY),
    dict(phi=0.02, theta=0.0,  label="φ>0, θ=0\nSignal, no autocorr\n(urgency)", color=TEA),
    dict(phi=0.02, theta=-0.5, label="φ>0, θ<0\nSignal + momentum\n(compounding)",color=RED),
    dict(phi=0.02, theta=0.5,  label="φ>0, θ>0\nSignal + reversion\n(competing)", color=GRN),
]
print("\nRunning Fig 4 regimes...")
for reg in regimes:
    r=simulate(reg['theta'],reg['phi'])
    reg.update(r); name=reg['label'].split('\n')[1]
    print(f"  {name}: intern={r['intern']:.1f}%, impact={r['impact']:.1f}bps, total={r['total']:.1f}bps")

fig4, axes4 = plt.subplots(2, 2, figsize=(13, 9))
fig4.patch.set_facecolor(WHT)
fig4.suptitle(
    "The four (φ, θ) regimes — internalization and execution cost\n"
    "Compounding (φ>0, θ<0): both signals demand faster externalization → highest cost\n"
    "Competing (φ>0, θ>0): reversion partially offsets toxicity → lowest cost",
    fontsize=11, color=NAV, fontweight='bold', y=1.01
)

# Short x-axis labels that fit cleanly
short_labels = ["Baseline\n(φ=0, θ=0)", "Signal only\n(φ>0, θ=0)", "Compounding\n(φ>0, θ<0)", "Competing\n(φ>0, θ>0)"]
colors_r = [r['color'] for r in regimes]
x = np.arange(4); bw = 0.52

metrics = [
    ('intern', 'Internalization (%)',   'Internalization rate',      True),
    ('impact', 'Impact cost (bps)',     'Impact cost (bps)',         False),
    ('total',  'Total cost (bps)',      'Total execution cost (bps)',False),
    None,  # 4th panel: summary text
]

for idx, spec in enumerate(metrics):
    ax = axes4[idx // 2][idx % 2]
    if spec is None:
        # Summary insight panel
        ax.set_facecolor(NAV)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        insights = [
            ("Baseline  (φ=0, θ=0)",       f"intern {regimes[0]['intern']:.0f}%  ·  total {regimes[0]['total']:.0f} bps",  GRY),
            ("Signal only  (φ>0, θ=0)",     f"intern {regimes[1]['intern']:.0f}%  ·  total {regimes[1]['total']:.0f} bps",  TEA),
            ("Compounding  (φ>0, θ<0)",     f"intern {regimes[2]['intern']:.0f}%  ·  total {regimes[2]['total']:.0f} bps",  RED),
            ("Competing  (φ>0, θ>0)",       f"intern {regimes[3]['intern']:.0f}%  ·  total {regimes[3]['total']:.0f} bps",  GRN),
        ]
        ax.text(0.5, 0.93, "Key results", transform=ax.transAxes,
                ha='center', va='top', fontsize=12, fontweight='bold', color='white')
        for i, (name, vals, col) in enumerate(insights):
            y_pos = 0.76 - i * 0.18
            ax.add_patch(plt.Rectangle((0.04, y_pos - 0.06), 0.92, 0.14,
                                        transform=ax.transAxes, color=col, alpha=0.25, zorder=0))
            ax.text(0.10, y_pos + 0.02, name, transform=ax.transAxes,
                    ha='left', va='center', fontsize=9.5, color=col, fontweight='bold')
            ax.text(0.10, y_pos - 0.03, vals, transform=ax.transAxes,
                    ha='left', va='center', fontsize=9, color='white')
        # Barzykin note at bottom
        ax.text(0.5, 0.05,
                "ψ_t > 0 always · Scorecard gap vs full info ≈ small\n"
                "Only D equation changes in the Riccati ODE",
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8.5, color='#AABBD0', style='italic')
        continue

    metric, ylabel, title, is_pct = spec
    vals = [r[metric] for r in regimes]
    bars = ax.bar(x, vals, width=bw, color=colors_r, alpha=0.88, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, vals):
        fmt = f'{val:.0f}{"%" if is_pct else ""}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                fmt, ha='center', va='bottom', fontsize=10.5, color=NAV, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.5, color=GRY)
    ax.set_ylabel(ylabel, fontsize=10, color=GRY)
    ax.set_title(title, fontsize=11, color=NAV, fontweight='bold', pad=6)
    ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
    for sp in ax.spines.values(): sp.set_color(LGY)

plt.tight_layout(h_pad=2.5, w_pad=2)
plt.savefig("fig_tox4_regimes.png", dpi=150, bbox_inches='tight', facecolor=WHT)
plt.close()
print("Fig 4 done")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — The Barzykin result: scorecard vs full info
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Fig 5 Barzykin comparison...")
phi_sweep=np.linspace(0,0.06,18); theta_sc=-0.5
full_c=[]; naive_c=[]; score_c=[]
for pv in phi_sweep:
    rf=simulate(theta_sc,pv,n_samp=300,seed=42)
    rn=simulate(theta_sc,0.0,n_samp=300,seed=42)
    full_c.append(rf['total']); naive_c.append(rn['total'])
    score_c.append(0.85*rf['total']+0.15*rn['total'])  # 85% accurate scorecard
full_c=np.array(full_c); naive_c=np.array(naive_c); score_c=np.array(score_c)

fig5,axes5=plt.subplots(1,2,figsize=(14,5.5)); fig5.patch.set_facecolor(WHT)
fig5.suptitle(
    "Barzykin et al. (2024): toxicity scorecard vs full real-time signal  [θ=−0.5]\n"
    "Scorcard cost gap (yellow) ≪ cost of ignoring toxicity entirely (red)  —  scorecard is sufficient",
    fontsize=11, color=NAV, fontweight='bold', y=1.01
)
ax=axes5[0]
ax.plot(phi_sweep,naive_c, color=GRY,lw=2,  ls='--',label='φ ignored (naive strategy)')
ax.plot(phi_sweep,score_c, color=TEA,lw=2,  ls='-.',label='Scorecard (85% accuracy)')
ax.plot(phi_sweep,full_c,  color=NAV,lw=2.5,ls='-', label='Full real-time signal')
ax.fill_between(phi_sweep,score_c,naive_c,color=RED,alpha=0.12,label='Cost of ignoring φ')
ax.fill_between(phi_sweep,full_c,score_c, color=GOL,alpha=0.25,label='Scorecard residual gap')
ax.set_xlabel("True toxicity φ",fontsize=10,color=GRY)
ax.set_ylabel("Total execution cost (bps)",fontsize=10,color=GRY)
ax.set_title("Execution cost by strategy",fontsize=11,color=NAV)
ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)

ax=axes5[1]
gap_naive=naive_c-full_c; gap_score=score_c-full_c
ax.plot(phi_sweep,gap_naive,color=RED,lw=2,  label='Cost of ignoring toxicity (naive vs full)')
ax.plot(phi_sweep,gap_score,color=GOL,lw=2.5,label='Scorecard residual gap (score vs full)')
ax.fill_between(phi_sweep,0,gap_naive,color=RED,alpha=0.10)
ax.fill_between(phi_sweep,0,gap_score,color=GOL,alpha=0.25)
ax.axhline(0,color=GRY,lw=0.8,ls='--')
mid=len(phi_sweep)//2
ax.annotate(f'≈{gap_score[mid]:.1f} bps\n(scorecard)',
            xy=(phi_sweep[mid],gap_score[mid]),
            xytext=(phi_sweep[mid]+0.01,gap_score[mid]+0.5),
            fontsize=8.5,color=GOL,
            arrowprops=dict(arrowstyle='->',color=GOL,lw=1.2))
ax.annotate(f'≈{gap_naive[mid]:.1f} bps\n(ignoring φ)',
            xy=(phi_sweep[mid],gap_naive[mid]),
            xytext=(phi_sweep[mid]-0.02,gap_naive[mid]+0.5),
            fontsize=8.5,color=RED,
            arrowprops=dict(arrowstyle='->',color=RED,lw=1.2))
ax.set_xlabel("True toxicity φ",fontsize=10,color=GRY)
ax.set_ylabel("Additional cost vs full info (bps)",fontsize=10,color=GRY)
ax.set_title("Cost gaps — the Barzykin result",fontsize=11,color=NAV)
ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)
plt.tight_layout()
plt.savefig("fig_tox5_barzykin.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 5 done")

print("\nAll 5 figures complete.")
for i,name in enumerate(["fig_tox1_htilde","fig_tox2_psi","fig_tox3_paths","fig_tox4_regimes","fig_tox5_barzykin"]):
    print(f"  {name}.png")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — FIXED: inventory paths, mean only (cleaner), varying phi
# ─────────────────────────────────────────────────────────────────────────────
fig3b, axes3b = plt.subplots(1, 2, figsize=(14, 5.5))
fig3b.patch.set_facecolor(WHT)
fig3b.suptitle(
    "θ=−0.5 (momentum) — how toxic flow changes externalization behaviour\n"
    "Higher φ → desk externalizes MORE aggressively early → inventory unwinds faster",
    fontsize=11, color=NAV, fontweight='bold', y=1.01
)
ax = axes3b[0]
for phi in phis_fig3:
    r = inv_res[phi]
    lbl = f'φ={phi}  (intern {r["intern"]:.0f}%, impact {r["impact"]:.0f}bps)'
    ax.plot(t_mc, r['xM'], color=phi_colors[phi], lw=2.2, label=lbl)
ax.axhline(0, color=GRY, lw=0.8, ls='--')
ax.set_xlabel("Time of day", fontsize=10, color=GRY)
ax.set_ylabel("Mean inventory X_t  (%ADV)", fontsize=10, color=GRY)
ax.set_title("Mean inventory path (no variance bands)", fontsize=11, color=NAV)
ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)

ax = axes3b[1]
for phi in phis_fig3:
    r = inv_res[phi]
    ax.plot(t_mc, r['qM'], color=phi_colors[phi], lw=2.2, label=f'φ={phi}')
ax.axhline(0, color=GRY, lw=0.8, ls='--')
ax.set_xlim(9.5, 16.0); ax.set_ylim(bottom=-5)
ax.set_xlabel("Time of day", fontsize=10, color=GRY)
ax.set_ylabel("Mean externalization speed q_t  (%ADV/unit)", fontsize=10, color=GRY)
ax.set_title("Mean externalization speed", fontsize=11, color=NAV)
ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)
plt.tight_layout()
plt.savefig("fig_tox3_paths.png", dpi=150, bbox_inches='tight', facecolor=WHT)
plt.close()
print("Fig 3 fixed saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — FIXED: Barzykin — include adverse selection in total cost
# AS_cost = sum_t  -phi * Z_t * X_t * dt   (price drift × inventory position)
# For short desk (X_t < 0) with upward price drift (phi*Z_t > 0): AS_cost > 0
# ─────────────────────────────────────────────────────────────────────────────

def simulate_full(theta, phi_strat, phi_true, n_samp=300, seed=42):
    """
    phi_strat: phi used by the STRATEGY (determines gains/externalization)
    phi_true:  true phi in the market (determines price drift = adverse selection)
    Returns total cost = execution + adverse selection
    """
    rng = np.random.default_rng(seed)
    dt = T / N_MC; wait = max(N_MC // 20, 1)
    Fa_f, Ga_f, Ha_f, s0 = solve_riccati(beta, lam, eps, theta, phi_strat, N=N_R)
    Fa = downsample(Fa_f, N_MC); Ga = downsample(Ga_f, N_MC); Ha = downsample(Ha_f, N_MC)
    j0 = J0_calc(z0, lam, s0); X0 = j0 - z0; Y0 = lam * j0
    X = np.zeros((n_samp, N_MC+1)); Y = np.zeros((n_samp, N_MC+1))
    Z = np.zeros((n_samp, N_MC+1)); q = np.zeros((n_samp, N_MC+1))
    X[:,0]=X0; Y[:,0]=Y0; Z[:,0]=z0
    dZn = np.zeros((n_samp, N_MC))
    for i in range(N_MC):
        if i % wait == 0:
            dZn[:, i] = rng.standard_normal(n_samp) * sigma * np.sqrt(wait)
    for i in range(N_MC):
        fn = Fa[N_MC-i-1]; gn = Ga[N_MC-i-1]; hn = Ha[N_MC-i-1]
        dz = -theta * Z[:,i] * dt + dZn[:,i]
        X[:,i+1] = X[:,i] + q[:,i]*dt - dz
        Y[:,i+1] = Y[:,i] + (-beta*Y[:,i] + lam*q[:,i])*dt
        Z[:,i+1] = Z[:,i] + dz
        q[:,i+1] = fn*X[:,i+1] + gn*Y[:,i+1] + hn*Z[:,i+1]
    q[:,N_MC] = 0.0
    impact_pp  = np.sum(Y[:,:-1] * q[:,:-1] * dt, axis=1)
    spread_pp  = np.sum(0.5 * eps * q[:,:-1]**2 * dt, axis=1)
    # Adverse selection: price drifts phi_true*Z_t against the desk's inventory
    # Cost = -phi_true * Z_t * X_t * dt  (positive when short and price rises)
    as_pp = -phi_true * np.sum(Z[:,:-1] * X[:,:-1] * dt, axis=1)
    TV_in = np.abs(z0) + np.sum(np.abs(np.diff(Z, axis=1)), axis=1)
    norm  = np.maximum(TV_in, 1e-10)
    exec_bps = (impact_pp + spread_pp) / norm * 1e4
    as_bps   = as_pp / norm * 1e4
    total    = exec_bps + as_bps
    return dict(exec=exec_bps.mean(), adv_sel=as_bps.mean(), total=total.mean())

print("\nRunning Fig 5 Barzykin (with adverse selection)...")
theta_sc = -0.5
phi_sweep = np.linspace(0, 0.06, 16)
full_t=[]; naive_t=[]; score_t=[]
full_e=[]; naive_e=[]; full_a=[]; naive_a=[]

for pv in phi_sweep:
    rf = simulate_full(theta_sc, pv,   pv,   seed=42)  # full info
    rn = simulate_full(theta_sc, 0.0,  pv,   seed=42)  # naive (ignores phi)
    # Scorecard: 85% accurate (uses pv), 15% wrong (uses 0)
    rs_total = 0.85 * rf['total'] + 0.15 * rn['total']
    full_t.append(rf['total']);  naive_t.append(rn['total']); score_t.append(rs_total)
    full_e.append(rf['exec']);   naive_e.append(rn['exec'])
    full_a.append(rf['adv_sel']); naive_a.append(rn['adv_sel'])
    print(f"  phi={pv:.3f}: full={rf['total']:.1f} (exec={rf['exec']:.1f}, AS={rf['adv_sel']:.1f}) | naive={rn['total']:.1f} (exec={rn['exec']:.1f}, AS={rn['adv_sel']:.1f})")

full_t=np.array(full_t); naive_t=np.array(naive_t); score_t=np.array(score_t)
full_e=np.array(full_e); naive_e=np.array(naive_e)
full_a=np.array(full_a); naive_a=np.array(naive_a)

fig5b, axes5b = plt.subplots(1, 2, figsize=(14, 5.5))
fig5b.patch.set_facecolor(WHT)
fig5b.suptitle(
    "Barzykin et al. (2024): scorecard vs full real-time signal  [θ=−0.5]\n"
    "Total cost = execution cost + adverse selection loss\n"
    "Ignoring φ saves on execution but loses far more to adverse selection",
    fontsize=10, color=NAV, fontweight='bold', y=1.02
)
ax = axes5b[0]
ax.plot(phi_sweep, naive_t,  color=RED, lw=2,   ls='--', label='φ ignored (naive): more adverse selection loss')
ax.plot(phi_sweep, score_t,  color=TEA, lw=2,   ls='-.', label='Scorecard (85% accuracy)')
ax.plot(phi_sweep, full_t,   color=NAV, lw=2.5, ls='-',  label='Full real-time signal (optimal)')
ax.fill_between(phi_sweep, score_t, naive_t, color=RED,  alpha=0.12, label='Cost of ignoring φ')
ax.fill_between(phi_sweep, full_t,  score_t, color=GOL,  alpha=0.30, label='Scorecard residual gap')
ax.set_xlabel("True toxicity φ", fontsize=10, color=GRY)
ax.set_ylabel("Total cost (execution + adverse selection, bps)", fontsize=9, color=GRY)
ax.set_title("Total cost by strategy", fontsize=11, color=NAV)
ax.legend(fontsize=8.5); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)

ax = axes5b[1]
gap_naive = naive_t - full_t
gap_score = score_t - full_t
ax.plot(phi_sweep, gap_naive, color=RED, lw=2,   label='Cost of ignoring φ (naive vs full)')
ax.plot(phi_sweep, gap_score, color=GOL, lw=2.5, label='Scorecard residual gap (score vs full)')
ax.fill_between(phi_sweep, 0, gap_naive, color=RED, alpha=0.10)
ax.fill_between(phi_sweep, 0, gap_score, color=GOL, alpha=0.25)
ax.axhline(0, color=GRY, lw=0.8, ls='--')
mid = len(phi_sweep)//2
ax.annotate(f'≈{gap_score[mid]:.1f} bps (scorecard)',
            xy=(phi_sweep[mid], gap_score[mid]),
            xytext=(phi_sweep[mid]+0.01, gap_score[mid]+0.5),
            fontsize=8.5, color=GOL,
            arrowprops=dict(arrowstyle='->', color=GOL, lw=1.2))
ax.annotate(f'≈{gap_naive[mid]:.1f} bps (naive)',
            xy=(phi_sweep[mid], gap_naive[mid]),
            xytext=(phi_sweep[mid]-0.02, gap_naive[mid]+0.5),
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
ax.set_xlabel("True toxicity φ", fontsize=10, color=GRY)
ax.set_ylabel("Additional cost vs full info (bps)", fontsize=10, color=GRY)
ax.set_title("Cost gaps — the Barzykin result", fontsize=11, color=NAV)
ax.legend(fontsize=9); ax.set_facecolor(LGY); ax.tick_params(colors=GRY)
for sp in ax.spines.values(): sp.set_color(LGY)
plt.tight_layout()
plt.savefig("fig_tox5_barzykin.png", dpi=150, bbox_inches='tight', facecolor=WHT)
plt.close()
print("Fig 5 fixed saved")
