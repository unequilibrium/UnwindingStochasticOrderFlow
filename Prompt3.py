"""
Prompt 3 — Toxic Flow Simulation (v2, reworked)

Key design choices vs v1:
  - phi range 0 → 0.20 (10× larger) so h̃_t differences are visible
  - Total cost = execution cost + adverse selection loss (phi*Z*X*dt)
  - All three theta regimes shown side by side
  - Shared axes for direct comparison

Figures:
  Fig 1 — h̃_t profiles: 3 theta panels, phi=0/0.05/0.10/0.20
  Fig 2 — Inventory paths: 3 theta panels, phi=0 vs phi=0.10 (base vs toxic)
  Fig 3 — Internalization and total cost heatmap across (phi, theta) grid
  Fig 4 — Cost decomposition: execution vs adverse selection for each regime
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Palette ───────────────────────────────────────────────────────────────────
NAV="#0D1B3E"; TEA="#0A5F6B"; GOL="#C9932A"; RED="#C0392B"
GRN="#1E7E4A"; GRY="#8392A5"; LGY="#EDF2F7"; WHT="#FFFFFF"; DGRY="#50606A"

THETA_COLS  = {-0.5: RED, 0.0: GRY, 0.5: GRN}
THETA_LBLS  = {-0.5: "θ=−0.5  (momentum)", 0.0: "θ=0  (martingale)", 0.5: "θ=+0.5  (reversion)"}
PHI_COLS    = {0.0: GRY, 0.05: TEA, 0.10: GOL, 0.20: RED}

# ── Parameters ────────────────────────────────────────────────────────────────
beta=8.0; lam=0.2; eps=0.01; sigma=0.10; z0=0.10; T=1.0
N_R=400; N_MC=150; N_SAMP=800

# ── Riccati: -phi in D equation (correct sign: makes h̃ > h) ─────────────────
def solve_riccati(theta, phi, N=N_R):
    dt=T/N; s=np.array([lam/2,1.,0.,0.,0.,0.]); Fa=np.zeros(N+1); Ha=np.zeros(N+1)
    def rhs(s):
        A,B,C,D,E,F=s; f=-(2*A+lam*B)/eps; g=-(1+B+2*lam*C)/eps; h=-(D+lam*E)/eps
        return np.array([-0.5*eps*f**2,-eps*f*g-beta*B,-0.5*eps*g**2-2*beta*C,
                         -eps*f*h+theta*(2*A-D)-phi,
                         -eps*g*h+B*theta-E*(beta+theta),-0.5*eps*h**2+(D-2*F)*theta])
    def gains(s): return -(2*s[0]+lam*s[1])/eps, -(s[3]+lam*s[4])/eps
    f,h=gains(s); Fa[0]=f; Ha[0]=h
    for i in range(N):
        k1=rhs(s); k2=rhs(s+0.5*dt*k1); k3=rhs(s+0.5*dt*k2); k4=rhs(s+dt*k3)
        s=s+dt*(k1+2*k2+2*k3+k4)/6; f,h=gains(s); Fa[i+1]=f; Ha[i+1]=h
    return Fa, Ha, s

def J0_calc(z0, lam, s0):
    A,B,C,D,E,F=s0
    return -(-2*A*z0+D*z0+lam*(-B*z0+E*z0))/((2*A+lam*B)+lam*(B+2*lam*C))

def downsample(arr, n): return arr[np.round(np.linspace(0,len(arr)-1,n+1)).astype(int)]

# ── Monte Carlo ───────────────────────────────────────────────────────────────
def simulate(theta, phi, n_samp=N_SAMP, seed=42):
    rng=np.random.default_rng(seed); dt=T/N_MC; wait=max(N_MC//20,1)
    Fa_f,Ha_f,s0=solve_riccati(theta,phi,N=N_R)
    Fa=downsample(Fa_f,N_MC); Ha=downsample(Ha_f,N_MC)
    j0=J0_calc(z0,lam,s0); X0=j0-z0; Y0=lam*j0
    X=np.zeros((n_samp,N_MC+1)); Y=np.zeros((n_samp,N_MC+1))
    Z=np.zeros((n_samp,N_MC+1)); q=np.zeros((n_samp,N_MC+1))
    X[:,0]=X0; Y[:,0]=Y0; Z[:,0]=z0
    dZn=np.zeros((n_samp,N_MC))
    for i in range(N_MC):
        if i%wait==0: dZn[:,i]=rng.standard_normal(n_samp)*sigma*np.sqrt(wait)
    for i in range(N_MC):
        fn=Fa[N_MC-i-1]; hn=Ha[N_MC-i-1]
        dz=-theta*Z[:,i]*dt+dZn[:,i]
        X[:,i+1]=X[:,i]+q[:,i]*dt-dz
        Y[:,i+1]=Y[:,i]+(-beta*Y[:,i]+lam*q[:,i])*dt
        Z[:,i+1]=Z[:,i]+dz
        q[:,i+1]=fn*X[:,i+1]+hn*Z[:,i+1]
    q[:,N_MC]=0.0
    impact_pp = np.sum(Y[:,:-1]*q[:,:-1]*dt,axis=1)
    spread_pp = np.sum(0.5*eps*q[:,:-1]**2*dt,axis=1)
    # Adverse selection: price drift phi*Z_t hits the desk's inventory X_t
    adv_sel_pp= -phi*np.sum(Z[:,:-1]*X[:,:-1]*dt,axis=1)
    TV_in =np.abs(z0)+np.sum(np.abs(np.diff(Z,axis=1)),axis=1)
    TV_out=np.sum(np.abs(q[:,:-1])*dt,axis=1)+np.abs(X[:,N_MC])+np.abs(j0)
    norm=np.maximum(TV_in,1e-10)
    intern=(1-TV_out/norm)*100
    exec_bps=(impact_pp+spread_pp)/norm*1e4
    as_bps=adv_sel_pp/norm*1e4
    total_bps=exec_bps+as_bps
    t_mc=np.linspace(9.5,16.0,N_MC+1)
    return dict(intern=intern.mean(), exec_bps=exec_bps.mean(),
                as_bps=as_bps.mean(), total_bps=total_bps.mean(),
                xM=X.mean(0)*100, xS=X.std(0)*100,
                qM=q.mean(0)*100, t=t_mc)

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(LGY)
    for sp in ax.spines.values(): sp.set_color(LGY)
    ax.tick_params(colors=DGRY,labelsize=8.5)
    ax.set_title(title,fontsize=10,color=NAV,fontweight='bold',pad=7)
    ax.set_xlabel(xlabel,fontsize=9,color=GRY)
    ax.set_ylabel(ylabel,fontsize=9,color=GRY)
    ax.grid(True,color=WHT,lw=0.8,zorder=0); ax.set_axisbelow(True)

# ──────────────────────────────────────────────────────────────────────────────
# FIG 1 — h̃_t profiles: 3 theta panels, varying phi
# ──────────────────────────────────────────────────────────────────────────────
thetas=[-0.5,0.0,0.5]; phis=[0.0,0.05,0.10,0.20]
t_grid=np.linspace(0,1,N_R+1)

def get_h(theta,phi):
    _,Ha,_=solve_riccati(theta,phi,N=N_R); return Ha[::-1]

fig1,axes1=plt.subplots(1,3,figsize=(15,5.5)); fig1.patch.set_facecolor(WHT)
fig1.suptitle(
    "Flow anticipation coefficient  h̃_t  vs time of day\n"
    "Larger φ → higher h̃_t → desk externalizes faster regardless of autocorrelation regime",
    fontsize=11,color=NAV,fontweight='bold',y=1.01)

for ax,theta in zip(axes1,thetas):
    h_base=get_h(theta,0.0)
    for phi in phis:
        h=get_h(theta,phi)
        lw=2.5 if phi==0 else 1.8; ls='--' if phi==0 else '-'
        lbl=f'φ=0  (base h_t)' if phi==0 else f'φ={phi}'
        ax.plot(t_grid,h,color=PHI_COLS[phi],lw=lw,ls=ls,label=lbl)
        if phi>0:
            ax.fill_between(t_grid,h_base,h,alpha=0.07,color=PHI_COLS[phi])
    ax.axhline(0,color=GRY,lw=0.8,ls=':')
    style(ax,THETA_LBLS[theta],"Time t  (0=open, 1=close)","h̃_t  (flow anticipation gain)")
    ax.legend(fontsize=8.5,framealpha=0.9,edgecolor=LGY)

plt.tight_layout()
plt.savefig("fig_tox1_htilde.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 1 saved")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 2 — Inventory paths: base (φ=0) vs toxic (φ=0.10), all 3 theta regimes
# Side-by-side with shared y-axis to see cross-regime differences
# ──────────────────────────────────────────────────────────────────────────────
phi_path=0.10
print("Running Fig 2 MC...")
path_res={}
for theta in thetas:
    path_res[(theta,0.0  )]=simulate(theta,0.0,  seed=42)
    path_res[(theta,phi_path)]=simulate(theta,phi_path,seed=42)
    print(f"  theta={theta}: base intern={path_res[(theta,0.0)]['intern']:.0f}% "
          f"| phi={phi_path} intern={path_res[(theta,phi_path)]['intern']:.0f}%")

# Shared y across all paths
all_hi=[path_res[k]['xM'].max()+path_res[k]['xS'].max() for k in path_res]
all_lo=[path_res[k]['xM'].min()-path_res[k]['xS'].min() for k in path_res]
y_lo=min(all_lo)*1.08; y_hi=min(max(all_hi)*1.2,3)

fig2,axes2=plt.subplots(1,3,figsize=(15,5.5),sharey=True); fig2.patch.set_facecolor(WHT)
fig2.suptitle(
    f"Inventory paths — base (φ=0) vs toxic (φ={phi_path}) across flow regimes\n"
    "Shared y-axis: momentum regime is most affected — desk externalizes faster all day",
    fontsize=11,color=NAV,fontweight='bold',y=1.01)

for ax,theta in zip(axes2,thetas):
    col=THETA_COLS[theta]
    rb=path_res[(theta,0.0)]; rt=path_res[(theta,phi_path)]
    t=rb['t']

    ax.fill_between(t,rb['xM']-rb['xS'],rb['xM']+rb['xS'],alpha=0.10,color=GRY)
    ax.plot(t,rb['xM'],color=GRY,lw=2,ls='--',label=f'φ=0  (intern {rb["intern"]:.0f}%)')

    ax.fill_between(t,rt['xM']-rt['xS'],rt['xM']+rt['xS'],alpha=0.12,color=col)
    ax.plot(t,rt['xM'],color=col,lw=2.2,ls='-',label=f'φ={phi_path}  (intern {rt["intern"]:.0f}%)')

    ax.axhline(0,color=NAV,lw=1.0,ls=':',alpha=0.5)
    ax.set_ylim(y_lo,y_hi)
    style(ax,THETA_LBLS[theta],"Time of day","Mean inventory X_t  (%ADV)")
    ax.legend(fontsize=8.5,framealpha=0.9,edgecolor=LGY)

plt.tight_layout()
plt.savefig("fig_tox2_paths.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 2 saved")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 3 — Cost decomposition: execution vs adverse selection across regimes
# 3 columns (theta), 3 metrics (exec, AS, total) with phi on x-axis
# ──────────────────────────────────────────────────────────────────────────────
phis_sweep=np.array([0.0,0.05,0.10,0.15,0.20])
print("\nRunning Fig 3 cost sweep...")
cost_res={}
for theta in thetas:
    cost_res[theta]=[]
    for phi in phis_sweep:
        r=simulate(theta,phi,n_samp=500,seed=42)
        cost_res[theta].append(r)
        print(f"  theta={theta}, phi={phi:.2f}: exec={r['exec_bps']:.1f} AS={r['as_bps']:.1f} total={r['total_bps']:.1f} intern={r['intern']:.1f}%")

fig3,axes3=plt.subplots(2,3,figsize=(15,9)); fig3.patch.set_facecolor(WHT)
fig3.suptitle(
    "Cost decomposition across toxicity (φ) and flow regime (θ)\n"
    "Adverse selection grows with φ² — ignoring toxicity means underestimating true cost",
    fontsize=11,color=NAV,fontweight='bold',y=1.01)

# Row 0: internalization rate
# Row 1: cost breakdown (stacked: execution + adverse selection)
for ci,theta in enumerate(thetas):
    col=THETA_COLS[theta]
    rows=cost_res[theta]

    intern_vals=[r['intern'] for r in rows]
    exec_vals  =[r['exec_bps'] for r in rows]
    as_vals    =[r['as_bps'] for r in rows]
    total_vals =[r['total_bps'] for r in rows]

    # Top row: internalization
    ax=axes3[0][ci]
    ax.plot(phis_sweep,intern_vals,'o-',color=col,lw=2.5,ms=7,
            markeredgecolor=WHT,markeredgewidth=1.2,zorder=3)
    ax.fill_between(phis_sweep,intern_vals[0],intern_vals,alpha=0.12,color=col)
    ax.axhline(intern_vals[0],color=GRY,lw=1,ls='--',alpha=0.6)
    style(ax,THETA_LBLS[theta],"Toxicity φ","Internalization (%)")
    ax.set_ylim(max(0,min(intern_vals)-5),max(intern_vals)+5)

    # Bottom row: stacked cost bars
    ax=axes3[1][ci]
    x=np.arange(len(phis_sweep)); bw=0.55

    # Execution cost (bottom)
    bars_e=ax.bar(x,exec_vals,width=bw,color=TEA,alpha=0.88,
                  edgecolor=WHT,linewidth=1.1,label='Execution cost',zorder=3)
    # Adverse selection on top
    bars_a=ax.bar(x,as_vals,bottom=exec_vals,width=bw,color=RED,alpha=0.88,
                  edgecolor=WHT,linewidth=1.1,label='Adverse selection cost',zorder=3)

    # Total labels
    for xi,tot in zip(x,total_vals):
        ax.text(xi,tot+0.4,f'{tot:.0f}',ha='center',va='bottom',
                fontsize=8.5,fontweight='bold',color=NAV,zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.2f}' for p in phis_sweep],fontsize=8.5,color=DGRY)
    ax.set_xlabel("Toxicity φ",fontsize=9,color=GRY)
    ax.set_ylabel("Cost (bps)",fontsize=9,color=GRY)
    ax.set_title(f"Cost breakdown — {THETA_LBLS[theta]}",fontsize=10,color=NAV,fontweight='bold',pad=7)
    ax.set_facecolor(LGY)
    for sp in ax.spines.values(): sp.set_color(LGY)
    ax.tick_params(colors=DGRY,labelsize=8.5)
    ax.grid(True,color=WHT,lw=0.8,axis='y',zorder=0); ax.set_axisbelow(True)
    if ci==0:
        patches=[mpatches.Patch(facecolor=TEA,label='Execution cost',edgecolor=WHT),
                 mpatches.Patch(facecolor=RED,label='Adverse selection',edgecolor=WHT)]
        ax.legend(handles=patches,fontsize=8.5,framealpha=0.9,edgecolor=LGY)

# Row labels
for ci,theta in enumerate(thetas):
    axes3[0][ci].set_title(THETA_LBLS[theta],fontsize=10,color=THETA_COLS[theta],fontweight='bold',pad=7)

plt.tight_layout(h_pad=3)
plt.savefig("fig_tox3_costs.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 3 saved")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 4 — Cross-regime summary heatmap: total cost and internalization
# ──────────────────────────────────────────────────────────────────────────────
fig4,axes4=plt.subplots(1,2,figsize=(13,5.5)); fig4.patch.set_facecolor(WHT)
fig4.suptitle(
    "Heatmap: total cost (execution + adverse selection) and internalization across (φ, θ)\n"
    "Compounding regime (φ>0, θ<0): highest cost · Competing (φ>0, θ>0): toxicity partially offset",
    fontsize=11,color=NAV,fontweight='bold',y=1.01)

theta_labels_short=["θ=−0.5\n(momentum)","θ=0\n(martingale)","θ=+0.5\n(reversion)"]
phi_labels=[f"φ={p:.2f}" for p in phis_sweep]

total_grid =np.array([[cost_res[theta][pi]['total_bps'] for theta in thetas] for pi in range(len(phis_sweep))])
intern_grid=np.array([[cost_res[theta][pi]['intern']    for theta in thetas] for pi in range(len(phis_sweep))])

for ax,grid,title,cmap,fmt in zip(
    axes4,
    [total_grid,intern_grid],
    ["Total cost (bps)","Internalization (%)"],
    ["YlOrRd","YlGn"],
    [".0f",".0f"]
):
    im=ax.imshow(grid,aspect="auto",cmap=cmap,origin="lower",interpolation='nearest')
    ax.set_xticks(range(len(thetas))); ax.set_xticklabels(theta_labels_short,fontsize=9,color=DGRY)
    ax.set_yticks(range(len(phis_sweep))); ax.set_yticklabels(phi_labels,fontsize=9,color=DGRY)
    ax.set_xlabel("Flow regime (θ)",fontsize=10,color=GRY)
    ax.set_ylabel("Toxicity (φ)",fontsize=10,color=GRY)
    ax.set_title(title,fontsize=11,color=NAV,fontweight='bold',pad=8)
    ax.grid(False); ax.tick_params(which='both',bottom=False,left=False)
    for sp in ax.spines.values(): sp.set_color(GRY); sp.set_linewidth(0.8)
    plt.colorbar(im,ax=ax,shrink=0.85)
    vmin,vmax=grid.min(),grid.max()
    for ii in range(len(phis_sweep)):
        for jj in range(len(thetas)):
            val=grid[ii,jj]; norm=(val-vmin)/(vmax-vmin+1e-8)
            txt_col=WHT if norm>0.55 else "#1a1a1a"
            ax.text(jj,ii,f"{val:{fmt}}",ha="center",va="center",
                    fontsize=10,fontweight='bold',color=txt_col)

plt.tight_layout(w_pad=3)
plt.savefig("fig_tox4_heatmap.png",dpi=150,bbox_inches='tight',facecolor=WHT); plt.close()
print("Fig 4 saved")
print("\nAll figures saved: fig_tox1_htilde, fig_tox2_paths, fig_tox3_costs, fig_tox4_heatmap")
