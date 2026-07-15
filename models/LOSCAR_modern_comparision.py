import matplotlib.pyplot as plt
import pandas as pd
from LOSCAR_modern import initialize_model, pp_carbonate_cs4


# ------------------- Model Setup ----------------------------

alpha_init = 0.3
rain_init = 6.1
run_time = "10 Myr"
time_step = "1 kyr"
debug = False

M = initialize_model(
    rain_init,
    alpha_init,
    run_time,
    time_step,
    debug,
)

M.run()
M.save_state()

# CS4 carbonate chemistry post-processing
pp_carbonate_cs4(M, ["A", "I", "P"])


plt.style.use("ggplot")

box_names = (
    "M.A_sb,M.A_ib,M.A_db,"
    "M.I_sb,M.I_ib,M.I_db,"
    "M.P_sb,M.P_ib,M.P_db,"
    "M.H_sb"
).split(",")

mark = {"A_": "D", "I_": "s", "P_": "^", "H_": "o"}
col = {"sb": "g", "ib": "k", "db": "r"}

# ============================================================
# Figure 1 — DIC vs TA
# ============================================================

fig1, ax1 = plt.subplots()
fig1.set_size_inches(5, 4)
fig1.set_dpi(200)

for n in box_names:
    x = M.dmo[n].DIC.c[-1] * 1e3
    y = M.dmo[n].TA.c[-1] * 1e3

    if n.startswith("M.H"):  # High Latitude
        facecol = "b"
    else:
        facecol = col[n[-2:]]

    ax1.plot(
        x,
        y,
        marker=mark[n[2:4]],
        markerfacecolor=facecol,
        markeredgecolor="none",
        markersize=12,
        alpha=0.4,
    )

l_data = pd.read_excel("loscar_DIC_vs_TA.xlsx", sheet_name="loscar")
x = l_data.iloc[:, 0]
y = l_data.iloc[:, 1]
lcol = l_data.iloc[:, 3]
symb = l_data.iloc[:, 4]

for i in range(len(x)):
    ax1.plot(
        x[i],
        y[i],
        marker=symb[i],
        markerfacecolor="none",
        color=lcol[i],
        markersize=8,
    )

ax1.set_xlabel("DIC [mmol/kg]")
ax1.set_ylabel("TA [mmol/kg]")
ax1.set_xlim(1.8, 2.5)
ax1.set_ylim(2.2, 2.5)
ax1.grid(True)

fig1.tight_layout()
fig1.savefig(f"loscar_DIC_TA_{alpha_init:.2f}-{rain_init:.2f}.pdf")

# ============================================================
# Figure 2 — PO4 vs O2
# ============================================================

fig2, ax2 = plt.subplots()
fig2.set_size_inches(5, 4)
fig2.set_dpi(200)

for n in box_names:
    x = M.dmo[n].PO4.c[-2] * 1e6
    y = M.dmo[n].O2.c[-2] * 1e6

    if n.startswith("M.H"):  # High Latitude
        facecol = "b"
    else:
        facecol = col[n[-2:]]

    ax2.plot(
        x,
        y,
        marker=mark[n[2:4]],
        markerfacecolor=facecol,
        markeredgecolor="none",
        markersize=12,
        alpha=0.4,
    )

l_data = pd.read_excel("loscar_PO4_vs_O2.xlsx")
x = l_data.iloc[:, 0]
y = l_data.iloc[:, 1]
lcol = l_data.iloc[:, 3]
symb = l_data.iloc[:, 4]

for i in range(len(x)):
    ax2.plot(
        x[i],
        y[i],
        marker=symb[i],
        markerfacecolor="none",
        color=lcol[i],
        markersize=8,
    )

ax2.set_xlabel("PO$_4$ [$\\mu$mol/kg]")
ax2.set_ylabel("O$_2$ [$\\mu$mol/kg]")
ax2.set_xlim(0, 3.5)
ax2.set_ylim(0, 600)
ax2.grid(True)

fig2.tight_layout()
fig2.savefig(f"loscar_PO4_O2_{alpha_init:.2f}-{rain_init:.2f}.pdf")

# ============================================================
# Figure 3 — Deep CO3
# ============================================================

fig3, ax3 = plt.subplots()
fig3.set_size_inches(5, 4)
fig3.set_dpi(200)

x = [1, 2, 3]
labels = ["Atlantic", "Indian", "Pacific"]
symb = ["D", "s", "^"]

y_model = [
    M.A_db.CO3.c[-2] * 1e6,
    M.I_db.CO3.c[-2] * 1e6,
    M.P_db.CO3.c[-2] * 1e6,
]

for i in range(3):
    ax3.plot(
        x[i],
        y_model[i],
        marker=symb[i],
        markerfacecolor="r",
        markeredgecolor="none",
        markersize=12,
        alpha=0.4,
    )

y_ref = [102 * 1.046, 80 * 1.046, 70 * 1.046]

for i in range(3):
    ax3.plot(
        x[i],
        y_ref[i],
        marker=symb[i],
        markerfacecolor="none",
        color="r",
        markersize=8,
    )

ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_ylabel("Deep [CO$_3^{2-}$] [$\\mu$mol/kg]")
ax3.set_ylim(60, 110)

fig3.tight_layout()
fig3.savefig(f"loscar_CO3_{alpha_init:.2f}-{rain_init:.2f}.pdf")

plt.show()