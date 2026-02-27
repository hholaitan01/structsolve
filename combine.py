"""
StructSolve — Structural Analysis Suite
BS 8110 · Slope Deflection Method
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd

from beam_solver import BeamSolver
from frame_solver import FrameSolver
from bs8110 import (
    beam_flexural_design, beam_shear_design,
    auto_design_from_beam_analysis, deflection_check,
    format_design_summary, beam_flexural_design_T, design_continuous_beam,
)
from pdf_export import (
    export_beam_analysis_pdf, export_beam_design_pdf,
    export_frame_pdf, export_rc_design_pdf,
)

st.set_page_config(page_title="StructSolve | BS 8110", page_icon="🏗️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#0F1117}
section[data-testid="stSidebar"]{background:#1A1D27!important}
section[data-testid="stSidebar"]>div{padding-top:1rem}
.main-header{background:linear-gradient(135deg,#1A1D27,#22263A);border:1px solid #2A2E40;
  border-radius:12px;padding:1.6rem 2rem;margin-bottom:1.5rem;position:relative;overflow:hidden}
.main-header::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#4ECDC4,#FF6B6B,#FFE66D)}
.main-header h1{font-family:'Syne',sans-serif!important;font-weight:800;font-size:2rem!important;
  color:#E8EAF0!important;margin:0!important;letter-spacing:-1px}
.main-header p{color:#8B92A8;font-size:.82rem;margin:.35rem 0 0}
.badge{display:inline-block;background:#4ECDC4;color:#0F1117;font-size:.62rem;font-weight:700;
  padding:2px 8px;border-radius:4px;margin-left:8px;vertical-align:middle;letter-spacing:1px}
.result-card{background:#22263A;border-left:3px solid #4ECDC4;border-radius:0 8px 8px 0;
  padding:.9rem 1.2rem;margin:.45rem 0;font-size:.83rem}
.result-card.warning{border-left-color:#FFE66D}
.result-card.danger{border-left-color:#FF6B6B}
.result-card.muted{border-left-color:#2A2E40}
.section-label{font-size:.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:#4ECDC4;border-bottom:1px solid #2A2E40;padding-bottom:.4rem;margin:1.4rem 0 .8rem}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.75rem;margin:1rem 0}
.metric-box{background:#22263A;border:1px solid #2A2E40;border-radius:8px;padding:.85rem;text-align:center}
.metric-box .val{font-size:1.3rem;font-weight:600;color:#4ECDC4}
.metric-box .val.red{color:#FF6B6B}.metric-box .val.yel{color:#FFE66D}.metric-box .val.grn{color:#6BCB77}
.metric-box .lbl{font-size:.68rem;color:#8B92A8;margin-top:3px}
.workings-box{background:#0A0C12;border:1px solid #2A2E40;border-radius:8px;padding:1rem 1.2rem;
  font-size:.74rem;line-height:1.9;color:#C0C6D4;white-space:pre-wrap;
  font-family:'JetBrains Mono',monospace;max-height:520px;overflow-y:auto}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label{color:#8B92A8!important;font-size:.78rem!important}
div[data-testid="stNumberInput"] input{background:#22263A!important;border:1px solid #2A2E40!important;
  color:#E8EAF0!important;border-radius:6px!important}
div[data-testid="stSelectbox"]>div>div{background:#22263A!important;border:1px solid #2A2E40!important;color:#E8EAF0!important}
.stButton>button{background:linear-gradient(135deg,#4ECDC4,#45B7D1)!important;color:#0F1117!important;
  border:none!important;border-radius:8px!important;font-weight:600!important;transition:all .2s!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 4px 15px rgba(78,205,196,.3)!important}
div[data-testid="stTabs"] [role="tab"]{color:#8B92A8;font-size:.82rem}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:#4ECDC4}
.stExpander{border:1px solid #2A2E40!important;border-radius:8px!important}
div[data-testid="stExpander"] summary{color:#8B92A8!important;font-size:.82rem!important}
</style>
""", unsafe_allow_html=True)

BG="#0F1117"; SURF="#1A1D27"; SURF2="#22263A"; GRID="#2A2E40"
TXT="#8B92A8"; TEAL="#4ECDC4"; RED="#FF6B6B"; YEL="#FFE66D"; GRN="#6BCB77"


# ─── Plot helpers ─────────────────────────────────────────────
def _ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURF2); ax.tick_params(colors=TXT, labelsize=8)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT); ax.title.set_color(TEAL)
    for s in ax.spines.values(): s.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=.6, ls="--", alpha=.7)
    if title:  ax.set_title(title,  fontsize=9, pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)

def _ann(ax, x, y):
    im = int(np.argmax(y)); iv = int(np.argmin(y))
    for i, c in [(im, GRN), (iv, RED)]:
        ax.plot(x[i], y[i], "o", color=c, ms=5, zorder=5)
        ax.annotate(f"{y[i]:.2f}", xy=(x[i], y[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=7, color=c)


# ─── Beam visualiser ──────────────────────────────────────────
def draw_beam_system(spans, support_types, span_loads):
    if not spans: return
    total_L = sum(s["L"] for s in spans)
    fig, ax = plt.subplots(figsize=(max(10, total_L * 1.6), 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    BY = 0.0; LT = .52
    ax.plot([0, total_L], [BY]*2, lw=5, color="#C0C6D4", solid_capstyle="round", zorder=3)
    node_x = [0.0]; cx = 0.0
    for s in spans:
        cx += s["L"]; node_x.append(cx)
    for i, (x0, stype_) in enumerate(zip(node_x, support_types)):
        ax.text(x0, BY-.3, chr(65+i), ha="center", va="top",
                color="#C0C6D4", fontsize=9, fontweight="600")
        if stype_ == "Fixed":
            ax.plot([x0]*2, [BY-.17, BY+.17], lw=5, color=TEAL)
            for dy in np.linspace(-.17, .17, 5):
                ax.plot([x0, x0-.1], [BY+dy, BY+dy-.09], lw=.9, color=TEAL, alpha=.55)
        else:
            tri = plt.Polygon([[x0, BY], [x0-.11, BY-.17], [x0+.11, BY-.17]],
                               closed=True, color=TEAL, fill=(stype_ == "Roller"), zorder=4)
            ax.add_patch(tri)
            if stype_ == "Roller":
                ax.plot(x0, BY-.21, "o", ms=5, color=TEAL, zorder=4)
    cx = 0.0
    for i, sp in enumerate(spans):
        L = sp["L"]
        for ld in span_loads.get(i, []):
            t = ld.get("type"); mg = ld.get("mag", 0)
            if t == "Point":
                a = cx + ld.get("pos", 0)
                ax.annotate("", xy=(a, BY+.04), xytext=(a, LT),
                            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, mutation_scale=12))
                ax.text(a, LT+.06, f"{mg} kN", ha="center", va="bottom", color=RED, fontsize=7.5)
            elif t == "UDL":
                xs = np.linspace(cx, cx+L, 20)
                ax.plot([cx, cx+L], [LT-.02]*2, lw=1.8, color=TEAL)
                for xp in xs:
                    ax.annotate("", xy=(xp, BY+.04), xytext=(xp, LT-.02),
                                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.1, mutation_scale=8))
                ax.text(cx+L/2, LT+.06, f"{mg} kN/m", ha="center", va="bottom", color=TEAL, fontsize=7.5)
            elif t == "UDL-P":
                a = cx+ld.get("pos", 0); b = cx+ld.get("end", 0)
                if b > a:
                    xs = np.linspace(a, b, 14)
                    ax.plot([a, b], [LT-.02]*2, lw=1.8, color=YEL)
                    for xp in xs:
                        ax.annotate("", xy=(xp, BY+.04), xytext=(xp, LT-.02),
                                    arrowprops=dict(arrowstyle="-|>", color=YEL, lw=1.1, mutation_scale=8))
                    ax.text((a+b)/2, LT+.06, f"{mg} kN/m", ha="center", va="bottom", color=YEL, fontsize=7.5)
            elif t == "UVL-P":
                a = cx+ld.get("pos", 0); b = cx+ld.get("end", 0)
                shape = ld.get("shape", "start_zero")
                if b > a:
                    xs = np.linspace(a, b, 18)
                    hh = [((xp-a)/(b-a) if shape == "start_zero" else (b-xp)/(b-a)) * (LT-.06) for xp in xs]
                    ax.fill([a]+list(xs)+[b, a], [BY]+[BY+h for h in hh]+[BY, BY], alpha=.12, color=GRN)
                    ax.plot([a]+list(xs)+[b], [BY]+[BY+h for h in hh]+[BY], lw=1.2, color=GRN, alpha=.75)
                    for xp, h in zip(xs, hh):
                        if h > .04:
                            ax.annotate("", xy=(xp, BY+.03), xytext=(xp, BY+h),
                                        arrowprops=dict(arrowstyle="-|>", color=GRN, lw=.9, mutation_scale=6))
                    ax.text((a+b)/2, BY+max(hh)+.1, f"UVL {mg}kN/m",
                            ha="center", va="bottom", color=GRN, fontsize=7.5)
        ax.text(cx+L/2, BY-.14, f"L={L:.1f}m", ha="center", va="top", color=TXT, fontsize=7)
        cx += L
    ax.set_xlim(-.4, total_L+.4); ax.set_ylim(-.5, LT+.32); ax.axis("off")
    fig.tight_layout(pad=.2)
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ─── SFD/BMD ──────────────────────────────────────────────────
def plot_sfd_bmd(x, v, m, spans):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.5))
    fig.patch.set_facecolor(BG)
    a1.plot(x, v, lw=2, color=TEAL)
    a1.fill_between(x, v, 0, where=(np.array(v) >= 0), alpha=.15, color=GRN)
    a1.fill_between(x, v, 0, where=(np.array(v) < 0),  alpha=.15, color=RED)
    a1.axhline(0, color=GRID, lw=1); _ann(a1, x, v)
    _ax(a1, title="Shear Force Diagram", ylabel="Shear (kN)")
    a2.plot(x, m, lw=2, color=YEL)
    a2.fill_between(x, m, 0, where=(np.array(m) >= 0), alpha=.15, color=YEL)
    a2.fill_between(x, m, 0, where=(np.array(m) < 0),  alpha=.12, color=RED)
    a2.axhline(0, color=GRID, lw=1); a2.invert_yaxis(); _ann(a2, x, m)
    _ax(a2, title="Bending Moment Diagram  (sagging \u2193)", xlabel="Distance (m)", ylabel="Moment (kNm)")
    cx = 0.0
    for sp in spans:
        cx += sp["L"]
        for ax_ in (a1, a2): ax_.axvline(cx, color=GRID, lw=1, ls=":", alpha=.7)
    fig.tight_layout(pad=1.4)
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ─── RC cross-section ─────────────────────────────────────────
def draw_section(bw, D, cover, tension_bars_str, section_type="rectangular",
                 bf=None, hf=None, compression_bars_str=""):
    def pdia(s):
        try: return int(str(s).split("\u00d8")[1].split()[0])
        except: return 16
    def pn(s):
        try: return max(int(str(s).split("\u00d8")[0].strip()), 2)
        except: return 3

    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    if section_type == "T-beam" and bf and hf:
        ax.fill([-bw/2, bw/2, bw/2, -bw/2, -bw/2], [0, 0, D, D, 0],
                facecolor="#1C2E48", edgecolor=TEAL, lw=2, zorder=2)
        ax.fill([-bf/2, bf/2, bf/2, -bf/2, -bf/2], [D, D, D+hf, D+hf, D],
                facecolor="#1C2E48", edgecolor=TEAL, lw=2, zorder=2)
        total_D = D + hf
    else:
        ax.fill([-bw/2, bw/2, bw/2, -bw/2, -bw/2], [0, 0, D, D, 0],
                facecolor="#1C2E48", edgecolor=TEAL, lw=2, zorder=2)
        total_D = D

    lk = cover
    ax.plot([-bw/2+lk, bw/2-lk, bw/2-lk, -bw/2+lk, -bw/2+lk],
            [lk, lk, D-lk, D-lk, lk], color=GRN, lw=2, zorder=3)

    bd = pdia(tension_bars_str); nb = pn(tension_bars_str)
    for xb in np.linspace(-bw/2+cover, bw/2-cover, nb):
        ax.add_patch(plt.Circle((xb, cover+bd/2), bd/2, color=RED, zorder=5))

    if compression_bars_str:
        bdc = pdia(compression_bars_str); nbc = pn(compression_bars_str)
        for xb in np.linspace(-bw/2+cover, bw/2-cover, nbc):
            ax.add_patch(plt.Circle((xb, D-cover-bdc/2), bdc/2, color="#6B9FFF", zorder=5))

    off = cover * 1.8
    ax.annotate("", xy=(bw/2, -off), xytext=(-bw/2, -off),
                arrowprops=dict(arrowstyle="<->", color=TXT, lw=1.2))
    ax.text(0, -off-cover*.8, f"b = {bw:.0f} mm", ha="center", va="top", fontsize=8, color=TXT)

    ax.annotate("", xy=(bw/2+cover*2.5, D), xytext=(bw/2+cover*2.5, 0),
                arrowprops=dict(arrowstyle="<->", color=TXT, lw=1.2))
    ax.text(bw/2+cover*3.5, D/2, f"D = {D:.0f}",
            ha="left", va="center", fontsize=8, color=TXT, rotation=90)

    d_eff = D - cover
    ax.text(bw/2+cover*.8, d_eff/2, f"d={d_eff:.0f}", ha="left", va="center",
            fontsize=7, color=YEL, alpha=.8)

    handles = [
        mpatches.Patch(facecolor="#1C2E48", edgecolor=TEAL, label="Concrete"),
        mlines.Line2D([], [], color=GRN, lw=2, label="Links"),
        mpatches.Patch(color=RED, label=f"Tension: {tension_bars_str}"),
    ]
    if compression_bars_str:
        handles.append(mpatches.Patch(color="#6B9FFF", label=f"Comp: {compression_bars_str}"))
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              facecolor=SURF, edgecolor=GRID, labelcolor=TXT, framealpha=.95)

    ax.set_aspect("equal")
    pad = max(bw, total_D) * .30
    ax.set_xlim(-bw/2-pad, bw/2+pad*2.5)
    ax.set_ylim(-cover*4, total_D+cover*3)
    ax.axis("off")
    ax.set_title("Cross-Section Detail", color=TEAL, fontsize=9, pad=8)
    fig.tight_layout(); return fig


# ─── Frame diagram ────────────────────────────────────────────
def draw_frame(nodes, members, results=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    fig.patch.set_facecolor(BG)
    _ax(ax, title="Portal Frame \u2014 BMD Overlay" if results else "Portal Frame",
        xlabel="x (m)", ylabel="y (m)")
    lbls = ["AB", "BC", "CD"]; cols = [TEAL, YEL, GRN]
    cmap = {0: (0, 1), 1: (1, 2), 2: (2, 3)}
    for idx, mem in enumerate(members):
        ni, nj = cmap[mem["id"]]
        xi, yi = nodes[ni]["x"], nodes[ni]["y"]
        xj, yj = nodes[nj]["x"], nodes[nj]["y"]
        ax.plot([xi, xj], [yi, yj], lw=3.5, color=cols[idx],
                solid_capstyle="round", label=lbls[idx])
        if results and idx in results:
            ms, me = results[idx]; Lm = np.hypot(xj-xi, yj-yi)
            if Lm > 0:
                dx, dy = (xj-xi)/Lm, (yj-yi)/Lm; px, py = -dy, dx
                sc = .32 / max(abs(ms), abs(me), 1e-6)
                ts = np.linspace(0, 1, 30)
                bx = [xi+t*(xj-xi)+(ms*(1-t)+me*t)*sc*px for t in ts]
                by = [yi+t*(yj-yi)+(ms*(1-t)+me*t)*sc*py for t in ts]
                ax.fill([xi]+bx+[xj, xi], [yi]+by+[yj, yi], alpha=.14, color=cols[idx])
                ax.plot(bx, by, lw=1.4, color=cols[idx], alpha=.85)
                for xx, yy, val in [(xi, yi, ms), (xj, yj, me)]:
                    ax.annotate(f"{val:+.2f}", xy=(xx, yy), xytext=(6, 6),
                                textcoords="offset points", fontsize=7.5,
                                color=cols[idx], fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", fc=SURF2, ec=cols[idx], alpha=.8))
    for n in nodes:
        ax.plot(n["x"], n["y"], "o", color="#E8EAF0", ms=9, zorder=5)
        ax.text(n["x"]-.2, n["y"], ["A","B","C","D"][n["id"]],
                color="#E8EAF0", fontsize=10, va="center", ha="right", fontweight="700")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7.5, facecolor=SURF, edgecolor=GRID, labelcolor=TXT)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)


# ─── Beam workings ────────────────────────────────────────────
def beam_workings(n_spans, spans, support_types, span_loads, fems, thetas, sway_corr):
    W = []
    def ln(s=""): W.append(s)
    SEP = "\u2550" * 60
    def h(s): W.extend([SEP, f"  {s}", SEP])

    h("STEP 1 \u2014 FIXED-END MOMENTS")
    ln()
    ln("  Formula:  M_Fab = -wL\u00b2/12  (UDL, near end)")
    ln("            M_Fba = +wL\u00b2/12  (UDL, far end)")
    ln("            M_Fab = -Pab\u00b2/L\u00b2 (Point, near)")
    ln("            M_Fba = +Pa\u00b2b/L\u00b2 (Point, far)")
    ln()
    for i in range(n_spans):
        A = chr(65+i); B = chr(66+i)
        L = spans[i]["L"]
        m1, m2 = fems[i]
        lds = span_loads.get(i, [])
        ln(f"  \u2500\u2500 Span {A}-{B}  (L = {L:.3f} m) \u2500\u2500")
        if not lds:
            ln(f"     No load  \u2192  M_F{A}{B} = 0.000 kNm,  M_F{B}{A} = 0.000 kNm")
        else:
            for ld in lds:
                t = ld["type"]
                if t == "UDL":
                    w = ld["mag"]
                    ln(f"     UDL:  w = {w:.2f} kN/m, L = {L:.3f} m")
                    ln(f"     M_F{A}{B} = -wL\u00b2/12 = -{w:.2f}\u00d7{L:.3f}\u00b2/12 = {-w*L**2/12:.4f} kNm")
                    ln(f"     M_F{B}{A} = +wL\u00b2/12 =  {w*L**2/12:.4f} kNm")
                elif t == "Point":
                    P = ld["mag"]; a = ld["pos"]; b = L-a
                    ln(f"     Point load: P = {P:.2f} kN, a = {a:.3f} m, b = {b:.3f} m")
                    ln(f"     M_F{A}{B} = -Pab\u00b2/L\u00b2 = {-P*a*b**2/L**2:.4f} kNm")
                    ln(f"     M_F{B}{A} = +Pa\u00b2b/L\u00b2 = {P*a**2*b/L**2:.4f} kNm")
                elif t == "UDL-P":
                    ln(f"     Partial UDL: w={ld['mag']:.2f} kN/m, {ld['pos']:.3f}\u2192{ld['end']:.3f} m")
                elif t == "UVL-P":
                    ln(f"     UVL: w_max={ld['mag']:.2f} kN/m, shape={ld.get('shape','')}")
            ln(f"     \u2234 M_F{A}{B} = {m1:.4f} kNm   M_F{B}{A} = {m2:.4f} kNm")
        ln()

    h("STEP 2 \u2014 SLOPE-DEFLECTION EQUATIONS")
    ln()
    ln("  M_ij = M_Fij + (2EI/L)[2\u03b8\u1d35 + \u03b8\u2c7c \u2212 3\u0394/L]")
    ln()
    for i in range(n_spans):
        A = chr(65+i); B = chr(66+i)
        L = spans[i]["L"]; EI = spans[i]["EI"]
        k = 2*EI/L; delta = sway_corr.get(i, 0.0)
        m1, m2 = fems[i]
        ln(f"  \u2500\u2500 Span {A}-{B}:  L={L:.3f}m,  EI={EI:.4f},  2EI/L = {k:.4f} \u2500\u2500")
        if abs(delta) > 1e-9:
            ln(f"     Chord shortening \u0394 = {delta:.6f} m  \u2192  3\u0394/L = {3*delta/L:.8f}")
        ln(f"     M_{A}{B} = {m1:+.4f} + {k:.4f}[2\u03b8_{A} + \u03b8_{B} \u2212 {3*delta/L:.6f}]")
        ln(f"     M_{B}{A} = {m2:+.4f} + {k:.4f}[2\u03b8_{B} + \u03b8_{A} \u2212 {3*delta/L:.6f}]")
        ln()

    h("STEP 3 \u2014 BOUNDARY CONDITIONS & EQUILIBRIUM")
    ln()
    for i in range(n_spans+1):
        bc = "0 (Fixed/pinned)" if support_types[i] in ("Fixed","Cantilever") else "free"
        ln(f"     Joint {chr(65+i)} ({support_types[i]}):  \u03b8_{chr(65+i)} = {bc}")
    ln()
    ln("  Equilibrium: \u03a3M = 0 at each free joint \u2192 solve simultaneous equations")
    ln()

    h("STEP 4 \u2014 SOLVED ROTATIONS")
    ln()
    for i in range(n_spans+1):
        th = thetas[i] if i < len(thetas) else 0.0
        ln(f"     \u03b8_{chr(65+i)} = {th:.10f} rad   [{support_types[i]}]")
    ln()

    h("STEP 5 \u2014 FINAL END MOMENTS  (back-substitution)")
    ln()
    for i in range(n_spans):
        A = chr(65+i); B = chr(66+i)
        L = spans[i]["L"]; EI = spans[i]["EI"]
        k = 2*EI/L; delta = sway_corr.get(i, 0.0)
        m1, m2 = fems[i]; tA = thetas[i]; tB = thetas[i+1]
        M_AB = m1 + k*(2*tA + tB - 3*delta/L)
        M_BA = m2 + k*(2*tB + tA - 3*delta/L)
        ln(f"  \u2500\u2500 Span {A}-{B} \u2500\u2500")
        ln(f"     M_{A}{B} = {m1:+.4f} + {k:.4f}[2\u00d7{tA:.8f} + {tB:.8f} \u2212 {3*delta/L:.8f}]")
        ln(f"          = {M_AB:.4f} kNm")
        ln(f"     M_{B}{A} = {m2:+.4f} + {k:.4f}[2\u00d7{tB:.8f} + {tA:.8f} \u2212 {3*delta/L:.8f}]")
        ln(f"          = {M_BA:.4f} kNm")
        ln()
    return W


# ─── Design workings ──────────────────────────────────────────
def design_workings(beam_system, section_type, L, bw, D, cover,
                    fcu, fy, bf, hf, gk, qk, wu, Mu, Vu, flex_zones, shear, defl):
    W = []
    def ln(s=""): W.append(s)
    SEP = "\u2550" * 60
    def h(s): W.extend([SEP, f"  {s}", SEP])
    d = D - cover
    fl = list(flex_zones.values())[0]

    h("STEP 1 \u2014 ULTIMATE DESIGN LOAD")
    ln(f"  wu = 1.4gk + 1.6qk = 1.4\u00d7{gk:.3f} + 1.6\u00d7{qk:.3f} = {wu:.4f} kN/m")
    ln()

    h("STEP 2 \u2014 DESIGN MOMENTS & SHEAR")
    if "Simply" in beam_system:
        ln(f"  Mu = wuL\u00b2/8 = {wu:.4f}\u00d7{L:.3f}\u00b2/8 = {wu*L**2/8:.4f} kNm")
        ln(f"  Vu = wuL/2  = {wu:.4f}\u00d7{L:.3f}/2  = {wu*L/2:.4f} kN")
    else:
        ln(f"  Governing sagging Mu = {Mu:.4f} kNm  (from analysis)")
        ln(f"  Governing shear Vu = {Vu:.4f} kN  (from analysis)")
    ln()

    h("STEP 3 \u2014 FLEXURAL DESIGN  (BS 8110 Cl. 3.4.4)")
    ln(f"  b = {bw:.0f} mm,  D = {D:.0f} mm,  cover = {cover:.0f} mm,  d = {d:.0f} mm")
    ln(f"  fcu = {fcu:.0f} N/mm\u00b2,  fy = {fy:.0f} N/mm\u00b2")
    ln()
    Mu_Nmm = Mu * 1e6
    K = Mu_Nmm / (fcu * bw * d**2)
    K_lim = 0.156
    z = min(d * (0.5 + (0.25 - K/0.9)**0.5), 0.95*d)
    ln(f"  K = Mu / (fcu\u00b7b\u00b7d\u00b2) = {Mu_Nmm:.0f} / ({fcu:.0f}\u00d7{bw:.0f}\u00d7{d:.0f}\u00b2)")
    ln(f"    = {K:.6f}   (K_lim = {K_lim})")
    ln(f"  \u2192 {'Singly reinforced' if K <= K_lim else 'DOUBLY reinforced'}")
    ln()
    ln(f"  Lever arm:  z = 0.5d[1 + \u221a(1 \u2212 K/0.225)]  \u2264 0.95d  = {z:.2f} mm")
    ln()
    ln(f"  As_req = Mu / (0.87\u00b7fy\u00b7z) = {Mu_Nmm:.0f} / (0.87\u00d7{fy:.0f}\u00d7{z:.2f})")
    ln(f"         = {fl['As_req']:.2f} mm\u00b2")
    ln(f"  Provide: {fl['tension_bars']}  \u2192  As_prov = {fl['As_prov']:.0f} mm\u00b2")
    ln()

    h("STEP 4 \u2014 SHEAR DESIGN  (BS 8110 Cl. 3.4.5)")
    sh = shear
    ln(f"  v = Vu/(b\u00b7d) = {Vu*1000:.0f} / ({bw:.0f}\u00d7{d:.0f}) = {sh['v']:.4f} N/mm\u00b2")
    ln(f"  vc (Table 3.8) = {sh['vc']:.4f} N/mm\u00b2")
    ln(f"  \u2192 {sh['status']}  |  Links: {sh['links']}")
    ln()

    h("STEP 5 \u2014 DEFLECTION CHECK  (BS 8110 Cl. 3.4.6)")
    basic = 20 if "Simply" in beam_system else 26
    mf = min(2.0, fl['As_prov'] / max(fl['As_req'], 1e-9))
    ln(f"  Basic span/depth ratio (Table 3.9) = {basic}")
    ln(f"  Modification factor = min(2.0, As_prov/As_req) = min(2.0, {mf:.4f}) = {mf:.4f}")
    ln(f"  Allowable L/d = {basic} \u00d7 {mf:.4f} = {defl['allowable']:.3f}")
    ln(f"  Actual L/d    = {L*1000:.0f}/{d:.0f} = {defl['actual']:.3f}")
    ln(f"  \u2192 {defl['status']}")
    ln()
    return W


# ─── Sidebar ──────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("""<div style='text-align:center;padding:.5rem 0 1.5rem'>
            <div style='font-size:2.2rem'>🏗️</div>
            <div style='font-family:Syne,sans-serif;font-weight:800;color:#E8EAF0;font-size:1.15rem'>StructSolve</div>
            <div style='color:#8B92A8;font-size:.68rem;margin-top:2px'>BS 8110-1:1997</div>
        </div>""", unsafe_allow_html=True)
        module = st.radio("SELECT MODULE",
            ["🔩 Beam Analysis", "🏛️ Frame Analysis", "🧱 RC Design"])
        st.markdown("---")
        st.markdown("""<div style='color:#8B92A8;font-size:.7rem;line-height:1.9'>
        <b style='color:#4ECDC4'>METHODS</b><br>
        ↳ Slope Deflection Method<br>
        ↳ BS 8110 RC Design<br>&nbsp;
        · Flexure (Cl. 3.4.4)<br>&nbsp;
        · Shear (Cl. 3.4.5)<br>&nbsp;
        · Deflection (Cl. 3.4.6)<br><br>
        <b style='color:#4ECDC4'>UNITS</b><br>
        ↳ Length: m / mm<br>
        ↳ Force: kN / kNm<br>
        ↳ Stress: N/mm²</div>""", unsafe_allow_html=True)
    return module


# ══════════════════════════════════════════════════════════════
# BEAM PAGE
# ══════════════════════════════════════════════════════════════
def beam_page():
    st.markdown("""<div class='main-header'>
        <h1>🔩 Beam Analysis <span class='badge'>SLOPE DEFLECTION</span></h1>
        <p>Continuous beam · Full slope-deflection workings · SFD &amp; BMD · RC design · PDF export</p>
    </div>""", unsafe_allow_html=True)

    for k, v in {"analysis_done": False, "run_design": False,
                 "internal_actions": {}, "spans_cache": []}.items():
        if k not in st.session_state: st.session_state[k] = v

    # Migrate stale loads (old schema: pos/end → pos_s/pos_e)
    if "beam_loads" in st.session_state:
        fixed = []
        for ld in st.session_state.beam_loads:
            if "pos_s" not in ld:
                ld2 = dict(ld)
                ld2["pos_s"] = ld.get("pos", 0.0)
                ld2["pos_e"] = ld.get("end", ld.get("pos", 0.0))
                ld2.setdefault("mag_end", ld.get("mag", 0.0))
                fixed.append(ld2)
            else:
                fixed.append(ld)
        st.session_state.beam_loads = fixed

    # Config
    st.markdown("<div class='section-label'>Beam Configuration</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        n_joints = st.number_input("Number of joints", 2, 10, 3, key="n_joints")
        n_spans = n_joints - 1
    with c2:
        st.markdown("<small style='color:#8B92A8'>Support type at each joint:</small>", unsafe_allow_html=True)
        support_types = []
        sc = st.columns(n_joints)
        for i in range(n_joints):
            support_types.append(sc[i].selectbox(f"**{chr(65+i)}**",
                ["Roller", "Fixed", "Cantilever"], key=f"s_{i}"))

    # Settlement / Rotation
    with st.expander("\u2699\ufe0f  Settlement & Prescribed Rotation (optional)", expanded=False):
        se1, se2 = st.columns(2)
        settlements = {}; rotations = {}
        for i in range(n_joints):
            settlements[i] = se1.number_input(f"Settlement {chr(65+i)} (mm)", value=0.0, step=1.0, key=f"sett_{i}")
            rotations[i]   = se2.number_input(f"Rotation {chr(65+i)} (rad)",  value=0.0, format="%.5f", key=f"rot_{i}")

    # Span geometry
    st.markdown("<div class='section-label'>Span Geometry & Stiffness</div>", unsafe_allow_html=True)
    ei_mode = st.radio("EI input mode",
        ["Relative EI (dimensionless)", "Actual E & I (E in GPa, I \u00d710\u2076 mm\u2074)"],
        horizontal=True, key="ei_mode")
    spans = []; spcols = st.columns(n_spans)
    for i in range(n_spans):
        with spcols[i]:
            st.markdown(f"<small style='color:{TEAL};font-weight:600'>Span {chr(65+i)}\u2013{chr(66+i)}</small>",
                        unsafe_allow_html=True)
            L = st.number_input("Length (m)", .1, 50.0, 5.0, key=f"L_{i}")
            if ei_mode.startswith("Relative"):
                EI = st.number_input("Relative EI", .001, 1e6, 1.0, format="%.3f", key=f"EI_{i}")
            else:
                Eg = st.number_input("E (GPa)",        1.0, 500.0, 200.0, key=f"E_{i}")
                Im = st.number_input("I (\u00d710\u2076 mm\u2074)", .001, 1e6,  40.0, format="%.3f", key=f"I_{i}")
                EI = (Eg*1e6) * (Im*1e6) / 1e9
                st.markdown(f"<small style='color:#8B92A8'>EI = {EI:.2f} kNm\u00b2</small>", unsafe_allow_html=True)
            spans.append({"L": L, "EI": EI})

    # Load manager
    st.markdown("#### Applied Loads")
    if "beam_loads" not in st.session_state:
        st.session_state.beam_loads = [
            {"type": "UDL",   "span": 0, "mag": 10.0, "mag_end": 10.0, "pos_s": 0.0, "pos_e": 5.0},
            {"type": "POINT", "span": 0, "mag": 25.0, "mag_end":  0.0, "pos_s": 2.5, "pos_e": 2.5},
        ]
    with st.expander("\u2795 Add Load", expanded=False):
        a1, a2, _ = st.columns(3)
        ntype = a1.selectbox("Type", ["UDL", "UVL", "POINT"], key="nl_type")
        nspan = a2.number_input("Span #", 1, n_spans, 1, step=1, key="nl_span") - 1
        Lsel = spans[nspan]["L"] if nspan < len(spans) else 5.0
        if ntype in ["UDL", "UVL"]:
            b1, b2 = st.columns(2)
            nps  = b1.number_input("Start (m)", 0.0, float(Lsel), 0.0,         step=.25, key="nl_ps")
            npe  = b2.number_input("End (m)",   0.0, float(Lsel), float(Lsel), step=.25, key="nl_pe")
            m1c, m2c = st.columns(2)
            nmag  = m1c.number_input("Intensity start (kN/m)", value=10.0, step=1.0, key="nl_mag")
            nmage = m2c.number_input("Intensity end (kN/m)",
                value=10.0 if ntype == "UDL" else 0.0, step=1.0, key="nl_mage",
                disabled=(ntype == "UDL"))
            if ntype == "UDL": nmage = nmag
            npos_e = npe
        else:
            nps    = st.number_input("Position (m)", 0.0, float(Lsel), min(2.0, Lsel), step=.1, key="nl_ps")
            npe    = nps; nmag = st.number_input("Force (kN)", value=25.0, step=1.0, key="nl_mag")
            nmage  = 0.0; npos_e = nps
        if st.button("Add Load"):
            st.session_state.beam_loads.append(
                {"type": ntype, "span": nspan, "mag": nmag, "mag_end": nmage,
                 "pos_s": nps, "pos_e": npos_e})
            st.rerun()

    if st.session_state.beam_loads:
        hd = st.columns([1, .8, 1.1, 1.1, 1.1, 1.1, .4])
        for lbl, col in zip(["Type","Span","Start(m)","End(m)","Mag(s)","Mag(e)",""], hd):
            col.markdown(f"<small style='color:{TXT}'>{lbl}</small>", unsafe_allow_html=True)
        for idx, ld in enumerate(st.session_state.beam_loads):
            c = st.columns([1, .8, 1.1, 1.1, 1.1, 1.1, .4])
            col = {"UDL": TEAL, "UVL": YEL, "POINT": RED}.get(ld["type"], "#E8EAF0")
            c[0].markdown(f"<span style='color:{col};font-weight:600'>`{ld['type']}`</span>",
                          unsafe_allow_html=True)
            c[1].markdown(f"Span {ld['span']+1}")
            c[2].markdown(f"{ld['pos_s']:.2f}")
            c[3].markdown(f"{ld['pos_e']:.2f}" if ld["type"] != "POINT" else "\u2014")
            c[4].markdown(f"{ld['mag']:.1f} kN/m" if ld["type"] != "POINT" else f"{ld['mag']:.1f} kN")
            c[5].markdown(f"{ld['mag_end']:.1f}" if ld["type"] == "UVL" else
                          ("\u2014" if ld["type"] == "POINT" else f"{ld['mag']:.1f}"))
            if c[6].button("\u00d7", key=f"dl_{idx}"):
                st.session_state.beam_loads.pop(idx); st.rerun()
    else:
        st.markdown("<div class='result-card muted' style='color:#8B92A8'>No loads \u2014 add above.</div>",
                    unsafe_allow_html=True)

    # Translate UI loads → solver format
    span_loads = {i: [] for i in range(n_spans)}; load_ok = True
    for ld in st.session_state.beam_loads:
        sp = ld["span"]
        if sp >= n_spans: continue
        Lsp = spans[sp]["L"]
        if ld["type"] == "UDL":
            if ld["pos_s"] == 0.0 and abs(ld["pos_e"] - Lsp) < 1e-6:
                span_loads[sp].append({"type": "UDL", "mag": ld["mag"], "pos": 0.0, "end": None, "shape": None})
            else:
                if ld["pos_e"] <= ld["pos_s"]: st.error(f"UDL span {sp+1}: end>start"); load_ok = False; continue
                span_loads[sp].append({"type": "UDL-P", "mag": ld["mag"], "pos": ld["pos_s"], "end": ld["pos_e"], "shape": None})
        elif ld["type"] == "UVL":
            if ld["pos_e"] <= ld["pos_s"]: st.error(f"UVL span {sp+1}: end>start"); load_ok = False; continue
            ms2, me2 = ld["mag"], ld["mag_end"]
            shape = "end_zero" if ms2 >= me2 else "start_zero"
            span_loads[sp].append({"type": "UVL-P", "mag": max(ms2, me2), "pos": ld["pos_s"], "end": ld["pos_e"], "shape": shape})
        elif ld["type"] == "POINT":
            span_loads[sp].append({"type": "Point", "mag": ld["mag"], "pos": ld["pos_s"], "end": None, "shape": None})

    # Tabs
    t_vis, t_res = st.tabs(["👁️  Beam Preview", "📊  Analysis Results"])
    with t_vis:
        draw_beam_system(spans, support_types, span_loads)

    with t_res:
        ca, cb, _ = st.columns([1, 1, 2])
        run_a  = ca.button("🔍  Analyse Only",     use_container_width=True)
        run_ad = cb.button("🧮  Analyse + Design", use_container_width=True)
        if run_ad: run_a = True

        if run_a and load_ok:
            try:
                sett_m = {k: v/1000 for k, v in settlements.items()}
                sway_corr = {i: sett_m.get(i, 0) - sett_m.get(i+1, 0) for i in range(n_spans)}
                prescribed = {k: v for k, v in rotations.items() if v != 0.0}

                thetas, fems = BeamSolver.solve_continuous_beam(
                    n_joints, spans, support_types, span_loads,
                    sway_corrections=sway_corr, prescribed_rotations=prescribed)

                all_x, all_v, all_m = [], [], []
                cx = 0.0; sup_mom = {}; sup_shr = {}; sp_res = {}; int_act = {}
                for i in range(n_spans):
                    L2 = spans[i]["L"]; EI = spans[i]["EI"]
                    delta = sway_corr.get(i, 0.0)
                    m_ab = fems[i][0] + (2*EI/L2)*(2*thetas[i]   + thetas[i+1] - 3*delta/L2)
                    m_ba = fems[i][1] + (2*EI/L2)*(2*thetas[i+1] + thetas[i]   - 3*delta/L2)
                    if i == 0: sup_mom["A"] = m_ab
                    sup_mom[chr(66+i)] = m_ba
                    x2, v2, m2 = BeamSolver.get_diagram_data(
                        {"id": i, "L": L2}, m_ab, m_ba,
                        [{"member": i, **ld} for ld in span_loads[i]])
                    m2 = -np.array(m2); v2 = np.array(v2)
                    all_x.extend(x2+cx); all_v.extend(v2); all_m.extend(m2); cx += L2
                    sp_res[f"{chr(65+i)}-{chr(66+i)}"] = {"M_sag": float(max(m2)), "V_max": float(max(abs(v2)))}
                    int_act[f"{chr(65+i)}-{chr(66+i)}"] = {"M_start": m_ab, "M_end": m_ba}
                sup_shr = {"left": abs(all_v[0]), "right": abs(all_v[-1])}
                wk = beam_workings(n_spans, spans, support_types, span_loads, fems, thetas, sway_corr)
                st.session_state.update({
                    "analysis_done": True,
                    "all_x": np.array(all_x), "all_v": np.array(all_v), "all_m": np.array(all_m),
                    "spans_cache": spans, "span_results": sp_res,
                    "support_moments": sup_mom, "support_shears": sup_shr,
                    "run_design": run_ad, "internal_actions": int_act,
                    "beam_workings": wk, "beam_fems": fems, "beam_thetas": list(thetas),
                    "beam_sway_corr": sway_corr, "beam_settlements": dict(settlements),
                    "beam_rotations": dict(rotations), "beam_span_loads": dict(span_loads),
                    "beam_support_types": list(support_types),
                })
            except Exception as e:
                import traceback; st.error(f"Analysis error: {e}")
                with st.expander("Details"): st.code(traceback.format_exc())
                return

        if not st.session_state.analysis_done:
            st.markdown("""<div style='display:flex;align-items:center;justify-content:center;height:260px;
                border:2px dashed #2A2E40;border-radius:12px;color:#8B92A8;text-align:center;padding:2rem'>
                <div><div style='font-size:2.5rem;margin-bottom:.8rem'>📊</div>
                Click <b>Analyse Only</b> or <b>Analyse + Design</b></div></div>""",
                unsafe_allow_html=True); return

        x = st.session_state.all_x; v = st.session_state.all_v; m = st.session_state.all_m
        spans_c = st.session_state.spans_cache; sr = st.session_state.span_results

        cards = "".join(
            f"<div class='metric-box'><div class='val'>{sr[k]['M_sag']:.2f}</div>"
            f"<div class='lbl'>M sag {k} (kNm)</div></div>" for k in sr)
        max_hog = max(abs(v2) for v2 in st.session_state.support_moments.values())
        max_shr = max(st.session_state.support_shears.values())
        cards += (f"<div class='metric-box'><div class='val red'>{max_hog:.2f}</div>"
                  f"<div class='lbl'>Max Hogging (kNm)</div></div>"
                  f"<div class='metric-box'><div class='val yel'>{max_shr:.2f}</div>"
                  f"<div class='lbl'>Max Shear (kN)</div></div>")
        st.markdown(f"<div class='metric-grid'>{cards}</div>", unsafe_allow_html=True)
        plot_sfd_bmd(x, v, m, spans_c)

        # Support actions table
        st.markdown("<div class='section-label'>Support Actions</div>", unsafe_allow_html=True)
        ext_shr = {"A": v[0], chr(65+len(spans_c)): v[-1]}
        ints = st.session_state.internal_actions; sup_rows = []
        for i in range(len(spans_c)+1):
            nm = chr(65+i)
            row = {"Support": nm, "V ext (kN)": round(ext_shr.get(nm, 0.0), 3),
                   "M left (kNm)": "\u2014", "M right (kNm)": "\u2014"}
            if i > 0:
                k2 = f"{chr(64+i)}-{nm}"
                if k2 in ints: row["M left (kNm)"] = round(ints[k2]["M_end"], 3)
            if i < len(spans_c):
                k2 = f"{nm}-{chr(66+i)}"
                if k2 in ints: row["M right (kNm)"] = round(ints[k2]["M_start"], 3)
            sup_rows.append(row)
        support_df = pd.DataFrame(sup_rows)
        st.dataframe(support_df, use_container_width=True, hide_index=True)

        # Workings
        wk = st.session_state.get("beam_workings", [])
        with st.expander("📐  Full Slope-Deflection Workings", expanded=False):
            st.markdown("<div class='workings-box'>" + "\n".join(wk) + "</div>", unsafe_allow_html=True)

        # PDF export
        sett    = st.session_state.get("beam_settlements", {})
        rots    = st.session_state.get("beam_rotations", {})
        fems_s  = st.session_state.get("beam_fems", [])
        thetas_s = st.session_state.get("beam_thetas", [])
        sway_c  = st.session_state.get("beam_sway_corr", {})
        sl_s    = st.session_state.get("beam_span_loads", {})
        stype_s = st.session_state.get("beam_support_types", support_types)
        try:
            pdf_a = export_beam_analysis_pdf(
                spans_c, stype_s, sl_s, sett, rots, thetas_s, fems_s, sway_c,
                ints, support_df, x, v, m, wk)
            st.download_button("📄  Export Analysis PDF",
                data=pdf_a, file_name="beam_analysis.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF error: {e}")

        if not st.session_state.run_design: return

        # RC Design
        st.markdown("<div class='section-label'>RC Design  (BS 8110)</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        bw    = d1.number_input("Web width b (mm)",     min_value=1.0, value=300.0)
        D     = d1.number_input("Overall depth D (mm)", min_value=1.0, value=550.0)
        cover = d2.number_input("Cover (mm)",            min_value=1.0, value=40.0)
        fcu   = d2.number_input("fcu (N/mm\u00b2)",     min_value=1.0, value=30.0)
        fy    = d3.number_input("fy (N/mm\u00b2)",      min_value=1.0, value=460.0)
        if bw <= 0 or D <= cover: st.warning("Invalid section parameters."); return
        d = D - cover; dp = cover
        Mu_sag = max(sr[k]["M_sag"] for k in sr)
        Mu_hog = max(abs(v2) for v2 in st.session_state.support_moments.values())
        des = auto_design_from_beam_analysis(
            moments={"sagging": Mu_sag, "hogging": Mu_hog},
            shears=st.session_state.support_shears, b=bw, d=d, dp=dp, fcu=fcu, fy=fy)
        fl = des["flexure"]["sagging"]; sh = des["shear"]
        tot_L = sum(sp["L"] for sp in spans_c)
        defl = deflection_check(L=tot_L, d=d, beam_type="continuous",
                                fy=fy, As_req=fl["As_req"], As_prov=fl["As_prov"])
        dc = "grn" if defl["status"] == "PASS" else "red"

        st.markdown(f"""<div class='metric-grid'>
            <div class='metric-box'><div class='val'>{Mu_sag:.2f}</div><div class='lbl'>Mu sag (kNm)</div></div>
            <div class='metric-box'><div class='val red'>{Mu_hog:.2f}</div><div class='lbl'>Mu hog (kNm)</div></div>
            <div class='metric-box'><div class='val'>{fl['As_req']:.0f}</div><div class='lbl'>As req (mm\u00b2)</div></div>
            <div class='metric-box'><div class='val grn'>{fl['As_prov']:.0f}</div><div class='lbl'>As prov (mm\u00b2)</div></div>
            <div class='metric-box'><div class='val yel'>{sh['v']:.3f}</div><div class='lbl'>v (N/mm\u00b2)</div></div>
            <div class='metric-box'><div class='val yel'>{sh['vc']:.3f}</div><div class='lbl'>vc (N/mm\u00b2)</div></div>
            <div class='metric-box'><div class='val {dc}'>{defl['actual']:.1f}/{defl['allowable']:.1f}</div><div class='lbl'>L/d act/lim</div></div>
            <div class='metric-box'><div class='val {dc}'>{defl['status']}</div><div class='lbl'>Deflection</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-card'>Tension bars: <b style='color:{TEAL}'>{fl['tension_bars']}</b></div>"
            f"<div class='result-card warning'>Shear links: <b style='color:{YEL}'>{sh['links']}</b> | {sh['status']}</div>",
            unsafe_allow_html=True)

        # Cross-section
        st.markdown("<div class='section-label'>Cross-Section Detail</div>", unsafe_allow_html=True)
        comp_str = fl.get("compression_bars", "") if fl.get("type") == "doubly" else ""
        fig_s = draw_section(bw, D, cover, fl["tension_bars"], compression_bars_str=comp_str)
        sc1, _ = st.columns([1, 1])
        with sc1: st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)

        dw = design_workings("Continuous Beam", "Rectangular", tot_L, bw, D, cover,
                             fcu, fy, None, None, 0, 0, 0, Mu_sag,
                             st.session_state.support_shears.get("left", 0),
                             {"Mid-span": fl}, sh, defl)
        with st.expander("📐  Full Design Workings  (BS 8110)", expanded=False):
            st.markdown("<div class='workings-box'>" + "\n".join(dw) + "</div>", unsafe_allow_html=True)

        try:
            pdf_d = export_beam_design_pdf(
                spans_c, stype_s, sl_s, sett, rots, thetas_s, fems_s, sway_c,
                ints, support_df, x, v, m, wk,
                bw, D, cover, fcu, fy, {"Mid-span": fl}, sh, defl, dw)
            st.download_button("📄  Export Analysis + Design PDF",
                data=pdf_d, file_name="beam_analysis_design.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF error: {e}")


# ══════════════════════════════════════════════════════════════
# FRAME PAGE
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# GENERAL FRAME DIAGRAM RENDERER
# ══════════════════════════════════════════════════════════════
def _draw_frame_general(nodes, members, nmap, results=None, loads=None):
    """Draw any plane frame with optional BMD overlay."""
    if not nodes or not members:
        return
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=TXT, labelsize=7)
    ax.grid(True, color=GRID, lw=0.4, ls="--", alpha=0.5)
    ax.set_xlabel("x (m)", fontsize=8, color=TXT)
    ax.set_ylabel("y (m)", fontsize=8, color=TXT)

    colours = [TEAL, YEL, GRN, RED, "#BB88FF", "#FF9966", "#66BBFF", "#FF66BB"]

    # BMD scale
    if results:
        all_m = [abs(v) for tup in results.values() for v in tup if tup]
        max_m = max(all_m) if all_m else 1.0
        all_x = [n["x"] for n in nodes]
        all_y = [n["y"] for n in nodes]
        span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)
        bmd_sc = 0.22 * span / max(max_m, 1e-6)

    for idx, mem in enumerate(members):
        c = colours[idx % len(colours)]
        ni = nmap[mem["ni"]]
        nj = nmap[mem["nj"]]
        xi, yi = ni["x"], ni["y"]
        xj, yj = nj["x"], nj["y"]
        ax.plot([xi, xj], [yi, yj], lw=3.5, color=c,
                solid_capstyle="round", label=mem["label"], zorder=3)

        # Load arrows
        if loads:
            ml = [ld for ld in loads if ld["member_id"] == mem["id"]]
            Lm = float(np.hypot(xj - xi, yj - yi))
            if Lm > 0:
                ex, ey = (xj - xi) / Lm, (yj - yi) / Lm
                px, py = -ey, ex   # perpendicular (upward for horizontal)
                arrow_sc = 0.06 * span
                for ld in ml:
                    t = ld["type"]
                    if t == "UDL":
                        for frac in np.linspace(0.1, 0.9, 8):
                            ax2 = xi + frac * (xj - xi)
                            ay2 = yi + frac * (yj - yi)
                            ax.annotate("", xy=(ax2, ay2),
                                        xytext=(ax2 + px * arrow_sc, ay2 + py * arrow_sc),
                                        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2))
                        # label
                        mx2 = (xi + xj) / 2 + px * arrow_sc * 1.6
                        my2 = (yi + yj) / 2 + py * arrow_sc * 1.6
                        ax.text(mx2, my2, f"{ld['mag']} kN/m",
                                color=TEAL, fontsize=7, ha="center")
                    elif t == "Point":
                        a = ld["pos"]
                        ax2 = xi + (a / Lm) * (xj - xi)
                        ay2 = yi + (a / Lm) * (yj - yi)
                        ax.annotate("", xy=(ax2, ay2),
                                    xytext=(ax2 + px * arrow_sc * 1.4, ay2 + py * arrow_sc * 1.4),
                                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                                    mutation_scale=12))
                        ax.text(ax2 + px * arrow_sc * 1.7, ay2 + py * arrow_sc * 1.7,
                                f"{ld['mag']} kN", color=RED, fontsize=7, ha="center")

        # BMD overlay — use actual computed moment distribution (parabolic for UDL)
        if results and mem["id"] in results:
            Mij, Mji = results[mem["id"]]
            Lm = float(np.hypot(xj - xi, yj - yi))
            if Lm > 0:
                dx2, dy2 = (xj - xi) / Lm, (yj - yi) / Lm
                px2, py2 = -dy2, dx2
                # Use actual moment distribution from solver
                try:
                    from frame_solver import GeneralFrameSolver
                    mem_loads_for_diag = [ld for ld in (loads or []) if ld["member_id"] == mem["id"]]
                    xs_d, _, M_d = GeneralFrameSolver.get_member_diagram(
                        mem, nmap, Mij, Mji, mem_loads_for_diag, n_pts=60)
                    # Convert local x to global coords + BMD offset
                    bx = [xi + (xs_d[k]/Lm)*(xj-xi) + M_d[k]*bmd_sc*px2 for k in range(len(xs_d))]
                    by = [yi + (xs_d[k]/Lm)*(yj-yi) + M_d[k]*bmd_sc*py2 for k in range(len(xs_d))]
                except Exception:
                    # Fallback: linear interpolation
                    ts = np.linspace(0, 1, 40)
                    bx = [xi+t*(xj-xi)+(Mij*(1-t)+Mji*t)*bmd_sc*px2 for t in ts]
                    by = [yi+t*(yj-yi)+(Mij*(1-t)+Mji*t)*bmd_sc*py2 for t in ts]
                ax.fill([xi] + bx + [xj, xi],
                        [yi] + by + [yj, yi],
                        alpha=0.15, color=c, zorder=2)
                ax.plot(bx, by, lw=1.5, color=c, alpha=0.9, zorder=4)
                for xx, yy, val in [(xi, yi, Mij), (xj, yj, Mji)]:
                    ax.annotate(f"{val:+.2f}",
                                xy=(xx, yy), xytext=(6, 6),
                                textcoords="offset points",
                                fontsize=7.5, color=c, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc=SURF2, ec=c, alpha=0.9))

    # Support symbols
    all_x2 = [n["x"] for n in nodes]
    all_y2 = [n["y"] for n in nodes]
    ss = max(max(all_x2) - min(all_x2), max(all_y2) - min(all_y2), 1.0) * 0.045

    for n in nodes:
        x0, y0 = n["x"], n["y"]
        sup = n.get("support", "Free")
        if sup == "Fixed":
            ax.plot([x0] * 2, [y0 - ss, y0 + ss], lw=6, color="#C0C6D4", zorder=5)
            for dy3 in np.linspace(-ss, ss, 5):
                ax.plot([x0, x0 - ss * 0.6], [y0 + dy3, y0 + dy3 - ss * 0.4],
                        lw=0.9, color="#C0C6D4", alpha=0.6)
        elif sup == "Pinned":
            tri = plt.Polygon([[x0, y0], [x0 - ss, y0 - ss * 1.3],
                                [x0 + ss, y0 - ss * 1.3]],
                               closed=True, facecolor=TEAL, zorder=5)
            ax.add_patch(tri)
        elif sup == "Roller":
            tri = plt.Polygon([[x0, y0], [x0 - ss, y0 - ss * 1.3],
                                [x0 + ss, y0 - ss * 1.3]],
                               closed=True, facecolor="none",
                               edgecolor=TEAL, lw=1.8, zorder=5)
            ax.add_patch(tri)
            ax.plot(x0, y0 - ss * 1.6, "o", ms=6, color=TEAL, zorder=5)
        ax.plot(x0, y0, "o", color="#E8EAF0", ms=9, zorder=6)
        ax.text(x0, y0, f"  {n.get('label', '')}", fontsize=10,
                fontweight="700", color="#E8EAF0", va="center", zorder=7)

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7.5,
              facecolor=SURF, edgecolor=GRID, labelcolor=TXT)
    ax.set_title("BMD Overlay" if results else "Frame Preview",
                 color=TEAL, fontsize=9, pad=6)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# FRAME PAGE  — General Frame Solver
# ══════════════════════════════════════════════════════════════
def frame_page():
    from frame_solver import GeneralFrameSolver
    import pandas as pd

    st.markdown("""<div class='main-header'>
        <h1>🏛️ Frame Analysis <span class='badge'>GENERAL SOLVER</span></h1>
        <p>Any topology · Sway &amp; non-sway · Settlements · Applied moments ·
        Modified stiffness · Step-by-step workings · PDF export</p>
    </div>""", unsafe_allow_html=True)

    # ── Session-state defaults ────────────────────────────────
    if "fr_nodes" not in st.session_state:
        st.session_state.fr_nodes = [
            {"id": 0, "x": 0.0, "y": 0.0, "support": "Fixed",  "label": "A"},
            {"id": 1, "x": 0.0, "y": 4.0, "support": "Free",   "label": "B"},
            {"id": 2, "x": 6.0, "y": 4.0, "support": "Free",   "label": "C"},
            {"id": 3, "x": 6.0, "y": 0.0, "support": "Fixed",  "label": "D"},
        ]
    if "fr_members" not in st.session_state:
        st.session_state.fr_members = [
            {"id": 0, "ni": 0, "nj": 1, "EI": 1000.0, "label": "AB"},
            {"id": 1, "ni": 1, "nj": 2, "EI": 2000.0, "label": "BC"},
            {"id": 2, "ni": 2, "nj": 3, "EI": 1000.0, "label": "CD"},
        ]
    if "fr_loads"  not in st.session_state:
        st.session_state.fr_loads = [
            {"member_id": 1, "type": "UDL", "mag": 20.0,
             "pos": 0.0, "end": 0.0, "shape": ""},
        ]
    if "fr_jmom"   not in st.session_state: st.session_state.fr_jmom  = {}
    if "fr_sett"   not in st.session_state: st.session_state.fr_sett  = {}
    if "fr_result" not in st.session_state: st.session_state.fr_result = None

    # ══════════════════════════════════════════════════════════
    # QUICK TEMPLATES
    # ══════════════════════════════════════════════════════════
    st.markdown("<div class='section-label'>Quick Templates</div>",
                unsafe_allow_html=True)

    tpl_names = [
        "Portal — fixed",
        "Portal — pinned",
        "L-Frame",
        "4\u00b0 Indet.",
        "Multi-bay",
        "Propped cant.",
        "Cantilever arm",
    ]
    tc = st.columns(len(tpl_names))
    chosen = None
    for i, (col, lbl) in enumerate(zip(tc, tpl_names)):
        if col.button(lbl, key=f"tpl_{i}", use_container_width=True):
            chosen = i

    if chosen == 0:   # fixed portal
        st.session_state.fr_nodes = [
            {"id":0,"x":0,"y":0,"support":"Fixed","label":"A"},
            {"id":1,"x":0,"y":4,"support":"Free", "label":"B"},
            {"id":2,"x":6,"y":4,"support":"Free", "label":"C"},
            {"id":3,"x":6,"y":0,"support":"Fixed","label":"D"},
        ]
        st.session_state.fr_members = [
            {"id":0,"ni":0,"nj":1,"EI":1000,"label":"AB"},
            {"id":1,"ni":1,"nj":2,"EI":2000,"label":"BC"},
            {"id":2,"ni":2,"nj":3,"EI":1000,"label":"CD"},
        ]
        st.session_state.fr_loads  = [{"member_id":1,"type":"UDL","mag":20,"pos":0,"end":0,"shape":""}]
        st.session_state.fr_jmom   = {}
        st.session_state.fr_sett   = {}
        st.session_state.fr_result = None
        st.rerun()

    elif chosen == 1:   # pinned portal
        st.session_state.fr_nodes = [
            {"id":0,"x":0,"y":0,"support":"Pinned","label":"A"},
            {"id":1,"x":0,"y":4,"support":"Free",  "label":"B"},
            {"id":2,"x":6,"y":4,"support":"Free",  "label":"C"},
            {"id":3,"x":6,"y":0,"support":"Pinned","label":"D"},
        ]
        st.session_state.fr_members = [
            {"id":0,"ni":0,"nj":1,"EI":1000,"label":"AB"},
            {"id":1,"ni":1,"nj":2,"EI":2000,"label":"BC"},
            {"id":2,"ni":2,"nj":3,"EI":1000,"label":"CD"},
        ]
        st.session_state.fr_loads  = [{"member_id":1,"type":"UDL","mag":20,"pos":0,"end":0,"shape":""}]
        st.session_state.fr_jmom   = {}
        st.session_state.fr_sett   = {}
        st.session_state.fr_result = None
        st.rerun()

    elif chosen == 2:   # L-Frame (Image 1: A-B-C with column B-E)
        st.session_state.fr_nodes = [
            {"id":0,"x":0,"y":4,"support":"Fixed", "label":"A"},
            {"id":1,"x":4,"y":4,"support":"Free",  "label":"B"},
            {"id":2,"x":6,"y":4,"support":"Free",  "label":"C"},
            {"id":3,"x":4,"y":0,"support":"Pinned","label":"E"},
        ]
        st.session_state.fr_members = [
            {"id":0,"ni":0,"nj":1,"EI":2,"label":"AB"},
            {"id":1,"ni":1,"nj":2,"EI":2,"label":"BC"},
            {"id":2,"ni":3,"nj":1,"EI":1,"label":"EB"},
        ]
        st.session_state.fr_loads = [
            {"member_id":0,"type":"UDL",  "mag":10,"pos":0.0,"end":4.0,"shape":""},
            {"member_id":1,"type":"Point","mag":10,"pos":2.0,"end":0.0,"shape":""},
            {"member_id":2,"type":"Point","mag":20,"pos":2.0,"end":0.0,"shape":""},
        ]
        st.session_state.fr_jmom   = {}
        st.session_state.fr_sett   = {}
        st.session_state.fr_result = None
        st.rerun()

    elif chosen == 3:   # 4° indeterminate (Image 2)
        EI = 270000.0
        st.session_state.fr_nodes = [
            {"id":0,"x":3, "y":0, "support":"Fixed", "label":"A"},
            {"id":1,"x":0, "y":6, "support":"Fixed", "label":"B"},
            {"id":2,"x":3, "y":6, "support":"Free",  "label":"C"},
            {"id":3,"x":8, "y":6, "support":"Pinned","label":"D"},
            {"id":4,"x":10,"y":6, "support":"Free",  "label":"E"},
        ]
        st.session_state.fr_members = [
            {"id":0,"ni":0,"nj":2,"EI":EI,"label":"AC"},
            {"id":1,"ni":1,"nj":2,"EI":EI,"label":"BC"},
            {"id":2,"ni":2,"nj":3,"EI":EI,"label":"CD"},
            {"id":3,"ni":3,"nj":4,"EI":EI,"label":"DE"},
        ]
        st.session_state.fr_loads = [
            {"member_id":2,"type":"UDL","mag":30,"pos":0,"end":0,"shape":""},
            {"member_id":3,"type":"UDL","mag":30,"pos":0,"end":0,"shape":""},
        ]
        st.session_state.fr_jmom   = {2: 150.0}
        st.session_state.fr_sett   = {0: {"dx": 0.0, "dy": -0.010}}
        st.session_state.fr_result = None
        st.rerun()

    elif chosen == 4:   # multi-bay
        st.session_state.fr_nodes = [
            {"id":0,"x":0, "y":0,"support":"Fixed","label":"A"},
            {"id":1,"x":0, "y":4,"support":"Free", "label":"B"},
            {"id":2,"x":5, "y":4,"support":"Free", "label":"C"},
            {"id":3,"x":10,"y":4,"support":"Free", "label":"D"},
            {"id":4,"x":5, "y":0,"support":"Fixed","label":"E"},
            {"id":5,"x":10,"y":0,"support":"Fixed","label":"F"},
        ]
        st.session_state.fr_members = [
            {"id":0,"ni":0,"nj":1,"EI":1000,"label":"AB"},
            {"id":1,"ni":1,"nj":2,"EI":2000,"label":"BC"},
            {"id":2,"ni":2,"nj":3,"EI":2000,"label":"CD"},
            {"id":3,"ni":4,"nj":2,"EI":1000,"label":"EC"},
            {"id":4,"ni":5,"nj":3,"EI":1000,"label":"FD"},
        ]
        st.session_state.fr_loads = [
            {"member_id":1,"type":"UDL","mag":20,"pos":0,"end":0,"shape":""},
            {"member_id":2,"type":"UDL","mag":20,"pos":0,"end":0,"shape":""},
        ]
        st.session_state.fr_jmom   = {}
        st.session_state.fr_sett   = {}
        st.session_state.fr_result = None
        st.rerun()

    elif chosen == 5:   # propped cantilever / L-frame with free end
        st.session_state.fr_nodes = [
            {"id":0,"x":0,"y":0,"support":"Fixed","label":"A"},
            {"id":1,"x":0,"y":5,"support":"Free", "label":"B"},
            {"id":2,"x":8,"y":5,"support":"Free", "label":"C"},
            {"id":3,"x":8,"y":0,"support":"Roller","label":"D"},
        ]
        st.session_state.fr_members = [
            {"id":0,"ni":0,"nj":1,"EI":2000,"label":"AB"},
            {"id":1,"ni":1,"nj":2,"EI":3000,"label":"BC"},
            {"id":2,"ni":2,"nj":3,"EI":2000,"label":"CD"},
        ]
        st.session_state.fr_loads = [
            {"member_id":1,"type":"UDL",  "mag":15,"pos":0,"end":0,"shape":""},
            {"member_id":0,"type":"Point","mag":30,"pos":2.5,"end":0,"shape":""},
        ]
        st.session_state.fr_jmom   = {}
        st.session_state.fr_sett   = {}
        st.session_state.fr_result = None
        st.rerun()

    elif chosen == 6:   # Cantilever arm frame (Example 4 from image)
        # A(fixed wall) — B(junction) — C(free cantilever tip)
        #                              |
        #                              E(pinned base)
        # AB: L=4m, EI=2I; BC: L=2m, EI=2I; BE: L=4m, EI=I
        # Loads: UDL 10kN/m on AB; 10kN point at C; 20kN horizontal on BE at 2m from B
        st.session_state.fr_nodes = [
            {"id":0,"x":0.0,"y":4.0,"support":"Fixed",  "label":"A"},
            {"id":1,"x":4.0,"y":4.0,"support":"Free",   "label":"B"},
            {"id":2,"x":6.0,"y":4.0,"support":"Free",   "label":"C"},
            {"id":3,"x":4.0,"y":0.0,"support":"Pinned", "label":"E"},
        ]
        st.session_state.fr_members = [
            {"id":0,"ni":0,"nj":1,"EI":2.0,"label":"AB"},
            {"id":1,"ni":1,"nj":2,"EI":2.0,"label":"BC"},
            {"id":2,"ni":3,"nj":1,"EI":1.0,"label":"EB"},
        ]
        st.session_state.fr_loads = [
            {"member_id":0,"type":"UDL",  "mag":10.0,"pos":0.0,"end":4.0,"shape":""},
            {"member_id":1,"type":"Point","mag":10.0,"pos":2.0,"end":0.0,"shape":""},
            {"member_id":2,"type":"Point","mag":20.0,"pos":2.0,"end":0.0,"shape":""},
        ]
        st.session_state.fr_jmom   = {}
        st.session_state.fr_sett   = {}
        st.session_state.fr_result = None
        st.rerun()
    nodes   = st.session_state.fr_nodes
    members = st.session_state.fr_members
    loads   = st.session_state.fr_loads
    nmap    = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]
    mem_ids  = [m["id"] for m in members]

    # ══════════════════════════════════════════════════════════
    # NODE EDITOR
    # ══════════════════════════════════════════════════════════
    st.markdown("<div class='section-label'>Nodes</div>", unsafe_allow_html=True)
    st.caption("Each node needs a label, coordinates (m), and support type. "
               "Δy = vertical settlement in mm (positive = upward).")

    hdr = st.columns([0.6, 1.1, 1.1, 1.6, 1.2, 0.45])
    for h, col in zip(["Label", "x (m)", "y (m)", "Support", "Δy (mm)", ""], hdr):
        col.markdown(f"<small style='color:{TXT};font-weight:600'>{h}</small>",
                     unsafe_allow_html=True)

    new_nodes = []
    remove_node = None
    for ni, n in enumerate(nodes):
        rc = st.columns([0.6, 1.1, 1.1, 1.6, 1.2, 0.45])
        lbl = rc[0].text_input("L", value=n["label"],
                                key=f"nl_{n['id']}", label_visibility="collapsed")
        xv  = rc[1].number_input("x", value=float(n["x"]),
                                  key=f"nx_{n['id']}", label_visibility="collapsed", step=0.5)
        yv  = rc[2].number_input("y", value=float(n["y"]),
                                  key=f"ny_{n['id']}", label_visibility="collapsed", step=0.5)
        opts = ["Fixed", "Pinned", "Roller", "Free"]
        sup  = rc[3].selectbox("S", opts,
                                index=opts.index(n["support"]) if n["support"] in opts else 0,
                                key=f"ns_{n['id']}", label_visibility="collapsed")
        dy_mm = float(st.session_state.fr_sett.get(n["id"], {}).get("dy", 0.0)) * 1000
        ndy   = rc[4].number_input("dy", value=dy_mm,
                                    key=f"ndy_{n['id']}", label_visibility="collapsed", step=1.0)
        if abs(ndy) > 1e-9:
            st.session_state.fr_sett[n["id"]] = {
                "dx": st.session_state.fr_sett.get(n["id"], {}).get("dx", 0.0),
                "dy": ndy / 1000.0}
        elif n["id"] in st.session_state.fr_sett:
            del st.session_state.fr_sett[n["id"]]
        if rc[5].button("✕", key=f"nd_{n['id']}") and len(nodes) > 2:
            remove_node = n["id"]
        else:
            new_nodes.append({"id": n["id"], "x": xv, "y": yv,
                               "support": sup, "label": lbl})

    if remove_node is not None:
        # Also remove members referencing this node
        st.session_state.fr_members = [m for m in members
                                         if m["ni"] != remove_node and m["nj"] != remove_node]
        st.session_state.fr_nodes = new_nodes
        st.rerun()

    st.session_state.fr_nodes = new_nodes

    if st.button("＋ Add Node", key="add_node"):
        used_ids = [n["id"] for n in new_nodes]
        new_id = max(used_ids) + 1 if used_ids else 0
        alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        nlbl = alph[new_id % 26] if new_id < 26 else f"N{new_id}"
        new_nodes.append({"id": new_id, "x": 0.0, "y": 0.0,
                           "support": "Free", "label": nlbl})
        st.session_state.fr_nodes = new_nodes
        st.rerun()

    nodes   = st.session_state.fr_nodes
    nmap    = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]

    # ══════════════════════════════════════════════════════════
    # MEMBER EDITOR
    # ══════════════════════════════════════════════════════════
    st.markdown("<div class='section-label'>Members</div>", unsafe_allow_html=True)

    hdr2 = st.columns([0.7, 1.2, 1.2, 1.6, 1.0, 0.45])
    for h, col in zip(["Label", "Near node", "Far node", "EI (kNm²)", "L (m)", ""], hdr2):
        col.markdown(f"<small style='color:{TXT};font-weight:600'>{h}</small>",
                     unsafe_allow_html=True)

    new_members = []
    remove_mem  = None
    node_fmt    = {n["id"]: n["label"] for n in nodes}

    for mi, mem in enumerate(members):
        mc = st.columns([0.7, 1.2, 1.2, 1.6, 1.0, 0.45])
        mlbl  = mc[0].text_input("ML", value=mem["label"],
                                  key=f"mlab_{mem['id']}", label_visibility="collapsed")
        ni_def = mem["ni"] if mem["ni"] in node_ids else node_ids[0]
        nj_def = mem["nj"] if mem["nj"] in node_ids else node_ids[-1]
        ni_sel = mc[1].selectbox("ni", options=node_ids,
                                  format_func=lambda x: node_fmt.get(x, str(x)),
                                  index=node_ids.index(ni_def),
                                  key=f"mni_{mem['id']}", label_visibility="collapsed")
        nj_sel = mc[2].selectbox("nj", options=node_ids,
                                  format_func=lambda x: node_fmt.get(x, str(x)),
                                  index=node_ids.index(nj_def),
                                  key=f"mnj_{mem['id']}", label_visibility="collapsed")
        ei_val = mc[3].number_input("EI", value=float(mem["EI"]), min_value=0.001,
                                     key=f"mei_{mem['id']}", label_visibility="collapsed",
                                     format="%.1f")
        ni_n = nmap.get(ni_sel); nj_n = nmap.get(nj_sel)
        Lm = float(np.hypot(nj_n["x"] - ni_n["x"], nj_n["y"] - ni_n["y"])) if ni_n and nj_n else 0.0
        mc[4].markdown(f"<div style='padding:.55rem 0;color:{TEAL};font-weight:600'>"
                       f"{Lm:.3f} m</div>", unsafe_allow_html=True)
        if mc[5].button("✕", key=f"md_{mem['id']}") and len(members) > 1:
            remove_mem = mem["id"]
        else:
            new_members.append({"id": mem["id"], "ni": ni_sel, "nj": nj_sel,
                                 "EI": ei_val, "label": mlbl})

    if remove_mem is not None:
        st.session_state.fr_members = new_members
        st.session_state.fr_loads   = [ld for ld in loads if ld["member_id"] != remove_mem]
        st.rerun()

    st.session_state.fr_members = new_members

    if st.button("＋ Add Member", key="add_member"):
        used_mids = [m["id"] for m in new_members]
        new_mid = max(used_mids) + 1 if used_mids else 0
        new_members.append({"id": new_mid, "ni": node_ids[0], "nj": node_ids[-1],
                             "EI": 1000.0, "label": f"M{new_mid}"})
        st.session_state.fr_members = new_members
        st.rerun()

    members = st.session_state.fr_members
    mem_ids = [m["id"] for m in members]
    mem_fmt = {m["id"]: m["label"] for m in members}

    # ══════════════════════════════════════════════════════════
    # LOAD EDITOR
    # ══════════════════════════════════════════════════════════
    st.markdown("<div class='section-label'>Loads on Members</div>",
                unsafe_allow_html=True)

    with st.expander("＋ Add Load", expanded=False):
        la, lb, lc = st.columns(3)
        l_mid  = la.selectbox("On member", options=mem_ids,
                               format_func=lambda x: mem_fmt.get(x, str(x)),
                               key="nl_mid_f")
        l_type = lb.selectbox("Load type",
                               ["UDL", "Point", "UDL-P", "UVL-P", "Moment"],
                               key="nl_type_f2")
        l_mag  = lc.number_input("Magnitude (kN or kNm)", value=10.0,
                                  key="nl_mag_f2", step=5.0)
        mem_sel = next((m for m in members if m["id"] == l_mid), None)
        ni_n2   = nmap.get(mem_sel["ni"]) if mem_sel else None
        nj_n2   = nmap.get(mem_sel["nj"]) if mem_sel else None
        L_sel   = float(np.hypot(nj_n2["x"] - ni_n2["x"], nj_n2["y"] - ni_n2["y"])) \
                  if ni_n2 and nj_n2 else 5.0
        l_pos = l_end = 0.0; l_shape = ""
        if l_type in ("Point", "UDL-P", "UVL-P", "Moment"):
            pd1, pd2 = st.columns(2)
            l_pos = pd1.number_input("Position (m from near end)", 0.0, float(L_sel),
                                      min(1.0, L_sel / 2), key="nl_pos_f2")
            if l_type in ("UDL-P", "UVL-P"):
                l_end = pd2.number_input("End position (m)", float(l_pos),
                                          float(L_sel), float(L_sel), key="nl_end_f2")
            if l_type == "UVL-P":
                l_shape = st.radio("UVL shape", ["start_zero", "end_zero"],
                                    horizontal=True, key="nl_shape_f2")
        if st.button("Add Load", key="add_load_f2"):
            loads.append({"member_id": l_mid, "type": l_type, "mag": l_mag,
                           "pos": l_pos, "end": l_end, "shape": l_shape})
            st.session_state.fr_loads = loads
            st.rerun()

    loads = st.session_state.fr_loads
    if loads:
        lhdr = st.columns([1.3, 1.0, 1.3, 0.9, 0.9, 1.0, 0.4])
        for h, col in zip(["Member", "Type", "Mag", "Pos(m)", "End(m)", "Shape", ""], lhdr):
            col.markdown(f"<small style='color:{TXT}'>{h}</small>", unsafe_allow_html=True)
        remove_load = None
        for idx, ld in enumerate(loads):
            lrc = st.columns([1.3, 1.0, 1.3, 0.9, 0.9, 1.0, 0.4])
            type_col = {"UDL": TEAL, "Point": RED, "UDL-P": YEL,
                        "UVL-P": GRN, "Moment": "#BB88FF"}.get(ld["type"], "#E8EAF0")
            lrc[0].markdown(f"<span style='color:{TEAL}'>{mem_fmt.get(ld['member_id'], str(ld['member_id']))}</span>",
                             unsafe_allow_html=True)
            lrc[1].markdown(f"<span style='color:{type_col}'>`{ld['type']}`</span>",
                             unsafe_allow_html=True)
            lrc[2].markdown(f"{ld['mag']:.2f}")
            lrc[3].markdown(f"{ld['pos']:.2f}")
            lrc[4].markdown(f"{ld.get('end', 0):.2f}"
                             if ld["type"] in ("UDL-P", "UVL-P") else "—")
            lrc[5].markdown(ld.get("shape", "") or "—")
            if lrc[6].button("✕", key=f"ld_f_{idx}"):
                remove_load = idx
        if remove_load is not None:
            loads.pop(remove_load)
            st.session_state.fr_loads = loads
            st.rerun()
    else:
        st.markdown("<div class='result-card muted' style='color:#8B92A8'>"
                    "No loads added — use the form above.</div>",
                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # APPLIED JOINT MOMENTS
    # ══════════════════════════════════════════════════════════
    with st.expander("⊕  Applied Joint Moments", expanded=bool(st.session_state.fr_jmom)):
        st.caption("+ve = anticlockwise (kNm). Leave 0 if none.")
        jm_cols = st.columns(min(len(nodes), 6))
        for ni, n in enumerate(nodes[:6]):
            cur = float(st.session_state.fr_jmom.get(n["id"], 0.0))
            nv  = jm_cols[ni].number_input(f"M at {n['label']}",
                                             value=cur, step=10.0, key=f"jm_f_{n['id']}")
            if abs(nv) > 1e-9:
                st.session_state.fr_jmom[n["id"]] = nv
            elif n["id"] in st.session_state.fr_jmom:
                del st.session_state.fr_jmom[n["id"]]

    # ══════════════════════════════════════════════════════════
    # ANALYSIS OPTIONS
    # ══════════════════════════════════════════════════════════
    st.markdown("<div class='section-label'>Analysis Options</div>",
                unsafe_allow_html=True)
    ao1, _ao2 = st.columns(2)
    sway_choice = ao1.radio("Sway",
                             ["Auto-detect sway DOFs", "Non-sway (Δ = 0 enforced)"],
                             horizontal=True, key="fr_sway_radio")
    sway = "Auto" in sway_choice

    # ══════════════════════════════════════════════════════════
    # PREVIEW + RESULTS TABS
    # ══════════════════════════════════════════════════════════
    tp, tr = st.tabs(["👁️  Frame Preview", "📊  Analysis Results"])

    with tp:
        _draw_frame_general(nodes, members, nmap, results=None, loads=loads)

    with tr:
        if st.button("🚀  Run Frame Analysis", use_container_width=False, key="run_frame_f"):
            try:
                moments_out, unknowns, workings = GeneralFrameSolver.solve(
                    nodes, members, loads,
                    joint_moments=st.session_state.fr_jmom,
                    settlements=st.session_state.fr_sett,
                    sway=sway)
                st.session_state.fr_result = {
                    "moments":  moments_out,
                    "unknowns": unknowns,
                    "workings": workings,
                }
            except Exception as e:
                import traceback
                st.error(f"Solver error: {e}")
                with st.expander("Traceback"): st.code(traceback.format_exc())

        res = st.session_state.fr_result
        if res is None:
            st.markdown("""<div style='display:flex;align-items:center;justify-content:center;
                height:220px;border:2px dashed #2A2E40;border-radius:12px;
                color:#8B92A8;text-align:center;padding:2rem'>
                <div><div style='font-size:2.5rem;margin-bottom:.8rem'>🏛️</div>
                Click <b>Run Frame Analysis</b> above</div></div>""",
                unsafe_allow_html=True)
            return

        moments_out = res["moments"]
        unknowns    = res["unknowns"]
        workings    = res["workings"]

        # ── Metric cards for unknowns ──────────────────────────
        cards = ""
        for k, v in unknowns.items():
            unit    = "rad" if k.startswith("theta") else "m"
            clscls  = "yel" if k.startswith("delta") else ""
            display = k.replace("theta_", "θ_").replace("delta_", "Δ_")
            cards += (f"<div class='metric-box'>"
                      f"<div class='val {clscls}' style='font-size:.9rem'>{v:.8f}</div>"
                      f"<div class='lbl'>{display} ({unit})</div></div>")
        st.markdown(f"<div class='metric-grid'>{cards}</div>",
                    unsafe_allow_html=True)

        # ── End moment table ───────────────────────────────────
        st.markdown("<div class='section-label'>Member End Moments</div>",
                    unsafe_allow_html=True)
        rows = []
        for mem in members:
            mid  = mem["id"]
            ni_n = nmap[mem["ni"]]; nj_n = nmap[mem["nj"]]
            li   = ni_n.get("label", "?"); lj = nj_n.get("label", "?")
            Mij, Mji = moments_out.get(mid, (0.0, 0.0))
            rows.append({
                "Member": mem["label"],
                f"M_{li}{lj} (kNm)": f"{Mij:+.4f}",
                f"M_{lj}{li} (kNm)": f"{Mji:+.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Equilibrium check summary
        chk = workings.get("check", "")
        chk_ok = "\u26a0\ufe0f" not in chk
        flag   = "\u2705 All joints in equilibrium" if chk_ok else "\u26a0\ufe0f Check equilibrium"
        st.markdown(f"<div class='result-card {'muted' if chk_ok else 'danger'}'>"
                    f"{flag}</div>", unsafe_allow_html=True)

        # ── BMD overlay diagram ────────────────────────────────
        st.markdown("<div class='section-label'>BMD Overlay</div>",
                    unsafe_allow_html=True)
        _draw_frame_general(nodes, members, nmap,
                             results=moments_out, loads=loads)

        # ── Full step-by-step workings ─────────────────────────
        wk_all = "\n\n".join(workings.get(k, "")
                              for k in ["fem", "sd_eqs", "equil",
                                        "solution", "final", "check"])
        with st.expander("📐  Full Slope-Deflection Workings (all 6 steps)",
                          expanded=False):
            st.markdown("<div class='workings-box'>" + wk_all + "</div>",
                        unsafe_allow_html=True)

        # ── PDF export ─────────────────────────────────────────
        try:
            from pdf_export import export_frame_pdf_general
            pdf_f = export_frame_pdf_general(
                nodes, members, loads,
                moments_out, unknowns, workings,
                title="General Frame Analysis")
            st.download_button(
                "📄  Export Frame Analysis PDF",
                data=pdf_f,
                file_name="frame_analysis.pdf",
                mime="application/pdf",
                key="dl_frame_pdf")
        except Exception as e:
            import traceback
            st.warning(f"PDF export error: {e}")
            with st.expander("PDF error details"):
                st.code(traceback.format_exc())

def design_page():
    st.markdown("""<div class='main-header'>
        <h1>🧱 RC Beam Design <span class='badge'>BS 8110</span></h1>
        <p>Flexure (Cl. 3.4.4) · Shear (Cl. 3.4.5) · Deflection check · Rectangular &amp; T-sections</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Beam System</div>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    bsys  = dc1.selectbox("System", ["Simply Supported Beam", "Continuous Beam"])
    stype = dc2.selectbox("Section", ["Rectangular Beam", "T-Beam"])

    st.markdown("<div class='section-label'>Geometry</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    L     = g1.number_input("Span L (m)",           .1, 50.0, 6.0)
    bw    = g1.number_input("Web width b (mm)",      1.0, 2000.0, 300.0)
    D     = g1.number_input("Overall depth D (mm)", 1.0, 3000.0, 550.0)
    cover = g1.number_input("Cover (mm)",            1.0, 200.0, 40.0)
    bf = hf = None
    if stype == "T-Beam":
        bf = g2.number_input("Flange width bf (mm)",      1.0, 5000.0, 1200.0)
        hf = g2.number_input("Flange thickness hf (mm)", 1.0, 1000.0, 150.0)
    d = D - cover; dp = cover

    st.markdown("<div class='section-label'>Materials</div>", unsafe_allow_html=True)
    m1c, m2c = st.columns(2)
    fcu = m1c.number_input("fcu (N/mm\u00b2)", 1.0, 100.0, 30.0)
    fy  = m2c.number_input("fy (N/mm\u00b2)",  1.0, 700.0, 460.0)

    st.markdown("<div class='section-label'>Characteristic Loading</div>", unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    gk = l1.number_input("Dead load gk (kN/m)", 0.0, 500.0, 10.0)
    qk = l2.number_input("Live load qk (kN/m)", 0.0, 500.0, 5.0)
    wu = 1.4*gk + 1.6*qk
    st.markdown(f"<div class='result-card'>wu = 1.4({gk:.2f}) + 1.6({qk:.2f}) = "
                f"<b style='color:{TEAL}'>{wu:.3f} kN/m</b></div>", unsafe_allow_html=True)

    if not st.button("🧮  Design Beam (BS 8110)", use_container_width=False):
        st.markdown("""<div style='display:flex;align-items:center;justify-content:center;height:200px;
            border:2px dashed #2A2E40;border-radius:12px;color:#8B92A8;text-align:center;padding:2rem'>
            <div><div style='font-size:2.5rem;margin-bottom:.8rem'>🧱</div>
            Click <b>Design Beam</b></div></div>""", unsafe_allow_html=True)
        return

    if bsys == "Simply Supported Beam":
        Mu = wu*L**2/8; Vu = wu*L/2
        fl = (beam_flexural_design(Mu=Mu, b=bw, d=d, dp=dp, fcu=fcu, fy=fy)
              if stype == "Rectangular Beam"
              else beam_flexural_design_T(Mu=Mu, bf=bf, hf=hf, bw=bw, d=d, dp=dp, fcu=fcu, fy=fy))
        sh = beam_shear_design(Vu=Vu, b=bw, d=d, fcu=fcu, As=fl["As_req"])
        defl = deflection_check(L=L, d=d, beam_type="simply_supported",
                                fy=fy, As_req=fl["As_req"], As_prov=fl["As_prov"])
        flex_zones = {"Mid-span": fl}
    else:
        st.markdown("<div class='result-card muted'>Enter governing moments &amp; shears from analysis.</div>",
                    unsafe_allow_html=True)
        ai1, ai2 = st.columns(2)
        Mu_AB = ai1.number_input("Sagging M \u2014 AB (kNm)", min_value=0.0)
        Mu_BC = ai1.number_input("Sagging M \u2014 BC (kNm)", min_value=0.0)
        Mu_B  = ai2.number_input("Hogging M at B (kNm)",       min_value=0.0)
        Vu_1  = ai2.number_input("Shear at B left (kN)",       min_value=0.0)
        Vu_2  = ai2.number_input("Shear at B right (kN)",      min_value=0.0)
        Mu = max(Mu_AB, Mu_BC); Vu = max(Vu_1, Vu_2)
        res = design_continuous_beam(
            moments={"sagging": {"AB": Mu_AB, "BC": Mu_BC}, "hogging": {"B": Mu_B}},
            shears={"B_left": Vu_1, "B_right": Vu_2},
            section={"type": "T-beam" if stype == "T-Beam" else "rectangular",
                     "bf": bf, "hf": hf, "bw": bw, "d": d, "dp": dp},
            materials={"fcu": fcu, "fy": fy})
        flex_zones = res["flexure"]; sh = res["shear"]
        fl = list(flex_zones.values())[0]
        defl = deflection_check(L=L, d=d, beam_type="continuous", fy=fy,
            As_req=max(v["As_req"] for v in flex_zones.values()),
            As_prov=max(v["As_prov"] for v in flex_zones.values()))

    dc = "grn" if defl["status"] == "PASS" else "red"
    st.markdown(f"""<div class='metric-grid'>
        <div class='metric-box'><div class='val'>{Mu:.2f}</div><div class='lbl'>Mu (kNm)</div></div>
        <div class='metric-box'><div class='val'>{fl['As_req']:.0f}</div><div class='lbl'>As req (mm\u00b2)</div></div>
        <div class='metric-box'><div class='val grn'>{fl['As_prov']:.0f}</div><div class='lbl'>As prov (mm\u00b2)</div></div>
        <div class='metric-box'><div class='val yel'>{sh['v']:.3f}</div><div class='lbl'>v (N/mm\u00b2)</div></div>
        <div class='metric-box'><div class='val yel'>{sh['vc']:.3f}</div><div class='lbl'>vc (N/mm\u00b2)</div></div>
        <div class='metric-box'><div class='val {dc}'>{defl['actual']:.1f}/{defl['allowable']:.1f}</div><div class='lbl'>L/d act/lim</div></div>
        <div class='metric-box'><div class='val {dc}'>{defl['status']}</div><div class='lbl'>Deflection</div></div>
    </div>""", unsafe_allow_html=True)

    for zone, data in flex_zones.items():
        is_d = data.get("type") == "doubly"
        extra = (f" | Asc={data.get('Asc_prov',0):.0f} mm\u00b2 | {data.get('compression_bars','')}"
                 if is_d else "")
        st.markdown(
            f"<div class='result-card {'warning' if is_d else ''}'>"
            f"<b>{zone}</b> \u2014 {'DOUBLY' if is_d else 'SINGLY'} REINFORCED<br>"
            f"As req = <b style='color:{TEAL}'>{data['As_req']:.0f} mm\u00b2</b> | "
            f"As prov = <b style='color:{GRN}'>{data['As_prov']:.0f} mm\u00b2</b> | "
            f"Bars = <b style='color:{GRN}'>{data['tension_bars']}</b>{extra}</div>",
            unsafe_allow_html=True)

    defl_card_cls = "danger" if defl["status"] != "PASS" else ""
    defl_col      = "#6BCB77" if defl["status"] == "PASS" else RED
    st.markdown(
        f"<div class='result-card warning'>Shear: <b style='color:{YEL}'>{sh['status']}</b> | "
        f"v={sh['v']:.3f} | vc={sh['vc']:.3f} | Links: <b style='color:{YEL}'>{sh['links']}</b></div>"
        f"<div class='result-card {defl_card_cls}'>"
        f"Deflection: <b style='color:{defl_col}'>{defl['status']}</b> | "
        f"L/d = {defl['actual']:.2f}  (limit {defl['allowable']:.2f})</div>",
        unsafe_allow_html=True)

    # Cross-section
    st.markdown("<div class='section-label'>Cross-Section Detail</div>", unsafe_allow_html=True)
    comp_str = fl.get("compression_bars", "") if fl.get("type") == "doubly" else ""
    cs_type = "T-beam" if stype == "T-Beam" else "rectangular"
    fig_s = draw_section(bw, D, cover, fl["tension_bars"],
                         section_type=cs_type, bf=bf, hf=hf, compression_bars_str=comp_str)
    cs1, _ = st.columns([1, 1])
    with cs1: st.pyplot(fig_s, use_container_width=True)
    plt.close(fig_s)

    dw = design_workings(bsys, stype, L, bw, D, cover, fcu, fy, bf, hf, gk, qk, wu,
                         Mu, Vu, flex_zones, sh, defl)
    with st.expander("📐  Full Design Workings  (BS 8110)", expanded=False):
        st.markdown("<div class='workings-box'>" + "\n".join(dw) + "</div>", unsafe_allow_html=True)

    try:
        pdf_rc = export_rc_design_pdf(bsys, stype, L, bw, D, cover, fcu, fy, bf, hf,
                                      gk, qk, wu, flex_zones, sh, defl, dw)
        st.download_button("📄  Export Design PDF",
            data=pdf_rc, file_name="rc_design.pdf", mime="application/pdf")
    except Exception as e:
        st.warning(f"PDF error: {e}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    module = _sidebar()
    if   "Beam"  in module: beam_page()
    elif "Frame" in module: frame_page()
    elif "RC"    in module: design_page()

if __name__ == "__main__":
    main()
