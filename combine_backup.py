"""
StructSolve — Structural Analysis Suite
BS 8110 · Slope Deflection Method
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from beam_solver import BeamSolver
from frame_solver import FrameSolver
from bs8110 import (
    beam_flexural_design,
    beam_shear_design,
    auto_design_from_beam_analysis,
    deflection_check,
    format_design_summary,
    beam_flexural_design_T,
    design_continuous_beam,
)
from pdf_export import export_analysis_pdf, export_analysis_design_pdf

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StructSolve | BS 8110",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME — matches previous RC Solver project
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace !important; }

.stApp { background-color: #0F1117; }
section[data-testid="stSidebar"] { background-color: #1A1D27 !important; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* ── Page header ─────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1A1D27 0%, #22263A 100%);
    border: 1px solid #2A2E40;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4ECDC4, #FF6B6B, #FFE66D);
}
.main-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: 2rem !important;
    color: #E8EAF0 !important;
    margin: 0 !important;
    letter-spacing: -1px;
}
.main-header p { color: #8B92A8; font-size: 0.82rem; margin: 0.35rem 0 0 0; }
.badge {
    display: inline-block;
    background: #4ECDC4;
    color: #0F1117;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: 1px;
}

/* ── Cards ───────────────────────────────────────────────── */
.calc-card {
    background: #1A1D27;
    border: 1px solid #2A2E40;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #8B92A8;
    line-height: 1.6;
}
.result-card {
    background: #22263A;
    border-left: 3px solid #4ECDC4;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.45rem 0;
    font-size: 0.83rem;
}
.result-card.warning { border-left-color: #FFE66D; }
.result-card.danger  { border-left-color: #FF6B6B; }
.result-card.muted   { border-left-color: #2A2E40; }

/* ── Section label ───────────────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4ECDC4;
    border-bottom: 1px solid #2A2E40;
    padding-bottom: 0.4rem;
    margin: 1.4rem 0 0.8rem;
}

/* ── Metric grid ─────────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}
.metric-box {
    background: #22263A;
    border: 1px solid #2A2E40;
    border-radius: 8px;
    padding: 0.85rem;
    text-align: center;
}
.metric-box .val { font-size: 1.3rem; font-weight: 600; color: #4ECDC4; }
.metric-box .val.red { color: #FF6B6B; }
.metric-box .val.yel { color: #FFE66D; }
.metric-box .val.grn { color: #6BCB77; }
.metric-box .lbl { font-size: 0.68rem; color: #8B92A8; margin-top: 3px; }

/* ── Load row ────────────────────────────────────────────── */
.load-row {
    background: #1A1D27;
    border: 1px solid #2A2E40;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.78rem;
    color: #C0C6D4;
}
.load-type-udl  { color: #4ECDC4; font-weight: 600; }
.load-type-uvl  { color: #FFE66D; font-weight: 600; }
.load-type-pt   { color: #FF6B6B; font-weight: 600; }

/* ── Calc steps ──────────────────────────────────────────── */
.calc-steps {
    background: #0A0C12;
    border: 1px solid #2A2E40;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.74rem;
    line-height: 1.85;
    color: #8B92A8;
    white-space: pre-wrap;
    max-height: 420px;
    overflow-y: auto;
}

/* ── Streamlit overrides ─────────────────────────────────── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label {
    color: #8B92A8 !important;
    font-size: 0.78rem !important;
}
div[data-testid="stNumberInput"] input {
    background: #22263A !important;
    border: 1px solid #2A2E40 !important;
    color: #E8EAF0 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #22263A !important;
    border: 1px solid #2A2E40 !important;
    color: #E8EAF0 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4ECDC4, #45B7D1) !important;
    color: #0F1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(78,205,196,0.3) !important;
}
div[data-testid="stTabs"] [role="tab"] {
    color: #8B92A8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #4ECDC4; }
.stExpander { border: 1px solid #2A2E40 !important; border-radius: 8px !important; }
div[data-testid="stExpander"] summary { color: #8B92A8 !important; font-size: 0.82rem !important; }
div[data-testid="stDataFrame"] table th {
    background: #0F1117 !important; color: #8B92A8 !important; font-size: 0.74rem !important;
}
div[data-testid="stDataFrame"] table td { color: #C0C6D4 !important; font-size: 0.78rem !important; }
.stAlert { border-radius: 8px !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────
BG   = "#0F1117"
SURF = "#1A1D27"
SURF2= "#22263A"
GRID = "#2A2E40"
TXT  = "#8B92A8"
TEAL = "#4ECDC4"
RED  = "#FF6B6B"
YEL  = "#FFE66D"
GRN  = "#6BCB77"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURF2)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TEAL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.6, ls="--", alpha=0.7)
    if title:  ax.set_title(title, fontsize=9, fontfamily="monospace", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, fontfamily="monospace")
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, fontfamily="monospace")

def _annotate_extremes(ax, x, y):
    im = int(np.argmax(y)); iv = int(np.argmin(y))
    for idx, c in [(im, GRN), (iv, RED)]:
        ax.plot(x[idx], y[idx], "o", color=c, ms=5, zorder=5)
        ax.annotate(f"{y[idx]:.2f}", xy=(x[idx], y[idx]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color=c, fontfamily="monospace")


# ─────────────────────────────────────────────────────────────
# BEAM VISUALISER
# ─────────────────────────────────────────────────────────────
def draw_beam_system(spans, support_types, span_loads):
    if not spans: return
    total_L = sum(sp["L"] for sp in spans)
    fig, ax = plt.subplots(figsize=(max(10, total_L * 1.6), 3.0))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    BEAM_Y = 0.0; LOAD_TOP = 0.52

    # Beam line
    ax.plot([0, total_L], [BEAM_Y]*2, lw=5, color="#C0C6D4", solid_capstyle="round", zorder=3)

    # Node x-positions
    node_x = [0.0]
    cx = 0.0
    for sp in spans: cx += sp["L"]; node_x.append(cx)

    # Supports + node labels
    for i, (nx, stype) in enumerate(zip(node_x, support_types)):
        ax.text(nx, BEAM_Y - 0.3, chr(65+i), ha="center", va="top",
                color="#C0C6D4", fontsize=9, fontfamily="monospace", fontweight="600")
        if stype == "Fixed":
            ax.plot([nx, nx], [BEAM_Y-0.17, BEAM_Y+0.17], lw=5, color=TEAL)
            for dy in np.linspace(-0.17, 0.17, 5):
                ax.plot([nx, nx-0.1], [BEAM_Y+dy, BEAM_Y+dy-0.09], lw=0.9, color=TEAL, alpha=0.55)
        else:  # Roller or Cantilever — triangle
            tri = plt.Polygon([[nx, BEAM_Y], [nx-0.11, BEAM_Y-0.17], [nx+0.11, BEAM_Y-0.17]],
                               closed=True, color=TEAL, fill=(stype=="Roller"), zorder=4)
            ax.add_patch(tri)
            if stype == "Roller":
                ax.plot(nx, BEAM_Y-0.21, "o", ms=5, color=TEAL, zorder=4)

    # Loads
    curr_x = 0.0
    for i, sp in enumerate(spans):
        L = sp["L"]
        for ld in span_loads.get(i, []):
            t = ld.get("type"); mag = ld.get("mag", 0)

            if t == "Point":
                a = curr_x + ld.get("pos", 0)
                ax.annotate("", xy=(a, BEAM_Y+0.04), xytext=(a, LOAD_TOP),
                            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0, mutation_scale=12))
                ax.text(a, LOAD_TOP+0.06, f"{mag} kN", ha="center", va="bottom",
                        color=RED, fontsize=7.5, fontfamily="monospace")

            elif t == "UDL":
                xs = np.linspace(curr_x, curr_x+L, 20)
                ax.plot([curr_x, curr_x+L], [LOAD_TOP-0.02]*2, lw=1.8, color=TEAL)
                for x in xs:
                    ax.annotate("", xy=(x, BEAM_Y+0.04), xytext=(x, LOAD_TOP-0.02),
                                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.1, mutation_scale=8))
                ax.text(curr_x+L/2, LOAD_TOP+0.06, f"{mag} kN/m",
                        ha="center", va="bottom", color=TEAL, fontsize=7.5, fontfamily="monospace")

            elif t == "UDL-P":
                a = curr_x + ld.get("pos", 0); b = curr_x + ld.get("end", 0)
                if b > a:
                    xs = np.linspace(a, b, 14)
                    ax.plot([a, b], [LOAD_TOP-0.02]*2, lw=1.8, color=YEL)
                    for x in xs:
                        ax.annotate("", xy=(x, BEAM_Y+0.04), xytext=(x, LOAD_TOP-0.02),
                                    arrowprops=dict(arrowstyle="-|>", color=YEL, lw=1.1, mutation_scale=8))
                    ax.text((a+b)/2, LOAD_TOP+0.06, f"{mag} kN/m",
                            ha="center", va="bottom", color=YEL, fontsize=7.5, fontfamily="monospace")

            elif t == "UVL-P":
                a = curr_x + ld.get("pos", 0); b = curr_x + ld.get("end", 0)
                w_max = ld.get("mag", 0); shape = ld.get("shape", "start_zero")
                if b > a:
                    xs = np.linspace(a, b, 18)
                    hh = [(((x-a)/(b-a)) if shape == "start_zero" else ((b-x)/(b-a))) * (LOAD_TOP-0.06)
                          for x in xs]
                    ax.fill([a]+list(xs)+[b, a],
                            [BEAM_Y]+[BEAM_Y+h for h in hh]+[BEAM_Y, BEAM_Y],
                            alpha=0.12, color=GRN)
                    ax.plot([a]+list(xs)+[b],
                            [BEAM_Y]+[BEAM_Y+h for h in hh]+[BEAM_Y],
                            lw=1.2, color=GRN, alpha=0.75)
                    for x, h in zip(xs, hh):
                        if h > 0.04:
                            ax.annotate("", xy=(x, BEAM_Y+0.03), xytext=(x, BEAM_Y+h),
                                        arrowprops=dict(arrowstyle="-|>", color=GRN, lw=0.9, mutation_scale=6))
                    ax.text((a+b)/2, BEAM_Y+max(hh)+0.1, f"UVL {w_max} kN/m",
                            ha="center", va="bottom", color=GRN, fontsize=7.5, fontfamily="monospace")

        # Span dim
        ax.text(curr_x+L/2, BEAM_Y-0.14, f"L={L:.1f}m",
                ha="center", va="top", color=TXT, fontsize=7, fontfamily="monospace")
        curr_x += L

    ax.set_xlim(-0.4, total_L+0.4); ax.set_ylim(-0.5, LOAD_TOP+0.32); ax.axis("off")
    fig.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ─────────────────────────────────────────────────────────────
# SFD / BMD
# ─────────────────────────────────────────────────────────────
def plot_sfd_bmd(x, v, m, spans):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5))
    fig.patch.set_facecolor(BG)

    ax1.plot(x, v, lw=2, color=TEAL)
    ax1.fill_between(x, v, 0, where=(np.array(v)>=0), alpha=0.15, color=GRN)
    ax1.fill_between(x, v, 0, where=(np.array(v)<0),  alpha=0.15, color=RED)
    ax1.axhline(0, color=GRID, lw=1)
    _annotate_extremes(ax1, x, v)
    _style_ax(ax1, title="Shear Force Diagram", ylabel="Shear  (kN)")

    ax2.plot(x, m, lw=2, color=YEL)
    ax2.fill_between(x, m, 0, where=(np.array(m)>=0), alpha=0.15, color=YEL)
    ax2.fill_between(x, m, 0, where=(np.array(m)<0),  alpha=0.12, color=RED)
    ax2.axhline(0, color=GRID, lw=1)
    ax2.invert_yaxis()
    _annotate_extremes(ax2, x, m)
    _style_ax(ax2, title="Bending Moment Diagram  (sagging ↓)",
              xlabel="Distance  (m)", ylabel="Moment  (kNm)")

    # Span dividers
    cx = 0.0
    for sp in spans:
        cx += sp["L"]
        for ax_ in (ax1, ax2):
            ax_.axvline(cx, color=GRID, lw=1, ls=":", alpha=0.7)

    fig.tight_layout(pad=1.4)
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ─────────────────────────────────────────────────────────────
# FRAME DIAGRAM
# ─────────────────────────────────────────────────────────────
def draw_frame(nodes, members, results=None):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax, title="Portal Frame", xlabel="x (m)", ylabel="y (m)")

    mem_labels = ["AB","BC","CD"]
    mem_colors = [TEAL, YEL, GRN]
    coord_map  = {0:(0,1), 1:(1,2), 2:(2,3)}

    for idx, mem in enumerate(members):
        ni, nj = coord_map[mem["id"]]
        xi,yi = nodes[ni]["x"], nodes[ni]["y"]
        xj,yj = nodes[nj]["x"], nodes[nj]["y"]
        ax.plot([xi,xj],[yi,yj], lw=3.5, color=mem_colors[idx],
                solid_capstyle="round", label=mem_labels[idx])

        if results and idx in results:
            ms, me = results[idx]
            L = np.hypot(xj-xi, yj-yi)
            if L > 0:
                dx, dy = (xj-xi)/L, (yj-yi)/L
                px, py = -dy, dx
                scale = 0.32 / max(abs(ms), abs(me), 1)
                ts = np.linspace(0, 1, 30)
                bx = [xi+t*(xj-xi)+(ms*(1-t)+me*t)*scale*px for t in ts]
                by = [yi+t*(yj-yi)+(ms*(1-t)+me*t)*scale*py for t in ts]
                ax.fill([xi]+bx+[xj,xi], [yi]+by+[yj,yi],
                        alpha=0.14, color=mem_colors[idx])
                ax.plot(bx, by, lw=1.3, color=mem_colors[idx], alpha=0.8)
                for (xx,yy,val) in [(xi,yi,ms),(xj,yj,me)]:
                    ax.annotate(f"{val:+.2f}", xy=(xx,yy), xytext=(5,5),
                                textcoords="offset points",
                                fontsize=7, color=mem_colors[idx], fontfamily="monospace")

    for n in nodes:
        ax.plot(n["x"], n["y"], "o", color="#E8EAF0", ms=8, zorder=5)
        ax.text(n["x"]-0.18, n["y"], ["A","B","C","D"][n["id"]],
                color="#E8EAF0", fontsize=9, fontfamily="monospace",
                va="center", ha="right", fontweight="600")

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, facecolor=SURF, edgecolor=GRID, labelcolor=TXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:0.5rem 0 1.5rem;'>
            <div style='font-size:2.2rem'>🏗️</div>
            <div style='font-family:Syne,sans-serif; font-weight:800; color:#E8EAF0; font-size:1.15rem; letter-spacing:-0.5px;'>StructSolve</div>
            <div style='color:#8B92A8; font-size:0.68rem; margin-top:2px;'>BS 8110-1:1997</div>
        </div>
        """, unsafe_allow_html=True)

        module = st.radio(
            "SELECT MODULE",
            ["🔩 Beam Analysis", "🏛️ Frame Analysis", "🧱 RC Design"],
            label_visibility="visible",
        )

        st.markdown("---")
        st.markdown("""
        <div style='color:#8B92A8; font-size:0.7rem; line-height:1.9;'>
        <b style='color:#4ECDC4;'>MODULES</b><br>
        ↳ Slope Deflection Method<br>
        ↳ Direct Stiffness Method<br>
        ↳ BS 8110 RC Design<br>
        &nbsp;&nbsp;· Flexure (Cl. 3.4.4)<br>
        &nbsp;&nbsp;· Shear (Cl. 3.4.5)<br>
        &nbsp;&nbsp;· Deflection check<br><br>
        <b style='color:#4ECDC4;'>UNITS</b><br>
        ↳ Lengths: m / mm<br>
        ↳ Forces: kN<br>
        ↳ Moments: kNm<br>
        ↳ Stress: N/mm²<br>
        </div>
        """, unsafe_allow_html=True)

    return module


# ─────────────────────────────────────────────────────────────
# BEAM PAGE
# ─────────────────────────────────────────────────────────────
def beam_page():
    st.markdown("""
    <div class='main-header'>
        <h1>🔩 Beam Analysis <span class='badge'>BS 8110</span></h1>
        <p>Continuous beam analysis using the Slope Deflection Method · SFD & BMD · RC design</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────
    defaults = {
        "analysis_done": False, "run_design": False,
        "internal_actions": {}, "spans_cache": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # ── Configuration ─────────────────────────────────────────
    st.markdown("<div class='section-label'>Beam Configuration</div>", unsafe_allow_html=True)
    col_cfg, col_sup = st.columns([1, 2])

    with col_cfg:
        n_joints = st.number_input("Number of joints", 2, 10, 3, key="n_joints")
        n_spans  = n_joints - 1

    with col_sup:
        st.markdown("<small style='color:#8B92A8'>Support conditions at each joint:</small>", unsafe_allow_html=True)
        support_types = []
        sup_cols = st.columns(n_joints)
        for i in range(n_joints):
            support_types.append(sup_cols[i].selectbox(
                f"**{chr(65+i)}**", ["Roller","Fixed","Cantilever"], key=f"s_{i}"))

    # ── Settlement & end rotation ─────────────────────────────
    st.markdown("<div class='section-label'>Support Settlement & End Rotation (optional)</div>", unsafe_allow_html=True)
    st.markdown("<small style='color:#8B92A8'>Enter settlement (mm, +ve downward) or prescribed end rotation (rad, +ve anticlockwise) at any joint. Leave 0 if not applicable.</small>", unsafe_allow_html=True)

    with st.expander("⚙️  Settlement / Rotation inputs", expanded=False):
        se1, se2 = st.columns(2)
        settlements = {}   # joint index → settlement in mm
        rotations   = {}   # joint index → rotation in rad
        for i in range(n_joints):
            settlements[i] = se1.number_input(f"Settlement at {chr(65+i)} (mm, +ve down)",
                                              value=0.0, step=1.0, key=f"sett_{i}")
            rotations[i]   = se2.number_input(f"Prescribed θ at {chr(65+i)} (rad)",
                                              value=0.0, format="%.5f", key=f"rot_{i}")

    # ── Span geometry ─────────────────────────────────────────
    st.markdown("<div class='section-label'>Span Geometry & Stiffness</div>", unsafe_allow_html=True)

    ei_mode = st.radio("EI input mode",
                       ["Relative EI (dimensionless)",
                        "Actual E & I (E in GPa, I in mm⁴ → EI in kNm²)"],
                       horizontal=True, key="ei_mode")

    spans = []
    span_cols = st.columns(n_spans)
    for i in range(n_spans):
        with span_cols[i]:
            st.markdown(f"<small style='color:#4ECDC4;font-weight:600'>Span {chr(65+i)}–{chr(66+i)}</small>",
                        unsafe_allow_html=True)
            L = st.number_input("Length (m)", 0.1, 50.0, 5.0, key=f"L_{i}")

            if ei_mode.startswith("Relative"):
                EI = st.number_input("Relative EI", 0.001, 1e6, 1.0, format="%.3f", key=f"EI_{i}")
            else:
                E_gpa = st.number_input("E (GPa)", 1.0, 500.0, 200.0, key=f"E_{i}")
                I_mm4 = st.number_input("I (mm⁴ ×10⁶)", 0.001, 1e6, 40.0,
                                        format="%.3f", key=f"I_{i}",
                                        help="Enter I in millions of mm⁴, e.g. 4e7 mm⁴ → enter 40")
                # EI in kNm²: E[GPa]*1e6[N/mm²/GPa] * I[mm⁴]*1e6 [from ×10⁶ input]
                #  N.mm² → kNm²: ÷1e9
                EI = (E_gpa * 1e6) * (I_mm4 * 1e6) / 1e9
                st.markdown(f"<small style='color:#8B92A8'>EI = {EI:.1f} kNm²</small>",
                            unsafe_allow_html=True)

            spans.append({"L": L, "EI": EI})

    # ── Load manager ──────────────────────────────────────────
    st.markdown("#### Applied Loads")

    # Initialise load list with same default schema as previous project
    if "beam_loads" not in st.session_state:
        st.session_state.beam_loads = [
            {"type": "UDL",   "span": 0, "mag": 10.0, "mag_end": 10.0, "pos_s": 0.0, "pos_e": 5.0},
            {"type": "POINT", "span": 0, "mag": 25.0,  "mag_end": 0.0,  "pos_s": 2.5, "pos_e": 2.5},
        ]

    # ── Add Load expander — exact structure from previous project ──
    with st.expander("➕ Add Load", expanded=False):
        ac1, ac2, ac3 = st.columns(3)
        new_type = ac1.selectbox("Load Type", ["UDL", "UVL", "POINT"], key="nl_type")
        new_span = ac2.number_input("Span", min_value=1, max_value=n_spans,
                                    value=1, step=1, key="nl_span") - 1

        L_sel = spans[new_span]["L"] if new_span < len(spans) else 5.0

        if new_type in ["UDL", "UVL"]:
            bc1, bc2 = st.columns(2)
            new_pos_s = bc1.number_input("Start position (m)", min_value=0.0,
                                         max_value=float(L_sel), value=0.0, step=0.25, key="nl_ps")
            new_pos_e = bc2.number_input("End position (m)", min_value=0.0,
                                         max_value=float(L_sel), value=float(L_sel), step=0.25, key="nl_pe")
            mc1, mc2 = st.columns(2)
            new_mag   = mc1.number_input("Intensity at start (kN/m)", value=10.0, step=1.0, key="nl_mag")
            new_mag_e = mc2.number_input(
                "Intensity at end (kN/m)" if new_type == "UVL" else "Intensity at end (kN/m) [same as start for UDL]",
                value=10.0 if new_type == "UDL" else 0.0, step=1.0, key="nl_mage",
                disabled=(new_type == "UDL"),
            )
            if new_type == "UDL":
                new_mag_e = new_mag
        else:  # POINT
            new_pos_s = st.number_input("Position (m from left of span)", min_value=0.0,
                                        max_value=float(L_sel), value=min(2.0, L_sel), step=0.1, key="nl_ps")
            new_pos_e = new_pos_s
            new_mag   = st.number_input("Force (kN)", value=25.0, step=1.0, key="nl_mag")
            new_mag_e = 0.0

        if st.button("Add Load"):
            st.session_state.beam_loads.append({
                "type": new_type, "span": new_span,
                "mag": new_mag, "mag_end": new_mag_e,
                "pos_s": new_pos_s, "pos_e": new_pos_e,
            })
            st.rerun()

    # ── Current loads table — exact column layout from previous project ──
    if st.session_state.beam_loads:
        hc = st.columns([1, 0.8, 1.1, 1.1, 1.1, 1.1, 0.4])
        for lbl, h in zip(["Type", "Span", "Start (m)", "End (m)", "Mag (start)", "Mag (end)", ""], hc):
            h.markdown(f"<small style='color:#8B92A8'>{lbl}</small>", unsafe_allow_html=True)

        for idx, ld in enumerate(st.session_state.beam_loads):
            c = st.columns([1, 0.8, 1.1, 1.1, 1.1, 1.1, 0.4])
            type_colors = {"UDL": TEAL, "UVL": YEL, "POINT": RED}
            col = type_colors.get(ld["type"], "#E8EAF0")
            c[0].markdown(f"<span style='color:{col}; font-weight:600'>`{ld['type']}`</span>",
                          unsafe_allow_html=True)
            c[1].markdown(f"Span {ld['span']+1}")
            c[2].markdown(f"{ld['pos_s']:.2f}")
            c[3].markdown(f"{ld['pos_e']:.2f}" if ld["type"] != "POINT" else "—")
            c[4].markdown(f"{ld['mag']:.1f} kN/m" if ld["type"] != "POINT" else f"{ld['mag']:.1f} kN")
            c[5].markdown(f"{ld['mag_end']:.1f}" if ld["type"] == "UVL"
                          else ("—" if ld["type"] == "POINT" else f"{ld['mag']:.1f}"))
            if c[6].button("✕", key=f"del_bl_{idx}"):
                st.session_state.beam_loads.pop(idx)
                st.rerun()
    else:
        st.markdown("<div class='result-card muted' style='color:#8B92A8'>No loads added yet — use the panel above.</div>",
                    unsafe_allow_html=True)

    # ── Translate UI schema → solver schema ───────────────────
    # UI:     type ∈ {UDL, UVL, POINT}, pos_s, pos_e, mag, mag_end
    # Solver: type ∈ {UDL, UDL-P, UVL-P, Point}, pos, end, mag, shape
    span_loads = {i: [] for i in range(n_spans)}
    load_ok = True
    for ld in st.session_state.beam_loads:
        sp = ld["span"]
        if sp >= n_spans:
            continue
        L_sp = spans[sp]["L"]

        if ld["type"] == "UDL":
            # Full-span if pos_s==0 and pos_e==L, else partial
            if ld["pos_s"] == 0.0 and abs(ld["pos_e"] - L_sp) < 1e-6:
                span_loads[sp].append({"type": "UDL", "mag": ld["mag"], "pos": 0.0, "end": None, "shape": None})
            else:
                if ld["pos_e"] <= ld["pos_s"]:
                    st.error(f"UDL on span {sp+1}: end position must be greater than start."); load_ok = False; continue
                span_loads[sp].append({"type": "UDL-P", "mag": ld["mag"],
                                       "pos": ld["pos_s"], "end": ld["pos_e"], "shape": None})

        elif ld["type"] == "UVL":
            if ld["pos_e"] <= ld["pos_s"]:
                st.error(f"UVL on span {sp+1}: end position must be greater than start."); load_ok = False; continue
            # mag = intensity at start, mag_end = intensity at end
            # Map to solver's UVL-P: mag=peak, shape=start_zero or end_zero
            # If start=0 and end>0: shape=end_zero (peak at start), mag=mag
            # If start>0 and end=0: shape=start_zero (peak at end), mag=mag_end
            # General: use whichever end is larger as the peak
            mag_s, mag_e = ld["mag"], ld["mag_end"]
            if mag_s >= mag_e:
                # Peak at start → triangular decreasing → end_zero
                span_loads[sp].append({"type": "UVL-P", "mag": mag_s,
                                       "pos": ld["pos_s"], "end": ld["pos_e"], "shape": "end_zero"})
            else:
                # Peak at end → triangular increasing → start_zero
                span_loads[sp].append({"type": "UVL-P", "mag": mag_e,
                                       "pos": ld["pos_s"], "end": ld["pos_e"], "shape": "start_zero"})

        elif ld["type"] == "POINT":
            span_loads[sp].append({"type": "Point", "mag": ld["mag"],
                                   "pos": ld["pos_s"], "end": None, "shape": None})

    # ── Tabs: Preview | Results ───────────────────────────────
    tab_vis, tab_results = st.tabs(["👁️  Beam Preview", "📊  Analysis Results"])

    with tab_vis:
        draw_beam_system(spans, support_types, span_loads)

    with tab_results:
        cA, cB, _ = st.columns([1,1,2])
        run_analysis        = cA.button("🔍  Analyse Only",       use_container_width=True)
        run_analysis_design = cB.button("🧮  Analyse + Design",   use_container_width=True)
        if run_analysis_design: run_analysis = True

        if run_analysis and load_ok:
            try:
                # Convert settlements mm→m, build per-span sway correction
                sett_m = {k: v / 1000.0 for k, v in settlements.items()}  # mm → m

                # Settlement correction Δ per span (near end sinks relative to far)
                sway_corr = {}  # span index → Δ in metres (positive = near sinks)
                for i in range(n_spans):
                    sway_corr[i] = sett_m.get(i, 0.0) - sett_m.get(i+1, 0.0)

                # Prescribed rotations at joints (for fixed/cantilever with imposed rotation)
                prescribed_theta = {k: v for k, v in rotations.items() if v != 0.0}

                thetas, fems = BeamSolver.solve_continuous_beam(
                    n_joints, spans, support_types, span_loads,
                    sway_corrections=sway_corr,
                    prescribed_rotations=prescribed_theta)

                all_x, all_v, all_m = [], [], []
                curr_x = 0.0
                support_moments, support_shears, span_results, internal_actions = {}, {}, [], {}

                for i in range(n_spans):
                    L  = spans[i]["L"]; EI = spans[i]["EI"]
                    delta = sway_corr.get(i, 0.0)  # m
                    m_ab = fems[i][0] + (2*EI/L)*(2*thetas[i]   + thetas[i+1] - 3*delta/L)
                    m_ba = fems[i][1] + (2*EI/L)*(2*thetas[i+1] + thetas[i]   - 3*delta/L)
                    if i == 0: support_moments["A"] = m_ab
                    support_moments[chr(66+i)] = m_ba

                    x, v, m = BeamSolver.get_diagram_data(
                        {"id": i, "L": L}, m_ab, m_ba,
                        [{"member": i, **ld} for ld in span_loads[i]])
                    m = -np.array(m); v = np.array(v)
                    all_x.extend(x + curr_x); all_v.extend(v); all_m.extend(m)
                    curr_x += L

                    span_results.append({
                        "span":  f"{chr(65+i)}-{chr(66+i)}",
                        "M_sag": float(max(m)),
                        "V_max": float(max(abs(v))),
                    })
                    internal_actions[f"{chr(65+i)}-{chr(66+i)}"] = {"M_start": m_ab, "M_end": m_ba}

                support_shears["left"]  = abs(all_v[0])
                support_shears["right"] = abs(all_v[-1])

                st.session_state.update({
                    "analysis_done": True,
                    "all_x": np.array(all_x), "all_v": np.array(all_v), "all_m": np.array(all_m),
                    "spans_cache": spans, "span_results": span_results,
                    "support_moments": support_moments, "support_shears": support_shears,
                    "run_design": run_analysis_design, "internal_actions": internal_actions,
                })

            except Exception as e:
                st.error(f"Analysis error: {e}"); return

        # ── Results display ───────────────────────────────────
        if not st.session_state.analysis_done:
            st.markdown("""
            <div style='display:flex;align-items:center;justify-content:center;height:260px;
                        border:2px dashed #2A2E40;border-radius:12px;color:#8B92A8;
                        font-size:0.82rem;text-align:center;padding:2rem;'>
                <div><div style='font-size:2.5rem;margin-bottom:0.8rem'>📊</div>
                Click <b>Analyse Only</b> or <b>Analyse + Design</b> to compute results.</div>
            </div>""", unsafe_allow_html=True)
            return

        x = st.session_state.all_x
        v = st.session_state.all_v
        m = st.session_state.all_m
        spans_c = st.session_state.spans_cache

        # Key metrics
        st.markdown(f"""
        <div class='metric-grid'>
            {''.join(
                f"<div class='metric-box'>"
                f"<div class='val'>{sr['M_sag']:.2f}</div>"
                f"<div class='lbl'>M sag {sr['span']} (kNm)</div>"
                f"</div>"
                for sr in st.session_state.span_results
            )}
            <div class='metric-box'>
                <div class='val red'>{max(abs(val) for val in st.session_state.support_moments.values()):.2f}</div>
                <div class='lbl'>Max Hogging (kNm)</div>
            </div>
            <div class='metric-box'>
                <div class='val yel'>{max(st.session_state.support_shears.values()):.2f}</div>
                <div class='lbl'>Max Shear (kN)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SFD / BMD
        plot_sfd_bmd(x, v, m, spans_c)

        # Support actions table
        st.markdown("<div class='section-label'>Support Actions</div>", unsafe_allow_html=True)
        support_positions = [0.0]
        cum = 0.0
        for sp in spans_c: cum += sp["L"]; support_positions.append(cum)

        external_shears = {"A": v[0], chr(65+len(spans_c)): v[-1]}
        internal = st.session_state.internal_actions
        support_rows = []
        for i in range(len(support_positions)):
            name = chr(65+i)
            row = {"Support": name,
                   "V ext (kN)": round(external_shears.get(name, 0.0), 3),
                   "M left (kNm)": "—", "M right (kNm)": "—"}
            if i > 0:
                k = f"{chr(64+i)}-{name}"
                if k in internal: row["M left (kNm)"] = round(internal[k]["M_end"], 3)
            if i < len(spans_c):
                k = f"{name}-{chr(66+i)}"
                if k in internal: row["M right (kNm)"] = round(internal[k]["M_start"], 3)
            support_rows.append(row)

        support_df = pd.DataFrame(support_rows)
        st.dataframe(support_df, use_container_width=True, hide_index=True)

        # PDF
        pdf_buf = export_analysis_pdf(support_df=support_df, x=x, v=v, m=m)
        st.download_button("📄  Export Analysis PDF", data=pdf_buf,
                           file_name="beam_analysis.pdf", mime="application/pdf")

        # ── Design section ────────────────────────────────────
        if not st.session_state.run_design:
            return

        st.markdown("<div class='section-label'>RC Design Parameters (BS 8110)</div>", unsafe_allow_html=True)
        st.markdown("<div class='calc-card'>Section geometry, materials, and cover for automatic design from analysis results.</div>", unsafe_allow_html=True)

        dc1, dc2, dc3 = st.columns(3)
        bw    = dc1.number_input("Web width b (mm)",      min_value=0.0, value=300.0)
        D     = dc1.number_input("Overall depth D (mm)",  min_value=0.0, value=550.0)
        cover = dc2.number_input("Cover (mm)",            min_value=0.0, value=40.0)
        fcu   = dc2.number_input("fcu (N/mm²)",           min_value=0.0, value=30.0)
        fy    = dc3.number_input("fy (N/mm²)",            min_value=0.0, value=460.0)

        if bw <= 0 or D <= cover or fcu <= 0 or fy <= 0:
            st.warning("Enter valid section parameters to run design."); return

        d = D - cover; dp = cover
        Mu_sag = max(r["M_sag"] for r in st.session_state.span_results)
        Mu_hog = max(abs(val) for val in st.session_state.support_moments.values())

        design = auto_design_from_beam_analysis(
            moments={"sagging": Mu_sag, "hogging": Mu_hog},
            shears=st.session_state.support_shears,
            b=bw, d=d, dp=dp, fcu=fcu, fy=fy)

        flex  = design["flexure"]["sagging"]
        shear = design["shear"]
        defl  = deflection_check(
            L=sum(sp["L"] for sp in spans_c), d=d,
            beam_type="continuous", fy=fy,
            As_req=flex["As_req"], As_prov=flex["As_prov"])

        defl_cls = "grn" if defl["status"] == "PASS" else "red"

        st.markdown(f"""
        <div class='metric-grid'>
            <div class='metric-box'><div class='val'>{Mu_sag:.2f}</div><div class='lbl'>Mu sagging (kNm)</div></div>
            <div class='metric-box'><div class='val red'>{Mu_hog:.2f}</div><div class='lbl'>Mu hogging (kNm)</div></div>
            <div class='metric-box'><div class='val'>{flex['As_req']:.0f}</div><div class='lbl'>As req (mm²)</div></div>
            <div class='metric-box'><div class='val grn'>{flex['As_prov']:.0f}</div><div class='lbl'>As prov (mm²)</div></div>
            <div class='metric-box'><div class='val yel'>{shear['v']:.3f}</div><div class='lbl'>v (N/mm²)</div></div>
            <div class='metric-box'><div class='val yel'>{shear['vc']:.3f}</div><div class='lbl'>vc (N/mm²)</div></div>
            <div class='metric-box'><div class='val {defl_cls}'>{defl['actual']:.1f} / {defl['allowable']:.1f}</div><div class='lbl'>L/d actual / limit</div></div>
            <div class='metric-box'><div class='val {defl_cls}'>{defl['status']}</div><div class='lbl'>Deflection</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='result-card'>Tension bars (sagging): <b style='color:{TEAL}'>{flex['tension_bars']}</b></div>
        <div class='result-card warning'>Shear links: <b style='color:{YEL}'>{shear['links']}</b> &nbsp;|&nbsp; Status: <b>{shear['status']}</b></div>
        """, unsafe_allow_html=True)

        summary = format_design_summary(
            wu=None, moments={"Sagging": Mu_sag, "Hogging": Mu_hog},
            flexure={"Mid-span": flex}, shear=shear, deflection=defl)

        with st.expander("📋  Full Design Workings"):
            st.markdown(f"<div class='calc-steps'>{summary}</div>", unsafe_allow_html=True)

        pdf_buf2 = export_analysis_design_pdf(
            support_df=support_df, x=x, v=v, m=m, design_summary=summary)
        st.download_button("📄  Export Analysis + Design PDF", data=pdf_buf2,
                           file_name="beam_analysis_design.pdf", mime="application/pdf")


# ─────────────────────────────────────────────────────────────
# FRAME PAGE
# ─────────────────────────────────────────────────────────────
def frame_page():
    st.markdown("""
    <div class='main-header'>
        <h1>🏛️ Frame Analysis <span class='badge'>STIFFNESS</span></h1>
        <p>Portal frame analysis using the Slope Deflection Method · Sway & Non-Sway cases</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Frame type ────────────────────────────────────────────
    st.markdown("<div class='section-label'>Frame Type & Case</div>", unsafe_allow_html=True)
    fc1, fc2 = st.columns([1, 1])
    frame_type = fc1.radio("Type", ["Sway Frame","Non-Sway Frame"], horizontal=True)
    case_type  = fc2.selectbox("Sway Case", [1,2,3,4,5]) if frame_type == "Sway Frame" else 0

    # ── Geometry ──────────────────────────────────────────────
    st.markdown("<div class='section-label'>Geometry & Stiffness</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    h1 = g1.number_input("Column height h₁ — AB (m)", 1.0, 20.0, 4.0)
    h2 = g1.number_input("Column height h₂ — CD (m)", 1.0, 20.0, 4.0)
    L  = g2.number_input("Beam span L — BC (m)",       1.0, 30.0, 6.0)

    st.markdown("<small style='color:#8B92A8'>Enter EI for each member (kNm² or relative). Use E×I if you have material properties.</small>", unsafe_allow_html=True)
    ei1, ei2, ei3 = st.columns(3)
    ei_AB = ei1.number_input("EI — Column AB", 0.001, 1e9, 1000.0, format="%.3f", key="ei_AB")
    ei_BC = ei2.number_input("EI — Beam BC",   0.001, 1e9, 2000.0, format="%.3f", key="ei_BC")
    ei_CD = ei3.number_input("EI — Column CD", 0.001, 1e9, 1000.0, format="%.3f", key="ei_CD")

    nodes   = [{"id":0,"x":0,"y":0},{"id":1,"x":0,"y":h1},{"id":2,"x":L,"y":h2},{"id":3,"x":L,"y":0}]
    members = [{"id":0,"L":h1,"I":ei_AB},{"id":1,"L":L,"I":ei_BC},{"id":2,"L":h2,"I":ei_CD}]
    loads   = []

    # ── Loading ───────────────────────────────────────────────
    st.markdown("<div class='section-label'>Loading</div>", unsafe_allow_html=True)

    if frame_type == "Sway Frame":
        lc1, lc2 = st.columns(2)
        if case_type == 1:
            p_beam = lc1.number_input("Point load on beam (kN)",   0.0, 500.0, 50.0)
            p_col  = lc1.number_input("Point load on left col (kN)", 0.0, 500.0, 20.0)
            h_p    = lc2.number_input("Height of P from A (m)",    0.0, h1, h1/2)
            w_col  = lc2.number_input("UDL on right col (kN/m)",   0.0, 200.0, 10.0)
            loads  = [{"member":1,"type":"Point","mag":p_beam,"pos":L/2},
                      {"member":0,"type":"Point","mag":p_col, "pos":h_p},
                      {"member":2,"type":"UDL",  "mag":w_col, "pos":0}]
        elif case_type == 2:
            w_col = st.number_input("UDL on left col (kN/m)", 0.0, 200.0, 10.0)
            loads = [{"member":0,"type":"UDL","mag":w_col,"pos":0}]
        elif case_type == 3:
            p_top = st.number_input("Point load at B (kN)", 0.0, 500.0, 30.0)
            loads = [{"member":0,"type":"Point","mag":p_top,"pos":h1}]
        elif case_type == 4:
            w_beam = st.number_input("UDL on beam (kN/m)", 0.0, 200.0, 20.0)
            loads  = [{"member":1,"type":"UDL","mag":w_beam,"pos":0}]
        elif case_type == 5:
            w_beam = lc1.number_input("UDL on beam (kN/m)",         0.0, 200.0, 20.0)
            p_col  = lc1.number_input("Point load on right col (kN)", 0.0, 500.0, 25.0)
            h_p    = lc2.number_input("Height of P from D (m)",     0.0, h2, h2/2)
            loads  = [{"member":1,"type":"UDL",  "mag":w_beam,"pos":0},
                      {"member":2,"type":"Point","mag":p_col, "pos":h_p}]
    else:
        ns_cols = st.columns(3)
        for i, (name, col) in enumerate(zip(["Left Column","Beam","Right Column"], ns_cols)):
            with col:
                st.markdown(f"<small style='color:#4ECDC4;font-weight:600'>{name}</small>", unsafe_allow_html=True)
                lt = st.selectbox("Load", ["None","Full UDL","Partial UDL","Point"], key=f"ns_lt_{i}")
                if lt == "Full UDL":
                    w = st.number_input("kN/m", 0.0, 500.0, 0.0, key=f"ns_wf_{i}")
                    if w: loads.append({"member":i,"type":"UDL","mag":w,"pos":0.0})
                elif lt == "Partial UDL":
                    w = st.number_input("kN/m", 0.0, 500.0, 0.0, key=f"ns_wp_{i}")
                    a = st.number_input("Start (m)", 0.0, members[i]["L"], 0.0, key=f"ns_a_{i}")
                    b = st.number_input("End (m)", a, members[i]["L"], members[i]["L"], key=f"ns_b_{i}")
                    if w and b > a: loads.append({"member":i,"type":"UDL","mag":w,"pos":a,"end":b})
                elif lt == "Point":
                    P = st.number_input("kN", 0.0, 500.0, 0.0, key=f"ns_p_{i}")
                    a = st.number_input("Position (m)", 0.0, members[i]["L"], members[i]["L"]/2, key=f"ns_ap_{i}")
                    if P: loads.append({"member":i,"type":"Point","mag":P,"pos":a})

    # ── Tabs: Preview | Results ───────────────────────────────
    tab_preview, tab_results = st.tabs(["👁️  Frame Preview", "📊  Analysis Results"])

    with tab_preview:
        draw_frame(nodes, members)

    with tab_results:
        if st.button("🚀  Run Frame Analysis", use_container_width=False):
            try:
                results, unknowns = (
                    FrameSolver.solve_frame_sway(nodes, members, loads, case_type)
                    if frame_type == "Sway Frame"
                    else FrameSolver.solve_frame_non_sway(nodes, members, loads)
                )
            except Exception as e:
                st.error(f"Frame analysis error: {e}"); return

            labels = ["AB","BC","CD"]
            st.markdown("<div class='section-label'>Member End Moments</div>", unsafe_allow_html=True)
            for i, (m_start, m_end) in results.items():
                st.markdown(f"""
                <div class='result-card'>
                    <b style='color:{TEAL}'>Member {labels[i]}</b> &nbsp;|&nbsp;
                    M_near = <b style='color:{TEAL}'>{m_start:+.3f} kNm</b> &nbsp;|&nbsp;
                    M_far = <b style='color:{YEL}'>{m_end:+.3f} kNm</b>
                </div>""", unsafe_allow_html=True)

            df_res = pd.DataFrame([
                {"Member": labels[i], "M_start (kNm)": round(m1, 3), "M_end (kNm)": round(m2, 3)}
                for i, (m1, m2) in results.items()
            ])
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            # Unknowns
            if len(unknowns) >= 3:
                tB, tC, delta = unknowns[0], unknowns[1], unknowns[2]
                st.markdown(f"""
                <div class='metric-grid'>
                    <div class='metric-box'><div class='val'>{tB:.4f}</div><div class='lbl'>θ_B</div></div>
                    <div class='metric-box'><div class='val'>{tC:.4f}</div><div class='lbl'>θ_C</div></div>
                    <div class='metric-box'><div class='val yel'>{delta:.4f}</div><div class='lbl'>Sway Δ</div></div>
                </div>""", unsafe_allow_html=True)

            # BMD overlay on frame
            st.markdown("<div class='section-label'>Frame Diagram + BMD Overlay</div>", unsafe_allow_html=True)
            draw_frame(nodes, members, results)

        else:
            st.markdown("""
            <div style='display:flex;align-items:center;justify-content:center;height:220px;
                        border:2px dashed #2A2E40;border-radius:12px;color:#8B92A8;
                        font-size:0.82rem;text-align:center;padding:2rem;'>
                <div><div style='font-size:2.5rem;margin-bottom:0.8rem'>🏛️</div>
                Click <b>Run Frame Analysis</b> to compute results.</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DESIGN PAGE
# ─────────────────────────────────────────────────────────────
def design_page():
    st.markdown("""
    <div class='main-header'>
        <h1>🧱 RC Beam Design <span class='badge'>BS 8110</span></h1>
        <p>Flexure (Cl. 3.4.4) · Shear (Cl. 3.4.5) · Deflection check · Rectangular & T-sections</p>
    </div>
    """, unsafe_allow_html=True)

    # ── System ────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Beam System</div>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    beam_system  = dc1.selectbox("System", ["Simply Supported Beam","Continuous Beam"])
    section_type = dc2.selectbox("Section", ["Rectangular Beam","T-Beam"])

    # ── Geometry ──────────────────────────────────────────────
    st.markdown("<div class='section-label'>Geometry</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    L     = g1.number_input("Span L (m)",          min_value=0.0, value=6.0)
    bw    = g1.number_input("Web width b (mm)",     min_value=0.0, value=300.0)
    D     = g1.number_input("Overall depth D (mm)", min_value=0.0, value=550.0)
    cover = g1.number_input("Cover (mm)",           min_value=0.0, value=40.0)

    bf = hf = None
    if section_type == "T-Beam":
        bf = g2.number_input("Flange width bf (mm)",     min_value=0.0, value=1200.0)
        hf = g2.number_input("Flange thickness hf (mm)", min_value=0.0, value=150.0)

    d = D - cover; dp = cover

    # ── Materials ─────────────────────────────────────────────
    st.markdown("<div class='section-label'>Materials</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    fcu = m1.number_input("fcu (N/mm²)", min_value=0.0, value=30.0)
    fy  = m2.number_input("fy (N/mm²)",  min_value=0.0, value=460.0)

    # ── Loading ───────────────────────────────────────────────
    st.markdown("<div class='section-label'>Characteristic Loading</div>", unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    gk = l1.number_input("Dead load gk (kN/m)", min_value=0.0, value=10.0)
    qk = l2.number_input("Live load qk (kN/m)", min_value=0.0, value=5.0)
    wu = 1.4 * gk + 1.6 * qk

    st.markdown(f"""
    <div class='result-card'>
        Ultimate load: <b>wu = 1.4({gk}) + 1.6({qk}) =
        <span style='color:{TEAL}'>{wu:.2f} kN/m</span></b>
    </div>""", unsafe_allow_html=True)

    # ── Design button ─────────────────────────────────────────
    st.markdown("<div class='section-label'>Run Design</div>", unsafe_allow_html=True)
    if not st.button("🧮  Design Beam (BS 8110)", use_container_width=False):
        st.markdown("""
        <div style='display:flex;align-items:center;justify-content:center;height:200px;
                    border:2px dashed #2A2E40;border-radius:12px;color:#8B92A8;
                    font-size:0.82rem;text-align:center;padding:2rem;'>
            <div><div style='font-size:2.5rem;margin-bottom:0.8rem'>🧱</div>
            Click <b>Design Beam</b> to run the BS 8110 design routine.</div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Simply supported ──────────────────────────────────────
    if beam_system == "Simply Supported Beam":
        Mu = wu * L**2 / 8; Vu = wu * L / 2

        flex = (beam_flexural_design(Mu=Mu, b=bw, d=d, dp=dp, fcu=fcu, fy=fy)
                if section_type == "Rectangular Beam"
                else beam_flexural_design_T(Mu=Mu, bf=bf, hf=hf, bw=bw, d=d, dp=dp, fcu=fcu, fy=fy))
        shear = beam_shear_design(Vu=Vu, b=bw, d=d, fcu=fcu, As=flex["As_req"])
        defl  = deflection_check(L=L, d=d, beam_type="simply_supported",
                                 fy=fy, As_req=flex["As_req"], As_prov=flex["As_prov"])
        _render_design_results({"Mu": Mu}, {"Mid-span": flex}, shear, defl, wu)

    # ── Continuous ────────────────────────────────────────────
    else:
        st.markdown("<div class='calc-card'>Enter governing ultimate moments and shears from your beam analysis.</div>", unsafe_allow_html=True)
        ai1, ai2 = st.columns(2)
        Mu_AB = ai1.number_input("Sagging M — span AB (kNm)", min_value=0.0)
        Mu_BC = ai1.number_input("Sagging M — span BC (kNm)", min_value=0.0)
        Mu_B  = ai2.number_input("Hogging M at B (kNm)",      min_value=0.0)
        Vu_B1 = ai2.number_input("Shear at B left (kN)",      min_value=0.0)
        Vu_B2 = ai2.number_input("Shear at B right (kN)",     min_value=0.0)

        results = design_continuous_beam(
            moments={"sagging": {"AB": Mu_AB,"BC": Mu_BC}, "hogging": {"B": Mu_B}},
            shears={"B_left": Vu_B1, "B_right": Vu_B2},
            section={"type": "T-beam" if section_type=="T-Beam" else "rectangular",
                     "bf": bf, "hf": hf, "bw": bw, "d": d, "dp": dp},
            materials={"fcu": fcu, "fy": fy})

        defl = deflection_check(
            L=L, d=d, beam_type="continuous", fy=fy,
            As_req=max(v["As_req"] for v in results["flexure"].values()),
            As_prov=max(v["As_prov"] for v in results["flexure"].values()))

        _render_design_results(
            {"Sagging AB": Mu_AB,"Sagging BC": Mu_BC,"Hogging B": Mu_B},
            results["flexure"], results["shear"], defl, wu)


def _render_design_results(moments, flex, shear, defl, wu):
    st.markdown("<div class='section-label'>Design Results</div>", unsafe_allow_html=True)

    # Moment summary
    moment_html = "<div class='metric-grid'>" + "".join(
        f"<div class='metric-box'><div class='val'>{v:.2f}</div><div class='lbl'>{k} (kNm)</div></div>"
        for k, v in moments.items()) + "</div>"
    st.markdown(moment_html, unsafe_allow_html=True)

    # Flexure per zone
    st.markdown("<div class='section-label'>Flexural Reinforcement</div>", unsafe_allow_html=True)
    for zone, data in flex.items():
        is_doubly = data.get("type") == "doubly"
        card_cls  = "warning" if is_doubly else ""
        tag       = "DOUBLY REINFORCED" if is_doubly else "SINGLY REINFORCED"
        extra = ""
        if is_doubly:
            extra = (f"&nbsp;|&nbsp; Asc req = <b style='color:{YEL}'>{data.get('Asc_req',0):.0f} mm²</b>"
                     f"&nbsp;|&nbsp; Top bars = <b style='color:{YEL}'>{data.get('compression_bars','—')}</b>")
        st.markdown(f"""
        <div class='result-card {card_cls}'>
            <b>{zone}</b> — <span style='color:#8B92A8;font-size:0.75rem'>{tag}</span><br>
            As req = <b style='color:{TEAL}'>{data['As_req']:.0f} mm²</b> &nbsp;|&nbsp;
            As prov = <b style='color:{GRN}'>{data['As_prov']:.0f} mm²</b> &nbsp;|&nbsp;
            Bars = <b style='color:{GRN}'>{data['tension_bars']}</b>{extra}
        </div>""", unsafe_allow_html=True)

    # Shear & deflection
    st.markdown("<div class='section-label'>Shear & Deflection</div>", unsafe_allow_html=True)
    defl_col = GRN if defl["status"] == "PASS" else RED
    defl_cls = "" if defl["status"] == "PASS" else "danger"

    st.markdown(f"""
    <div class='result-card warning'>
        Shear: <b style='color:{YEL}'>{shear['status']}</b> &nbsp;|&nbsp;
        v = <b>{shear['v']:.3f} N/mm²</b> &nbsp;|&nbsp;
        vc = <b>{shear['vc']:.3f} N/mm²</b> &nbsp;|&nbsp;
        Links = <b style='color:{YEL}'>{shear['links']}</b>
    </div>
    <div class='result-card {defl_cls}'>
        Deflection: <b style='color:{defl_col}'>{defl['status']}</b> &nbsp;|&nbsp;
        L/d actual = <b style='color:{defl_col}'>{defl['actual']:.1f}</b> &nbsp;|&nbsp;
        L/d limit = <b>{defl['allowable']:.1f}</b>
    </div>""", unsafe_allow_html=True)

    summary = format_design_summary(wu=wu, moments=moments, flexure=flex, shear=shear, deflection=defl)
    with st.expander("📋  Full Design Workings"):
        st.markdown(f"<div class='calc-steps'>{summary}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    module = _sidebar()
    if   "Beam"  in module: beam_page()
    elif "Frame" in module: frame_page()
    elif "RC"    in module: design_page()

if __name__ == "__main__":
    main()
