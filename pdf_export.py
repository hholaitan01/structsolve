"""
StructSolve — Professional PDF Export
ReportLab-based, A4, navy/teal/amber palette
"""
import io, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm

W, H   = A4
MARGIN = 2.0 * cm
IW     = W - 2 * MARGIN          # usable inner width ≈ 515 pt

# ── Palette ──────────────────────────────────────────────────
NAVY  = colors.HexColor("#1A2744")
TEAL  = colors.HexColor("#2E7D6E")
AMB   = colors.HexColor("#B07000")
RED_C = colors.HexColor("#8B2020")
LGREY = colors.HexColor("#F2F4F8")
MGREY = colors.HexColor("#C4C9D8")
DGREY = colors.HexColor("#464E64")
WHITE = colors.white


# ══════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════
def _styles():
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)
    return {
        "title":   ps("title",  fontName="Helvetica-Bold",  fontSize=18, textColor=NAVY,
                       spaceAfter=2,  leading=22),
        "sub":     ps("sub",    fontName="Helvetica",        fontSize=9,  textColor=DGREY,
                       spaceAfter=10, leading=13),
        "h1":      ps("h1",     fontName="Helvetica-Bold",   fontSize=12, textColor=NAVY,
                       spaceBefore=10,spaceAfter=4,  leading=15),
        "h2":      ps("h2",     fontName="Helvetica-Bold",   fontSize=10, textColor=TEAL,
                       spaceBefore=8, spaceAfter=3,  leading=13),
        "body":    ps("body",   fontName="Helvetica",        fontSize=8.5,textColor=DGREY,
                       spaceAfter=4,  leading=13),
        "eq":      ps("eq",     fontName="Courier",          fontSize=7.8,textColor=colors.black,
                       spaceAfter=1,  leading=11.5, leftIndent=10),
        "eqb":     ps("eqb",    fontName="Courier-Bold",     fontSize=7.8,textColor=NAVY,
                       spaceAfter=2,  leading=11.5, leftIndent=10),
        "lbl":     ps("lbl",    fontName="Helvetica-Bold",   fontSize=7,  textColor=TEAL,
                       spaceAfter=2,  leading=10,   spaceBefore=4),
        "cap":     ps("cap",    fontName="Helvetica-Oblique",fontSize=7.5,textColor=DGREY,
                       spaceAfter=6,  leading=11,   alignment=1),
        "mono":    ps("mono",   fontName="Courier",          fontSize=7,  textColor=DGREY,
                       spaceAfter=1,  leading=10.5),
    }


# ══════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════
def _doc(buf):
    return SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN)

def _hdr(title, subtitle, S):
    data = [[
        Paragraph(title, S["title"]),
        Paragraph(
            f"StructSolve · BS 8110-1:1997<br/>"
            f"{datetime.now().strftime('%d %B %Y  %H:%M')}",
            S["sub"])
    ]]
    t = Table(data, colWidths=[IW * 0.62, IW * 0.38])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), LGREY),
        ("LINEABOVE",     (0,0),(-1, 0), 3, NAVY),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    return t

def _rule(label, S):
    return [
        Spacer(1, 5),
        HRFlowable(width=IW, thickness=1.5, color=TEAL, spaceAfter=3),
        Paragraph(label.upper(), S["lbl"]),
    ]

def _tbl(hdrs, rows, widths=None):
    if not widths:
        widths = [IW / len(hdrs)] * len(hdrs)
    data = [hdrs] + [[str(c) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1, 0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1, 0), WHITE),
        ("FONTNAME",      (0,0),(-1, 0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("GRID",          (0,0),(-1,-1), 0.4, MGREY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGREY]),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("ALIGN",         (0,0),(0,-1),  "LEFT"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("RIGHTPADDING",  (0,0),(-1,-1), 5),
    ]))
    return t

def _fig_img(fig, width=None, height=None):
    if width is None: width = IW
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    kw = {"width": width}
    if height: kw["height"] = height
    return Image(buf, **kw)

def _wk_block(lines, S):
    """Render workings lines → list of Paragraph flowables."""
    el = []
    for line in lines:
        if line.startswith("══") or line.startswith("──"):
            el.append(HRFlowable(width=IW * 0.9, thickness=0.6, color=MGREY, spaceAfter=2))
        elif line.lstrip().startswith("STEP") or (line.startswith("  ") and line.strip().startswith("STEP")):
            el.append(Paragraph(line.strip(), S["h2"]))
        elif line == "":
            el.append(Spacer(1, 3))
        else:
            el.append(Paragraph(line.replace("<","&lt;").replace(">","&gt;"), S["eq"]))
    return el


# ══════════════════════════════════════════════════════════════
# MATPLOTLIB FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════
def _sfd_bmd_fig(x, v, m, span_Ls):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5))
    fig.patch.set_facecolor("white")
    for ax in (a1, a2):
        ax.set_facecolor("#F5F7FA")
        ax.grid(True, lw=0.5, ls="--", color="#CDD2DE")
        ax.tick_params(labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#CDD2DE")

    xv = np.array(x); vv = np.array(v); mv = np.array(m)

    a1.plot(xv, vv, lw=1.8, color="#2E7D6E")
    a1.fill_between(xv, vv, 0, where=vv >= 0, alpha=.15, color="#2E7D6E")
    a1.fill_between(xv, vv, 0, where=vv <  0, alpha=.15, color="#8B2020")
    a1.axhline(0, color="#888", lw=.8)
    a1.set_ylabel("Shear (kN)", fontsize=7)
    a1.set_title("Shear Force Diagram", fontsize=8, fontweight="bold",
                 color="#1A2744", pad=5)

    a2.plot(xv, mv, lw=1.8, color="#B07000")
    a2.fill_between(xv, mv, 0, where=mv >= 0, alpha=.15, color="#B07000")
    a2.fill_between(xv, mv, 0, where=mv <  0, alpha=.12, color="#8B2020")
    a2.axhline(0, color="#888", lw=.8)
    a2.invert_yaxis()
    a2.set_ylabel("Moment (kNm)", fontsize=7)
    a2.set_xlabel("x (m)", fontsize=7)
    a2.set_title("Bending Moment Diagram  [sagging +ve, plotted down]",
                 fontsize=8, fontweight="bold", color="#1A2744", pad=5)

    cx = 0.0
    for L in span_Ls:
        cx += L
        for ax in (a1, a2):
            ax.axvline(cx, color="#CDD2DE", lw=0.8, ls=":")

    for ax, y in [(a1, vv), (a2, mv)]:
        im = int(np.argmax(y)); iv = int(np.argmin(y))
        ax.annotate(f"{y[im]:.2f}", xy=(xv[im], y[im]), xytext=(3, 3),
                    textcoords="offset points", fontsize=6, color="#2E7D6E", fontweight="bold")
        ax.annotate(f"{y[iv]:.2f}", xy=(xv[iv], y[iv]), xytext=(3, -9),
                    textcoords="offset points", fontsize=6, color="#8B2020", fontweight="bold")

    fig.tight_layout(pad=1.2)
    return fig


def _frame_fig(nodes, members, results):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F5F7FA")
    ax.grid(True, lw=0.4, ls="--", color="#CDD2DE")
    for sp in ax.spines.values(): sp.set_edgecolor("#CDD2DE")

    labels = ["AB", "BC", "CD"]
    cls    = ["#2E7D6E", "#B07000", "#1A2744"]
    cmap   = {0:(0,1), 1:(1,2), 2:(2,3)}

    for idx, mem in enumerate(members):
        ni, nj = cmap[mem["id"]]
        xi, yi = nodes[ni]["x"], nodes[ni]["y"]
        xj, yj = nodes[nj]["x"], nodes[nj]["y"]
        ax.plot([xi,xj],[yi,yj], lw=3, color=cls[idx],
                solid_capstyle="round", label=labels[idx])

        if results and idx in results:
            ms, me = results[idx]
            Lm = np.hypot(xj-xi, yj-yi)
            if Lm > 0:
                dx,dy = (xj-xi)/Lm, (yj-yi)/Lm
                px,py = -dy, dx
                sc = 0.22 / max(abs(ms), abs(me), 1e-6)
                ts = np.linspace(0, 1, 30)
                bx = [xi+t*(xj-xi)+(ms*(1-t)+me*t)*sc*px for t in ts]
                by = [yi+t*(yj-yi)+(ms*(1-t)+me*t)*sc*py for t in ts]
                ax.fill([xi]+bx+[xj,xi],[yi]+by+[yj,yi], alpha=.12, color=cls[idx])
                ax.plot(bx, by, lw=1.2, color=cls[idx], alpha=.85)
                for xx,yy,val in [(xi,yi,ms),(xj,yj,me)]:
                    ax.annotate(f"{val:+.2f}", xy=(xx,yy), xytext=(5,5),
                                textcoords="offset points", fontsize=7,
                                color=cls[idx], fontweight="bold")

    for n in nodes:
        ax.plot(n["x"], n["y"], "o", color="#1A2744", ms=7, zorder=5)
        ax.text(n["x"]-0.15, n["y"], ["A","B","C","D"][n["id"]],
                fontsize=9, fontweight="bold", color="#1A2744",
                va="center", ha="right")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title("Portal Frame — BMD Overlay", fontsize=8, fontweight="bold",
                 color="#1A2744", pad=6)
    ax.legend(loc="upper right", fontsize=7, framealpha=.9)
    fig.tight_layout()
    return fig


def _section_fig(bw, D, cover, tension_bars_str,
                 section_type="rectangular", bf=None, hf=None,
                 compression_bars_str=""):
    def pdia(s):
        try: return int(str(s).split("Ø")[1].split()[0])
        except: return 16
    def pn(s):
        try: return max(int(str(s).split("Ø")[0].strip()), 2)
        except: return 3

    fig, ax = plt.subplots(figsize=(3.8, 4.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    face = "#D6E8F5"; edge = "#1A2744"

    if section_type == "T-beam" and bf and hf:
        ax.fill([-bw/2,bw/2,bw/2,-bw/2,-bw/2],[0,0,D,D,0],
                facecolor=face, edgecolor=edge, lw=1.5)
        ax.fill([-bf/2,bf/2,bf/2,-bf/2,-bf/2],[D,D,D+hf,D+hf,D],
                facecolor=face, edgecolor=edge, lw=1.5)
        total_D = D + hf
    else:
        ax.fill([-bw/2,bw/2,bw/2,-bw/2,-bw/2],[0,0,D,D,0],
                facecolor=face, edgecolor=edge, lw=1.5)
        total_D = D

    lk = cover
    ax.plot([-bw/2+lk,bw/2-lk,bw/2-lk,-bw/2+lk,-bw/2+lk],
            [lk,lk,D-lk,D-lk,lk], color="#2E7D6E", lw=1.8)

    # Tension bars
    bd = pdia(tension_bars_str); nb = pn(tension_bars_str)
    for xb in np.linspace(-bw/2+cover, bw/2-cover, nb):
        ax.add_patch(plt.Circle((xb, cover+bd/2), bd/2, color="#8B2020", zorder=5))

    # Compression bars
    if compression_bars_str:
        bdc = pdia(compression_bars_str); nbc = pn(compression_bars_str)
        for xb in np.linspace(-bw/2+cover, bw/2-cover, nbc):
            ax.add_patch(plt.Circle((xb, D-cover-bdc/2), bdc/2, color="#1A5FAD", zorder=5))

    off = cover * 1.8
    ax.annotate("", xy=(bw/2,-off), xytext=(-bw/2,-off),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1))
    ax.text(0, -off-cover*.8, f"b = {bw:.0f} mm",
            ha="center", va="top", fontsize=7.5, color="#333")

    ax.annotate("", xy=(bw/2+cover*2.5,D), xytext=(bw/2+cover*2.5,0),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1))
    ax.text(bw/2+cover*3.5, D/2, f"D = {D:.0f}",
            ha="left", va="center", fontsize=7.5, color="#333", rotation=90)

    ax.text(bw/2+cover*.6, D-cover, f"d={D-cover:.0f}",
            ha="left", va="top", fontsize=7, color="#B07000", alpha=.9, style="italic")

    import matplotlib.patches as mpatches, matplotlib.lines as mlines
    handles = [
        mpatches.Patch(facecolor=face, edgecolor=edge, label="Concrete"),
        mlines.Line2D([],[],color="#2E7D6E",lw=2,label="Links"),
        mpatches.Patch(color="#8B2020", label=f"Tension: {tension_bars_str}"),
    ]
    if compression_bars_str:
        handles.append(mpatches.Patch(color="#1A5FAD", label=f"Comp: {compression_bars_str}"))
    ax.legend(handles=handles, loc="upper right", fontsize=6.5, framealpha=.95)

    ax.set_aspect("equal")
    pad = max(bw, total_D) * .28
    ax.set_xlim(-bw/2-pad, bw/2+pad*2.4)
    ax.set_ylim(-cover*4, total_D+cover*2.5)
    ax.axis("off")
    ax.set_title("Cross-Section Detail", fontsize=9, fontweight="bold",
                 color="#1A2744", pad=8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# PUBLIC EXPORT FUNCTIONS
# ══════════════════════════════════════════════════════════════

def export_beam_analysis_pdf(spans, support_types, span_loads,
                              settlements, rotations,
                              thetas, fems, sway_corr,
                              internal_actions, support_df,
                              x, v, m, workings_lines):
    S = _styles(); buf = io.BytesIO(); doc = _doc(buf); el = []
    el += [_hdr("Continuous Beam Analysis","Slope Deflection Method", S), Spacer(1,10)]

    el += _rule("Beam Geometry", S)
    rows = [[f"{chr(65+i)}-{chr(66+i)}", f"{sp['L']:.3f}", f"{sp['EI']:.4f}",
             support_types[i], support_types[i+1]]
            for i,sp in enumerate(spans)]
    el += [_tbl(["Span","L (m)","EI","Near","Far"], rows,
                [IW*w for w in [.12,.16,.18,.27,.27]]), Spacer(1,5)]

    has_s = any(v2!=0 for v2 in settlements.values())
    has_r = any(v2!=0 for v2 in rotations.values())
    if has_s or has_r:
        parts = []
        if has_s: parts.append("Settlements: " + ", ".join(
            f"{chr(65+k)}={v2:.1f} mm" for k,v2 in settlements.items() if v2))
        if has_r: parts.append("Rotations: " + ", ".join(
            f"{chr(65+k)}={v2:.5f} rad" for k,v2 in rotations.items() if v2))
        el.append(Paragraph(" | ".join(parts), S["body"]))

    el += _rule("Applied Loads", S)
    lrows = []
    for i in range(len(spans)):
        for ld in span_loads.get(i,[]):
            t = ld["type"]
            if   t=="UDL":   d2 = f"{ld['mag']:.2f} kN/m (full span)"
            elif t=="UDL-P": d2 = f"{ld['mag']:.2f} kN/m, {ld['pos']:.2f}–{ld['end']:.2f} m"
            elif t=="UVL-P": d2 = f"{ld['mag']:.2f} kN/m peak, {ld.get('shape','')}"
            elif t=="Point": d2 = f"{ld['mag']:.2f} kN @ {ld['pos']:.2f} m"
            else: d2 = str(ld)
            lrows.append([f"{chr(65+i)}-{chr(66+i)}", t, d2])
    if lrows:
        el.append(_tbl(["Span","Type","Description"], lrows, [IW*w for w in [.14,.14,.72]]))
    else:
        el.append(Paragraph("No applied loads.", S["body"]))
    el.append(Spacer(1,5))

    el += _rule("Slope-Deflection Method — Full Workings", S)
    el.append(Paragraph(
        "General equation:  M_ij = M_Fij + (2EI/L)[2θᵢ + θⱼ − 3Δ/L]  "
        "where θ = joint rotation (rad), Δ = chord shortening (m)", S["body"]))
    el.append(Spacer(1,4))
    el += _wk_block(workings_lines, S)
    el.append(Spacer(1,6))

    el += _rule("Member End Moments", S)
    mr = [[k, f"{v2['M_start']:+.4f}", f"{v2['M_end']:+.4f}"]
          for k,v2 in internal_actions.items()]
    el += [_tbl(["Span","M_start (kNm)","M_end (kNm)"], mr, [IW/3]*3), Spacer(1,5)]

    el += _rule("Support Actions", S)
    hdrs2 = list(support_df.columns)
    rows2 = [list(r) for r in support_df.values.tolist()]
    el += [_tbl(hdrs2, rows2, [IW/len(hdrs2)]*len(hdrs2)), Spacer(1,8)]

    el += _rule("Shear Force & Bending Moment Diagrams", S)
    fig = _sfd_bmd_fig(x, v, m, [sp["L"] for sp in spans])
    el += [_fig_img(fig, width=IW, height=190),
           Paragraph("Fig. 1 — SFD (top) and BMD (bottom). Sagging plotted downward.", S["cap"])]

    doc.build(el); buf.seek(0); return buf


def export_beam_design_pdf(spans, support_types, span_loads,
                            settlements, rotations,
                            thetas, fems, sway_corr,
                            internal_actions, support_df,
                            x, v, m, workings_lines,
                            bw, D, cover, fcu, fy,
                            flex, shear, defl, design_workings_lines):
    S = _styles(); buf = io.BytesIO(); doc = _doc(buf); el = []
    el += [_hdr("Beam Analysis & RC Design","Slope Deflection Method  |  BS 8110-1:1997", S),
           Spacer(1,10)]

    # Analysis section
    el += _rule("Beam Geometry", S)
    rows = [[f"{chr(65+i)}-{chr(66+i)}", f"{sp['L']:.3f}", f"{sp['EI']:.4f}",
             support_types[i], support_types[i+1]]
            for i,sp in enumerate(spans)]
    el += [_tbl(["Span","L (m)","EI","Near","Far"], rows,
                [IW*w for w in [.12,.16,.18,.27,.27]]), Spacer(1,5)]

    el += _rule("Analysis Workings", S)
    el += _wk_block(workings_lines, S)
    el.append(Spacer(1,5))

    el += _rule("Member End Moments & Support Actions", S)
    mr = [[k, f"{v2['M_start']:+.4f}", f"{v2['M_end']:+.4f}"]
          for k,v2 in internal_actions.items()]
    el += [_tbl(["Span","M_start (kNm)","M_end (kNm)"], mr, [IW/3]*3), Spacer(1,4)]
    hdrs2 = list(support_df.columns)
    rows2 = [list(r) for r in support_df.values.tolist()]
    el += [_tbl(hdrs2, rows2, [IW/len(hdrs2)]*len(hdrs2)), Spacer(1,6)]

    el += _rule("Shear Force & Bending Moment Diagrams", S)
    fig = _sfd_bmd_fig(x, v, m, [sp["L"] for sp in spans])
    el += [_fig_img(fig, width=IW, height=185),
           Paragraph("Fig. 1 — SFD and BMD.", S["cap"])]

    el.append(PageBreak())
    el += [_hdr("RC Design — continued","BS 8110-1:1997", S), Spacer(1,10)]
    el += _rule("Design Workings  (BS 8110 Cl. 3.4)", S)
    el += _wk_block(design_workings_lines, S)
    el.append(Spacer(1,6))

    el += _rule("Flexural Reinforcement", S)
    fl_rows = []
    for zone, data in flex.items():
        tp = "Doubly" if data.get("type")=="doubly" else "Singly"
        fl_rows.append([zone, tp, f"{data['As_req']:.0f}", f"{data['As_prov']:.0f}",
                        data.get("tension_bars","—"),
                        data.get("compression_bars","—") if data.get("type")=="doubly" else "—"])
    el += [_tbl(["Zone","Type","As req","As prov","Tension bars","Comp bars"], fl_rows,
                [IW*w for w in [.14,.10,.12,.12,.27,.25]]), Spacer(1,4)]

    el += _rule("Shear & Deflection", S)
    sh = shear
    sr = [["v (N/mm²)", f"{sh['v']:.4f}", "vc (N/mm²)", f"{sh['vc']:.4f}"],
          ["Status",    sh['status'],      "Links",       sh['links']]]
    el += [_tbl(["Parameter","Value","Parameter","Value"], sr, [IW*w for w in [.27,.23,.27,.23]]),
           Spacer(1,4)]
    dr = [["Actual L/d", f"{defl['actual']:.2f}", "Allowable L/d", f"{defl['allowable']:.2f}"],
          ["Status",     defl['status'],           "",              ""]]
    el += [_tbl(["Parameter","Value","Parameter","Value"], dr, [IW*w for w in [.27,.23,.27,.23]]),
           Spacer(1,8)]

    el += _rule("Cross-Section Detail", S)
    first_zone = list(flex.values())[0]
    cs_type = "rectangular"
    fig_s = _section_fig(bw, D, cover, first_zone.get("tension_bars","3Ø16 bottom"),
                         section_type=cs_type,
                         compression_bars_str=first_zone.get("compression_bars","") if first_zone.get("type")=="doubly" else "")
    el += [_fig_img(fig_s, width=IW*0.46, height=180),
           Paragraph("Fig. 2 — Cross-section: tension bars (red), compression bars (blue), links (green).", S["cap"])]

    doc.build(el); buf.seek(0); return buf


def export_frame_pdf(h1, h2, L, ei_AB, ei_BC, ei_CD,
                     frame_type, case_type, loads,
                     results, unknowns, workings_lines, nodes, members):
    S = _styles(); buf = io.BytesIO(); doc = _doc(buf); el = []
    el += [_hdr("Portal Frame Analysis","Slope Deflection Method", S), Spacer(1,10)]

    el += _rule("Frame Geometry & Stiffness", S)
    gr = [["Left col AB height",  f"{h1:.3f} m", "EI_AB", f"{ei_AB:.4f}"],
          ["Beam BC span",        f"{L:.3f} m",  "EI_BC", f"{ei_BC:.4f}"],
          ["Right col CD height", f"{h2:.3f} m", "EI_CD", f"{ei_CD:.4f}"],
          ["Frame type",          frame_type,    "Case",  str(case_type)]]
    el += [_tbl(["Parameter","Value","Parameter","Value"], gr,
                [IW*w for w in [.32,.18,.32,.18]]), Spacer(1,5)]

    el += _rule("Applied Loads", S)
    mn = {0:"Column AB", 1:"Beam BC", 2:"Column CD"}
    lr = [[mn.get(ld["member"],"?"), ld["type"],
           f"{ld['mag']:.2f} kN/m" if ld["type"]=="UDL" else f"{ld['mag']:.2f} kN",
           f"{ld.get('pos',0):.3f} m"] for ld in loads]
    if lr:
        el.append(_tbl(["Member","Type","Magnitude","Position"], lr,
                       [IW*w for w in [.30,.18,.27,.25]]))
    else:
        el.append(Paragraph("No applied loads.", S["body"]))
    el.append(Spacer(1,5))

    el += _rule("Slope-Deflection Method — Full Workings", S)
    el += _wk_block(workings_lines, S)
    el.append(Spacer(1,6))

    el += _rule("Member End Moments", S)
    labels = ["AB","BC","CD"]
    rr = [[f"Member {labels[i]}", f"{ms:+.4f}", f"{me:+.4f}"]
          for i,(ms,me) in results.items()]
    el += [_tbl(["Member","M_start (kNm)","M_end (kNm)"], rr, [IW/3]*3), Spacer(1,4)]

    tB = unknowns[0]; tC = unknowns[1]; delta = unknowns[2] if len(unknowns)>2 else 0
    uk_rows = [["θ_B (rad)", f"{tB:.8f}", "θ_C (rad)", f"{tC:.8f}"],
               ["Sway Δ (m)", f"{delta:.8f}", "", ""]]
    el += [_tbl(["Unknown","Value","Unknown","Value"], uk_rows,
                [IW*w for w in [.27,.23,.27,.23]]), Spacer(1,8)]

    el += _rule("Frame Diagram with BMD Overlay", S)
    fig = _frame_fig(nodes, members, results)
    el += [_fig_img(fig, width=IW*0.65, height=210),
           Paragraph("Fig. 1 — Portal frame with BMD overlay. "
                     "Values at member ends in kNm (+ve anticlockwise).", S["cap"])]

    doc.build(el); buf.seek(0); return buf


def export_frame_pdf_general(nodes, members, loads, moments_out,
                             unknowns, workings, title="Frame Analysis"):
    """
    PDF export that works with ANY frame topology (v4.1).
    No hard-coded member count or layout.
    """
    S = _styles(); buf = io.BytesIO(); doc = _doc(buf); el = []
    el += [_hdr("Frame Analysis — Slope Deflection Method",
                f"{title}  |  v4.1", S), Spacer(1, 10)]

    # ── Geometry table ────────────────────────────────────────
    el += _rule("Frame Geometry", S)
    nmap = {n["id"]: n for n in nodes}
    node_rows = [[n.get("label", str(n["id"])), f"({n['x']:.2f}, {n['y']:.2f}) m",
                  n.get("support", "Free"), ""] for n in nodes]
    el += [_tbl(["Node", "Coords (x,y)", "Support", ""],
                node_rows, [IW*w for w in [.15, .25, .25, .35]]), Spacer(1, 4)]

    mem_rows = [[m.get("label", str(m["id"])),
                 nmap[m["ni"]].get("label","?"),
                 nmap[m["nj"]].get("label","?"),
                 f"{m['EI']:.1f}"] for m in members]
    el += [_tbl(["Member", "Near node", "Far node", "EI (kNm²)"],
                mem_rows, [IW*w for w in [.20, .20, .20, .40]]), Spacer(1, 5)]

    # ── Loads table ───────────────────────────────────────────
    el += _rule("Applied Loads", S)
    mfmt = {m["id"]: m.get("label", str(m["id"])) for m in members}
    lr = [[mfmt.get(ld["member_id"], str(ld["member_id"])), ld["type"],
           f"{ld['mag']:.2f}", f"{ld.get('pos', 0):.3f} m"] for ld in loads]
    if lr:
        el.append(_tbl(["Member", "Type", "Magnitude", "Position"],
                       lr, [IW*w for w in [.25, .20, .30, .25]]))
    else:
        el.append(Paragraph("No applied loads.", S["body"]))
    el.append(Spacer(1, 5))

    # ── Unknowns table ────────────────────────────────────────
    el += _rule("Solved Unknowns", S)
    uk_rows = [[k.replace("theta_", "\u03b8_").replace("delta_", "\u0394_"),
                f"{v:.8f}",
                "rad" if k.startswith("theta") else "m", ""]
               for k, v in unknowns.items()]
    if uk_rows:
        el += [_tbl(["Unknown", "Value", "Unit", ""],
                    uk_rows, [IW*w for w in [.20, .30, .15, .35]]), Spacer(1, 4)]

    # ── Workings ──────────────────────────────────────────────
    el += _rule("Slope-Deflection Method — Full Workings", S)
    all_wk = "\n\n".join(workings.get(k, "") for k in
                         ["fem", "sd_eqs", "equil", "solution", "final", "check"])
    el += _wk_block(all_wk.split("\n"), S)
    el.append(Spacer(1, 6))

    # ── End moment table ──────────────────────────────────────
    el += _rule("Member End Moments", S)
    rr = []
    for mem in members:
        mid = mem["id"]
        li = nmap[mem["ni"]].get("label", "?")
        lj = nmap[mem["nj"]].get("label", "?")
        Mij, Mji = moments_out.get(mid, (0.0, 0.0))
        rr.append([mem.get("label", str(mid)),
                   f"M_{li}{lj} = {Mij:+.4f} kNm",
                   f"M_{lj}{li} = {Mji:+.4f} kNm"])
    el += [_tbl(["Member", "Near-end Moment", "Far-end Moment"],
                rr, [IW*w for w in [.20, .40, .40]]), Spacer(1, 8)]

    # ── Frame diagram ─────────────────────────────────────────
    el += _rule("Frame Diagram with BMD Overlay", S)
    fig = _frame_fig_general(nodes, members, moments_out)
    el += [_fig_img(fig, width=IW * 0.70, height=230),
           Paragraph("BMD Overlay — end moments annotated in kNm.", S["cap"])]

    doc.build(el); buf.seek(0); return buf


def _frame_fig_general(nodes, members, results):
    """General BMD figure for any frame topology."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#F5F7FA")
    ax.grid(True, lw=0.4, ls="--", color="#CDD2DE")
    for sp in ax.spines.values(): sp.set_edgecolor("#CDD2DE")

    nmap = {n["id"]: n for n in nodes}
    cols = ["#2E7D6E", "#B07000", "#1A2744", "#8B2020", "#7B3F9E", "#1A6B8A"]

    all_m = [abs(v) for tup in results.values() for v in tup if tup]
    max_m = max(all_m) if all_m else 1.0
    all_x = [n["x"] for n in nodes]; all_y = [n["y"] for n in nodes]
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)
    bmd_sc = 0.22 * span / max(max_m, 1e-6)

    for idx, mem in enumerate(members):
        c = cols[idx % len(cols)]
        ni = nmap[mem["ni"]]; nj = nmap[mem["nj"]]
        xi, yi = ni["x"], ni["y"]; xj, yj = nj["x"], nj["y"]
        ax.plot([xi, xj], [yi, yj], lw=3, color=c,
                solid_capstyle="round", label=mem.get("label", str(mem["id"])))

        if mem["id"] in results:
            Mij, Mji = results[mem["id"]]
            Lm = np.hypot(xj - xi, yj - yi)
            if Lm > 0:
                dx2, dy2 = (xj - xi) / Lm, (yj - yi) / Lm
                px2, py2 = -dy2, dx2
                ts = np.linspace(0, 1, 40)
                bx = [xi+t*(xj-xi)+(Mij*(1-t)+Mji*t)*bmd_sc*px2 for t in ts]
                by = [yi+t*(yj-yi)+(Mij*(1-t)+Mji*t)*bmd_sc*py2 for t in ts]
                ax.fill([xi]+bx+[xj, xi], [yi]+by+[yj, yi], alpha=0.12, color=c)
                ax.plot(bx, by, lw=1.2, color=c, alpha=0.85)
                for xx, yy, val in [(xi, yi, Mij), (xj, yj, Mji)]:
                    ax.annotate(f"{val:+.2f}", xy=(xx, yy), xytext=(5, 5),
                                textcoords="offset points", fontsize=7.5,
                                color=c, fontweight="bold")

    for n in nodes:
        ax.plot(n["x"], n["y"], "o", color="#1A2744", ms=7, zorder=5)
        ax.text(n["x"] - 0.1, n["y"] + 0.05,
                n.get("label", str(n["id"])),
                fontsize=8, fontweight="bold", color="#1A2744", va="bottom")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title("Frame — BMD Overlay", fontsize=8, fontweight="bold",
                 color="#1A2744", pad=6)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig


def export_rc_design_pdf(beam_system, section_type, L, bw, D, cover,
                          fcu, fy, bf, hf, gk, qk, wu,
                          flex, shear, defl, workings_lines):
    S = _styles(); buf = io.BytesIO(); doc = _doc(buf); el = []
    el += [_hdr("RC Beam Design","BS 8110-1:1997  |  Flexure · Shear · Deflection", S),
           Spacer(1,10)]

    el += _rule("Design Parameters", S)
    pr = [["System",       beam_system,   "Section",    section_type],
          ["Span L",       f"{L:.2f} m",  "Cover",      f"{cover:.0f} mm"],
          ["Web width b",  f"{bw:.0f} mm","Depth D",    f"{D:.0f} mm"],
          ["fcu",          f"{fcu:.0f} N/mm²","fy",     f"{fy:.0f} N/mm²"],
          ["gk",           f"{gk:.2f} kN/m","qk",       f"{qk:.2f} kN/m"],
          ["wu = 1.4gk+1.6qk",f"{wu:.3f} kN/m",
           "bf / hf",      (f"{bf:.0f} / {hf:.0f} mm" if bf else "N/A")]]
    el += [_tbl(["Parameter","Value","Parameter","Value"], pr,
                [IW*w for w in [.28,.22,.28,.22]]), Spacer(1,5)]

    el += _rule("Design Workings  (BS 8110)", S)
    el += _wk_block(workings_lines, S)
    el.append(Spacer(1,6))

    el += _rule("Flexural Reinforcement", S)
    fl_rows = []
    for zone, data in flex.items():
        tp = "Doubly" if data.get("type")=="doubly" else "Singly"
        fl_rows.append([zone, tp, f"{data['As_req']:.0f}", f"{data['As_prov']:.0f}",
                        data.get("tension_bars","—"),
                        data.get("compression_bars","—") if data.get("type")=="doubly" else "—"])
    el += [_tbl(["Zone","Type","As req","As prov","Tension bars","Comp bars"], fl_rows,
                [IW*w for w in [.14,.10,.12,.12,.27,.25]]), Spacer(1,4)]

    el += _rule("Shear Design  (Cl. 3.4.5)", S)
    sh = shear
    sr = [["Applied v", f"{sh['v']:.4f} N/mm²", "vc (Table 3.8)", f"{sh['vc']:.4f} N/mm²"],
          ["Status",    sh['status'],             "Links",           sh['links']]]
    el += [_tbl(["Parameter","Value","Parameter","Value"], sr,
                [IW*w for w in [.27,.23,.27,.23]]), Spacer(1,4)]

    el += _rule("Deflection Check  (Cl. 3.4.6)", S)
    dr = [["Actual L/d", f"{defl['actual']:.2f}", "Allowable L/d", f"{defl['allowable']:.2f}"],
          ["Status", defl['status'], "", ""]]
    el += [_tbl(["Parameter","Value","Parameter","Value"], dr,
                [IW*w for w in [.27,.23,.27,.23]]), Spacer(1,8)]

    el += _rule("Cross-Section Detail", S)
    first_zone = list(flex.values())[0]
    cs_type = "T-beam" if "T-Beam" in section_type or "T-beam" in section_type else "rectangular"
    fig_s = _section_fig(bw, D, cover,
                         first_zone.get("tension_bars","3Ø16 bottom"),
                         section_type=cs_type, bf=bf, hf=hf,
                         compression_bars_str=first_zone.get("compression_bars","") if first_zone.get("type")=="doubly" else "")
    el += [_fig_img(fig_s, width=IW*0.50, height=195),
           Paragraph("Fig. 1 — Cross-section: tension bars (red), "
                     "compression bars (blue), shear links (green).", S["cap"])]

    doc.build(el); buf.seek(0); return buf


# ── Legacy wrappers (keep old callers working) ────────────────
def export_analysis_pdf(support_df, x, v, m):
    S = _styles(); buf = io.BytesIO(); doc = _doc(buf); el = []
    el += [_hdr("Beam Analysis","SFD & BMD", S), Spacer(1,8)]
    el += _rule("Support Actions", S)
    hdrs = list(support_df.columns); rows = [list(r) for r in support_df.values.tolist()]
    el += [_tbl(hdrs, rows, [IW/len(hdrs)]*len(hdrs)), Spacer(1,8)]
    el += _rule("Diagrams", S)
    fig = _sfd_bmd_fig(x, v, m, [])
    el += [_fig_img(fig, width=IW, height=195)]
    doc.build(el); buf.seek(0); return buf

def export_analysis_design_pdf(support_df, x, v, m, design_summary):
    S = _styles(); buf = io.BytesIO(); doc = _doc(buf); el = []
    el += [_hdr("Beam Analysis & Design","BS 8110-1:1997", S), Spacer(1,8)]
    el += _rule("Support Actions", S)
    hdrs = list(support_df.columns); rows = [list(r) for r in support_df.values.tolist()]
    el += [_tbl(hdrs, rows, [IW/len(hdrs)]*len(hdrs)), Spacer(1,8)]
    el += _rule("Diagrams", S)
    fig = _sfd_bmd_fig(x, v, m, [])
    el += [_fig_img(fig, width=IW, height=190), Spacer(1,8)]
    el += _rule("Design Summary", S)
    for line in design_summary.split("\n"):
        el.append(Paragraph(line or " ", S["mono"]))
    doc.build(el); buf.seek(0); return buf
