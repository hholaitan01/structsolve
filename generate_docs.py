"""
StructSolve — Technical Documentation PDF Generator
Generates a comprehensive engineering-grade technical manual.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem, KeepTogether,
)
from reportlab.lib import colors

# ── Colors ────────────────────────────────────────────────────
NAVY   = HexColor("#1A2744")
TEAL   = HexColor("#2E7D6E")
GREY   = HexColor("#4A5568")
LGREY  = HexColor("#E2E8F0")
WHITE  = HexColor("#FFFFFF")

# ── Styles ────────────────────────────────────────────────────
_base = getSampleStyleSheet()

def _s(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=_base[parent], **kw)

styles = {
    "title":     _s("DocTitle",   fontSize=28, leading=34, textColor=NAVY,
                     alignment=TA_CENTER, spaceAfter=6),
    "subtitle":  _s("DocSub",     fontSize=13, leading=18, textColor=GREY,
                     alignment=TA_CENTER, spaceAfter=20),
    "h1":        _s("H1",         fontSize=18, leading=24, textColor=NAVY,
                     spaceBefore=24, spaceAfter=10, fontName="Helvetica-Bold"),
    "h2":        _s("H2",         fontSize=14, leading=18, textColor=TEAL,
                     spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold"),
    "h3":        _s("H3",         fontSize=11, leading=15, textColor=NAVY,
                     spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"),
    "body":      _s("Body",       fontSize=10, leading=14, textColor=GREY,
                     alignment=TA_JUSTIFY, spaceAfter=6),
    "code":      _s("CodeBlock",  fontSize=8.5, leading=11, fontName="Courier",
                     textColor=NAVY, backColor=HexColor("#F7FAFC"),
                     borderPadding=6, spaceAfter=8),
    "bullet":    _s("Bullet",     fontSize=10, leading=14, textColor=GREY,
                     leftIndent=18, bulletIndent=6, spaceAfter=3),
    "footnote":  _s("Footnote",   fontSize=8,  leading=10, textColor=GREY,
                     alignment=TA_CENTER),
    "toc":       _s("TOC",        fontSize=11, leading=16, textColor=NAVY,
                     leftIndent=12, spaceAfter=4),
}

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=LGREY,
                      spaceBefore=6, spaceAfter=10)

def bullet_list(items):
    return [Paragraph(f"• {item}", styles["bullet"]) for item in items]

def code_block(text):
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(escaped.replace("\n", "<br/>"), styles["code"])

def table_block(headers, rows, col_widths=None):
    data = [headers] + rows
    w = col_widths or [None] * len(headers)
    t = Table(data, colWidths=w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("TEXTCOLOR",  (0, 1), (-1, -1), GREY),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("GRID",       (0, 0), (-1, -1), 0.4, LGREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, HexColor("#F7FAFC")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return t


def build_doc():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "StructSolve_Documentation.pdf")
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
        title="StructSolve Technical Documentation",
        author="StructSolve Team",
    )
    S = []  # story

    # ══════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════
    S.append(Spacer(1, 80))
    S.append(Paragraph("🏗️ StructSolve", styles["title"]))
    S.append(Paragraph("Technical Documentation", _s("CoverSub",
        fontSize=16, leading=20, textColor=TEAL, alignment=TA_CENTER, spaceAfter=30)))
    S.append(Paragraph("Structural Analysis &amp; RC Design Suite", styles["subtitle"]))
    S.append(hr())
    S.append(Paragraph(
        "Version 1.0  •  BS 8110-1:1997  •  Slope-Deflection Method  •  Matrix Stiffness",
        styles["footnote"]))
    S.append(Spacer(1, 40))
    S.append(Paragraph(
        "A browser-based structural engineering tool for analyzing continuous beams, "
        "plane frames, and designing reinforced concrete members. Built with Python, "
        "Streamlit, and verified against 52 textbook benchmark problems.",
        _s("CoverBody", fontSize=11, leading=16, textColor=GREY,
           alignment=TA_CENTER, spaceAfter=12)))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("Table of Contents", styles["h1"]))
    S.append(hr())
    toc_items = [
        "1.  Overview",
        "2.  System Architecture",
        "3.  Module Reference",
        "    3.1  beam_solver.py — Beam Analysis Engine",
        "    3.2  frame_solver.py — Frame Analysis Engine",
        "    3.3  bs8110.py — RC Design (BS 8110)",
        "    3.4  combine.py — Application Interface",
        "    3.5  pdf_export.py — PDF Report Generation",
        "    3.6  homepage.py — Landing Page",
        "4.  Analysis Methods",
        "    4.1  Slope-Deflection Method",
        "    4.2  Cantilever Span Handling",
        "    4.3  Critical Ordinate Computation",
        "5.  RC Design Methods (BS 8110)",
        "6.  Sign Conventions",
        "7.  Input / Output Reference",
        "8.  Verification &amp; Testing",
        "9.  Deployment",
    ]
    for item in toc_items:
        S.append(Paragraph(item, styles["toc"]))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 1. OVERVIEW
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("1.  Overview", styles["h1"]))
    S.append(hr())
    S.append(Paragraph(
        "StructSolve is an open-source, browser-based structural engineering application "
        "built with Python and Streamlit. It provides three core modules:", styles["body"]))
    S += bullet_list([
        "<b>Beam Analysis</b> — Continuous beams using the slope-deflection method",
        "<b>Frame Analysis</b> — Plane frames (sway and non-sway) using the matrix stiffness method",
        "<b>RC Design</b> — Reinforced concrete design to BS 8110-1:1997",
    ])
    S.append(Spacer(1, 8))
    S.append(Paragraph("<b>Key Capabilities:</b>", styles["body"]))
    S += bullet_list([
        "Ordinate-driven SFD and BMD with labelled critical points",
        "Full step-by-step workings (FEMs, SDEs, equilibrium)",
        "Cantilever spans with modified stiffness handling",
        "PDF report export with diagrams and calculation summaries",
        "Verified against 52 benchmark problems (Kassimali textbook)",
    ])
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Technology Stack:</b>", styles["body"]))
    S.append(table_block(
        ["Component", "Technology"],
        [["Language", "Python 3.11+"],
         ["Web Framework", "Streamlit"],
         ["Numerical", "NumPy"],
         ["Plotting", "Matplotlib"],
         ["PDF Generation", "ReportLab"],
         ["Data Tables", "Pandas"]],
        col_widths=[120, 300]))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 2. SYSTEM ARCHITECTURE
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("2.  System Architecture", styles["h1"]))
    S.append(hr())
    S.append(Paragraph(
        "The application follows a modular architecture with clear separation between "
        "solver engines, UI logic, and export functionality:", styles["body"]))
    S.append(Spacer(1, 8))
    S.append(table_block(
        ["File", "Role", "Lines"],
        [["beam_solver.py", "Beam analysis engine (SDE method, critical points)", "~540"],
         ["frame_solver.py", "Frame analysis engine (matrix stiffness)", "~900"],
         ["bs8110.py", "RC design calculations to BS 8110", "~400"],
         ["combine.py", "Streamlit UI, plotting, workings display", "~1850"],
         ["pdf_export.py", "PDF report generation", "~900"],
         ["homepage.py", "Landing page module", "~360"],
         ["test_kassimali.py", "Verification test suite (52 checks)", "~750"]],
        col_widths=[110, 280, 50]))
    S.append(Spacer(1, 10))
    S.append(Paragraph("<b>Data Flow:</b>", styles["body"]))
    S.append(code_block(
        "User Input (Streamlit UI)\n"
        "    ↓\n"
        "BeamSolver.solve_continuous_beam()  or  FrameSolver.solve()\n"
        "    ↓  returns: thetas, fems\n"
        "Back-substitution → end moments (M_ab, M_ba)\n"
        "    ↓\n"
        "BeamSolver.get_diagram_data()      → x, V, M arrays\n"
        "BeamSolver.get_critical_points()   → labelled ordinates\n"
        "    ↓\n"
        "plot_sfd_bmd() / beam_workings()   → display in Streamlit\n"
        "    ↓\n"
        "pdf_export.export_*_pdf()          → downloadable PDF"))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 3. MODULE REFERENCE
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("3.  Module Reference", styles["h1"]))
    S.append(hr())

    # -- 3.1 beam_solver --
    S.append(Paragraph("3.1  beam_solver.py — Beam Analysis Engine", styles["h2"]))
    S.append(Paragraph(
        "Core solver class <font face='Courier' size='9'>BeamSolver</font> implementing "
        "the slope-deflection method for continuous beams.", styles["body"]))
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Key Methods:</b>", styles["body"]))
    S.append(table_block(
        ["Method", "Description"],
        [["solve_continuous_beam()", "Main solver. Assembles SDE matrix, handles cantilevers "
          "with modified stiffness (3EI/L), solves for joint rotations."],
         ["beam_fixed_end_moments()", "Computes FEMs for Point, UDL, UDL-P, UVL-P loads "
          "on a fixed-fixed beam."],
         ["get_diagram_data()", "Generates V(x) and M(x) arrays for plotting using "
          "equilibrium-based section cuts at 200 points per span."],
         ["get_critical_points()", "Analytically computes all critical ordinates: "
          "supports, point loads, zero-shear (M max/min), contraflexure (M=0)."],
         ["_eval_vm()", "Evaluates V and M at any single point using equilibrium."],
         ["_cantilever_root_moment()", "Static moment at cantilever root by statics."]],
        col_widths=[150, 290]))
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Parameters for solve_continuous_beam():</b>", styles["body"]))
    S.append(table_block(
        ["Parameter", "Type", "Description"],
        [["n", "int", "Number of joints (nodes) = spans + 1"],
         ["spans", "list[dict]", "{'L': float, 'EI': float} for each span"],
         ["support_types", "list[str]", "Fixed | Roller | Pinned | Free (length n)"],
         ["span_loads", "dict", "{span_idx: [load_dicts]} with type, mag, pos, end"],
         ["sway_corrections", "dict", "{span_idx: delta} chord rotation from settlement"],
         ["prescribed_rotations", "dict", "{node_idx: theta_rad}"]],
        col_widths=[120, 80, 240]))
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Returns:</b>", styles["body"]))
    S += bullet_list([
        "<font face='Courier'>thetas</font> — ndarray of joint rotations (patched for cantilevers)",
        "<font face='Courier'>fems</font> — list of [M_ab, M_ba] (patched for cantilever back-substitution)",
    ])

    # -- 3.2 frame_solver --
    S.append(Paragraph("3.2  frame_solver.py — Frame Analysis Engine", styles["h2"]))
    S.append(Paragraph(
        "Class <font face='Courier' size='9'>FrameSolver</font> implementing the matrix "
        "stiffness (direct stiffness) method for plane frames with optional sway.", styles["body"]))
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Capabilities:</b>", styles["body"]))
    S += bullet_list([
        "Arbitrary plane frame geometry with inclined members",
        "Fixed, pinned, and roller supports",
        "Sway analysis with story-level side-sway DOFs",
        "Point loads, UDLs on any member",
        "End moment, shear, and axial force output",
    ])

    # -- 3.3 bs8110 --
    S.append(Paragraph("3.3  bs8110.py — RC Design (BS 8110-1:1997)", styles["h2"]))
    S.append(Paragraph(
        "Reinforced concrete design functions implementing clauses from "
        "BS 8110-1:1997.", styles["body"]))
    S.append(table_block(
        ["Function", "Clause", "Description"],
        [["beam_flexural_design()", "3.4.4", "Singly/doubly reinforced rectangular beam design"],
         ["beam_flexural_design_T()", "3.4.4", "Flanged (T/L) beam flexural design"],
         ["beam_shear_design()", "3.4.5", "Shear reinforcement design with v_c tables"],
         ["deflection_check()", "3.4.6", "Span/depth check with modification factors"],
         ["design_continuous_beam()", "—", "Auto-design all spans from beam analysis output"],
         ["format_design_summary()", "—", "Format results for display"]],
        col_widths=[140, 50, 250]))

    # -- 3.4 combine --
    S.append(Paragraph("3.4  combine.py — Application Interface", styles["h2"]))
    S.append(Paragraph(
        "Main Streamlit application file (~1850 lines). Contains the UI layout, "
        "input forms, plotting functions, and results display for all modules.", styles["body"]))
    S.append(Paragraph("<b>Key Functions:</b>", styles["body"]))
    S.append(table_block(
        ["Function", "Description"],
        [["beam_page()", "Full beam analysis UI (input, solve, diagrams, workings, design)"],
         ["frame_page()", "Frame analysis UI with geometry builder and BMD overlay"],
         ["design_page()", "Standalone RC design page"],
         ["plot_sfd_bmd()", "Ordinate-driven SFD/BMD plotting with critical point labels"],
         ["draw_beam_system()", "Visual beam diagram with supports and loads"],
         ["beam_workings()", "Step-by-step SDE workings display"],
         ["_sidebar()", "Navigation radio with module selection"]],
        col_widths=[140, 300]))

    # -- 3.5 pdf_export --
    S.append(Paragraph("3.5  pdf_export.py — PDF Report Generation", styles["h2"]))
    S.append(Paragraph(
        "ReportLab-based PDF export with engineering-grade formatting. "
        "Generates downloadable reports for beam analysis, frame analysis, and RC design.", styles["body"]))
    S.append(table_block(
        ["Function", "Description"],
        [["export_beam_analysis_pdf()", "Full beam report with diagrams, ordinates, support actions"],
         ["export_beam_design_pdf()", "RC design report linked to beam analysis"],
         ["export_frame_pdf()", "Frame analysis report with member forces"],
         ["export_rc_design_pdf()", "Standalone RC design report"],
         ["_sfd_bmd_fig()", "Matplotlib figure builder with ordinate annotations"]],
        col_widths=[160, 280]))

    # -- 3.6 homepage --
    S.append(Paragraph("3.6  homepage.py — Landing Page", styles["h2"]))
    S.append(Paragraph(
        "Professional homepage with feature cards, workflow steps, audience sections, "
        "and navigation buttons that route to each module.", styles["body"]))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 4. ANALYSIS METHODS
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("4.  Analysis Methods", styles["h1"]))
    S.append(hr())

    S.append(Paragraph("4.1  Slope-Deflection Method", styles["h2"]))
    S.append(Paragraph(
        "The beam solver implements the classical slope-deflection method. "
        "For each span, the end moments are expressed as:", styles["body"]))
    S.append(code_block(
        "M_ij = FEM_ij + (2EI/L)(2θ_i + θ_j − 3Δ/L)\n"
        "M_ji = FEM_ji + (2EI/L)(2θ_j + θ_i − 3Δ/L)"))
    S.append(Paragraph("where:", styles["body"]))
    S += bullet_list([
        "<font face='Courier'>FEM_ij, FEM_ji</font> = fixed-end moments from applied loads",
        "<font face='Courier'>θ_i, θ_j</font> = joint rotations (unknowns)",
        "<font face='Courier'>Δ</font> = relative settlement (chord rotation correction)",
        "<font face='Courier'>E, I, L</font> = material and geometric properties",
    ])
    S.append(Paragraph(
        "Joint equilibrium (ΣM = 0 at each free-rotation node) yields a system of "
        "linear equations solved by NumPy's <font face='Courier'>linalg.solve()</font>.",
        styles["body"]))

    S.append(Paragraph("4.2  Cantilever Span Handling", styles["h2"]))
    S.append(Paragraph(
        "For spans with a free end (cantilever overhang), the solver uses "
        "<b>modified stiffness</b> at the root joint:", styles["body"]))
    S.append(code_block(
        "Standard stiffness:  4EI/L  (far end fixed)\n"
        "Modified stiffness:  3EI/L  (far end free)\n\n"
        "Modified SDE: M_near = (3EI/L)·θ_root + FEM_near − FEM_far/2 − M_static\n\n"
        "where M_static = cantilever root moment from applied loads (by statics)"))
    S.append(Paragraph(
        "The far-end rotation θ_D is condensed out analytically using M_DC = 0, "
        "giving θ_D = −(FEM_DC + 2EI/L · θ_C) / (4EI/L). This eliminates one "
        "unknown per cantilever span from the system.", styles["body"]))

    S.append(Paragraph("4.3  Critical Ordinate Computation", styles["h2"]))
    S.append(Paragraph(
        "The <font face='Courier'>get_critical_points()</font> method computes ordinate "
        "values at all locations needed for exam-quality SFD/BMD diagrams:", styles["body"]))
    S.append(table_block(
        ["Critical Point", "How Found", "Label"],
        [["Span start/end", "x = 0, x = L", "Support"],
         ["Point load locations", "x = a (load position)", "Point load"],
         ["Partial load boundaries", "x = a, x = b", "Load boundary"],
         ["Zero shear (V = 0)", "Analytical: x = x₁ + V₁/w_net", "V=0 (M max) ★"],
         ["Contraflexure (M = 0)", "Bisection between sign changes", "M=0 ○"]],
        col_widths=[120, 200, 120]))
    S.append(Spacer(1, 6))
    S.append(Paragraph(
        "Zero-shear points are found analytically under uniform loading (V is linear) "
        "and by interpolation under point loads. Contraflexure points use 50-iteration "
        "bisection on the equilibrium-based M(x) function.", styles["body"]))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 5. RC DESIGN METHODS
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("5.  RC Design Methods (BS 8110)", styles["h1"]))
    S.append(hr())
    S.append(Paragraph(
        "All RC design follows BS 8110-1:1997 with the following assumptions:", styles["body"]))
    S += bullet_list([
        "Rectangular stress block (clause 3.4.4.4)",
        "Maximum neutral axis depth: x/d ≤ 0.5 (moment redistribution ≤ 10%)",
        "Concrete: C25, C30, C32, or C40 (f_cu in N/mm²)",
        "Steel: fy = 460 N/mm² (high-yield), fy = 250 N/mm² (mild)",
    ])
    S.append(Paragraph("<b>Flexural Design Procedure:</b>", styles["h3"]))
    S.append(code_block(
        "1. K = M / (b·d²·f_cu)\n"
        "2. If K ≤ K' = 0.156 → singly reinforced\n"
        "     z = d[0.5 + √(0.25 − K/0.9)]  (capped at 0.95d)\n"
        "     A_s = M / (0.87·f_y·z)\n"
        "3. If K > K' → doubly reinforced\n"
        "     A_s' = (K − K')·f_cu·b·d² / (0.87·f_y·(d − d'))\n"
        "     A_s  = K'·f_cu·b·d² / (0.87·f_y·z') + A_s'"))
    S.append(Paragraph("<b>Shear Design (Cl. 3.4.5):</b>", styles["h3"]))
    S.append(code_block(
        "v = V / (b·d)\n"
        "v_c = 0.79·(100·A_s/(b·d))^(1/3)·(400/d)^(1/4) / γ_m\n"
        "If v ≤ v_c/2:      no shear links required\n"
        "If v_c/2 < v ≤ v_c: minimum links A_sv/s = 0.4·b/(0.87·f_yv)\n"
        "If v > v_c:         A_sv/s = b·(v − v_c) / (0.87·f_yv)"))
    S.append(Paragraph("<b>Deflection Check (Cl. 3.4.6):</b>", styles["h3"]))
    S.append(code_block(
        "Basic span/depth ratio (Table 3.9):\n"
        "  Cantilever: 7    Simply supported: 20    Continuous: 26\n\n"
        "Tension steel modification factor (Table 3.10):\n"
        "  f_s = 2·f_y·A_s,req / (3·A_s,prov) × β_b\n"
        "  MF_t = 0.55 + (477 − f_s) / (120·(0.9 + M/(b·d²)))\n\n"
        "Allowable L/d = basic ratio × MF_t × MF_c\n"
        "Check: actual L/d ≤ allowable L/d"))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 6. SIGN CONVENTIONS
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("6.  Sign Conventions", styles["h1"]))
    S.append(hr())
    S.append(table_block(
        ["Quantity", "Positive Direction", "Notes"],
        [["End moment M_ij", "Clockwise on near end (i)", "Standard SDE convention"],
         ["FEM (UDL)", "Hogging = negative at near end", "FEM_AB = −wL²/12"],
         ["Shear force V", "Upward on left face", "V at x=0 = left reaction"],
         ["Bending moment M", "Sagging positive (solver)", "Negated for display (hogging ↑)"],
         ["Joint rotation θ", "Clockwise positive", "Consistent with SDE"],
         ["Loads", "Downward positive", "Gravity convention"]],
        col_widths=[110, 160, 170]))
    S.append(Spacer(1, 8))
    S.append(Paragraph(
        "<b>Display Convention:</b> In the BMD plot, the solver's sagging-positive moments "
        "are negated so that hogging plots upward and sagging plots downward (inverted y-axis). "
        "This matches the standard engineering drawing convention where sagging is drawn "
        "on the tension face.", styles["body"]))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 7. INPUT/OUTPUT REFERENCE
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("7.  Input / Output Reference", styles["h1"]))
    S.append(hr())
    S.append(Paragraph("<b>Load Types:</b>", styles["h3"]))
    S.append(table_block(
        ["Type Key", "Parameters", "Description"],
        [["Point", "mag, pos", "Concentrated load at distance 'pos' from left end"],
         ["UDL", "mag", "Uniform load over entire span"],
         ["UDL-P", "mag, pos, end", "Partial UDL from 'pos' to 'end'"],
         ["UVL-P", "mag, pos, end, shape", "Triangular load; shape = start_zero or end_zero"],
         ["Moment", "mag, pos", "Applied moment at distance 'pos'"]],
        col_widths=[70, 110, 260]))
    S.append(Spacer(1, 8))
    S.append(Paragraph("<b>Support Types:</b>", styles["h3"]))
    S.append(table_block(
        ["Type", "θ", "Δ", "M", "Description"],
        [["Fixed", "= 0", "= 0", "≠ 0", "Fully restrained (θ = 0)"],
         ["Roller", "≠ 0", "= 0", "= 0", "Free rotation, ΣM = 0"],
         ["Pinned", "≠ 0", "= 0", "= 0", "Same as Roller for beams"],
         ["Free", "≠ 0", "≠ 0", "= 0", "Cantilever tip (M = V = 0)"]],
        col_widths=[60, 50, 50, 50, 230]))
    S.append(Spacer(1, 8))
    S.append(Paragraph("<b>Units:</b>", styles["h3"]))
    S.append(table_block(
        ["Quantity", "Unit"],
        [["Length (geometry)", "m"],
         ["Force / load", "kN, kN/m"],
         ["Moment", "kNm"],
         ["Stress", "N/mm²"],
         ["Section dimensions", "mm"],
         ["Reinforcement area", "mm²"]],
        col_widths=[140, 300]))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 8. VERIFICATION & TESTING
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("8.  Verification &amp; Testing", styles["h1"]))
    S.append(hr())
    S.append(Paragraph(
        "The solver is verified against problems from Aslam Kassimali's "
        "<i>Structural Analysis</i> textbook using an automated test suite "
        "(<font face='Courier'>test_kassimali.py</font>).", styles["body"]))
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Test Summary: 52/52 checks passed ✓</b>", styles["h3"]))
    S.append(Spacer(1, 4))
    S.append(table_block(
        ["Category", "Count", "Description"],
        [["Beam — basic", "6", "Fixed-fixed beams with UDL and point loads"],
         ["Beam — propped cantilever", "4", "Fixed-Free with various loads"],
         ["Beam — multi-span", "8", "Two/three-span with mixed supports and EI"],
         ["Beam — pure cantilever", "2", "Free-tip cantilever"],
         ["Frame — non-sway", "6", "Portal frames without side-sway"],
         ["Frame — sway", "8", "Lateral loads, sway with fixed/pinned bases"],
         ["Frame — multi-bay", "4", "Two-bay symmetric frame"],
         ["Frame — L-frame", "2", "Non-rectangular frame"],
         ["Diagram checks", "12", "SFD/BMD values at known locations"]],
        col_widths=[120, 40, 280]))
    S.append(Spacer(1, 8))
    S.append(Paragraph("<b>Verification Criteria:</b>", styles["body"]))
    S += bullet_list([
        "End moments match textbook values within 0.5% tolerance",
        "Joint equilibrium: ΣM = 0 at every free-rotation node",
        "Symmetry checks for symmetric structures",
        "Boundary conditions: M = 0 at pinned/roller/free ends",
        "Shear and moment values at known points (midspan, supports)",
    ])
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Running the test suite:</b>", styles["body"]))
    S.append(code_block("python test_kassimali.py"))
    S.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # 9. DEPLOYMENT
    # ══════════════════════════════════════════════════════════════
    S.append(Paragraph("9.  Deployment", styles["h1"]))
    S.append(hr())
    S.append(Paragraph("<b>Requirements:</b>", styles["h3"]))
    S.append(code_block(
        "streamlit\n"
        "numpy\n"
        "matplotlib\n"
        "reportlab\n"
        "pandas"))
    S.append(Paragraph("<b>Installation &amp; Run:</b>", styles["h3"]))
    S.append(code_block(
        "# Clone the repository\n"
        "git clone https://github.com/hholaitan01/structsolve.git\n"
        "cd structsolve\n\n"
        "# Install dependencies\n"
        "pip install -r requirements.txt\n\n"
        "# Run the application\n"
        "streamlit run combine.py"))
    S.append(Spacer(1, 6))
    S.append(Paragraph(
        "The application opens at <font face='Courier'>http://localhost:8501</font>. "
        "No database or external service is required. All computation is performed "
        "client-side in the Python process.", styles["body"]))
    S.append(Spacer(1, 8))
    S.append(Paragraph("<b>Project Structure:</b>", styles["h3"]))
    S.append(code_block(
        "structsolve/\n"
        "├── combine.py          # Main Streamlit app\n"
        "├── homepage.py         # Landing page\n"
        "├── beam_solver.py      # Beam analysis engine\n"
        "├── frame_solver.py     # Frame analysis engine\n"
        "├── bs8110.py           # RC design (BS 8110)\n"
        "├── pdf_export.py       # PDF report generation\n"
        "├── test_kassimali.py   # Verification tests\n"
        "├── requirements.txt    # Python dependencies\n"
        "├── setup_and_run.bat   # Windows launcher\n"
        "└── README.md           # Project readme"))

    S.append(Spacer(1, 30))
    S.append(hr())
    S.append(Paragraph(
        "StructSolve Technical Documentation  •  Generated automatically  •  "
        "github.com/hholaitan01/structsolve",
        styles["footnote"]))

    # ── Build ──
    doc.build(S)
    print(f"✅ Documentation saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    build_doc()
