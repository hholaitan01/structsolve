"""
StructSolve — Technical Documentation Page
Full technical reference rendered as a Streamlit page.
"""
import streamlit as st


def docs_page():
    """Render the technical documentation as a Streamlit page."""

    st.markdown("""
    <style>
    .doc-header {
        text-align: center; padding: 2rem 0 1.5rem;
        border-bottom: 2px solid #2A2E40; margin-bottom: 2rem;
    }
    .doc-header h1 {
        font-family: 'Syne', sans-serif !important; font-weight: 800;
        font-size: 2rem !important; color: #E8EAF0 !important; margin: 0 !important;
    }
    .doc-header p { color: #8B92A8; font-size: .85rem; margin-top: .4rem; }
    .doc-section { max-width: 820px; margin: 0 auto 2rem; }
    .doc-section h2 {
        font-family: 'Syne', sans-serif !important; font-weight: 700;
        font-size: 1.3rem !important; color: #4ECDC4 !important;
        margin: 2rem 0 .8rem !important; padding-bottom: .4rem;
        border-bottom: 1px solid #2A2E40;
    }
    .doc-section h3 {
        font-size: 1.05rem !important; color: #E8EAF0 !important;
        margin: 1.2rem 0 .5rem !important; font-weight: 600;
    }
    .doc-section p, .doc-section li {
        color: #8B92A8; font-size: .85rem; line-height: 1.75;
    }
    .doc-section ul { padding-left: 1.2rem; }
    .doc-section code {
        background: #22263A; padding: 1px 5px; border-radius: 3px;
        font-size: .82rem; color: #4ECDC4;
    }
    .doc-section pre {
        background: #1A1D27; border: 1px solid #2A2E40; border-radius: 8px;
        padding: 1rem; font-size: .8rem; color: #B0B8CC; overflow-x: auto;
        line-height: 1.6;
    }
    .doc-toc {
        background: #1A1D27; border: 1px solid #2A2E40; border-radius: 10px;
        padding: 1.2rem 1.5rem; max-width: 820px; margin: 0 auto 2rem;
    }
    .doc-toc h3 { color: #E8EAF0 !important; margin: 0 0 .6rem !important; font-size: 1rem !important; }
    .doc-toc ol { padding-left: 1.2rem; margin: 0; }
    .doc-toc li { color: #8B92A8; font-size: .82rem; line-height: 2; }
    .doc-toc a { color: #4ECDC4; text-decoration: none; }
    .doc-table { width: 100%; border-collapse: collapse; margin: .8rem 0 1.2rem; font-size: .82rem; }
    .doc-table th {
        background: #2E7D6E; color: white; text-align: left;
        padding: 6px 10px; font-weight: 600;
    }
    .doc-table td {
        padding: 6px 10px; color: #8B92A8; border-bottom: 1px solid #2A2E40;
    }
    .doc-table tr:nth-child(even) td { background: #1A1D27; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="doc-header">
        <h1>📖 Technical Documentation</h1>
        <p>StructSolve — Structural Analysis &amp; RC Design Suite</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Table of Contents ──
    st.markdown("""
    <div class="doc-toc">
        <h3>Contents</h3>
        <ol>
            <li>Overview</li>
            <li>System Architecture</li>
            <li>Module Reference</li>
            <li>Analysis Methods</li>
            <li>RC Design Methods (BS 8110)</li>
            <li>Sign Conventions</li>
            <li>Input / Output Reference</li>
            <li>Verification &amp; Testing</li>
            <li>Deployment</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # All content inside a max-width div
    def section(html):
        st.markdown(f'<div class="doc-section">{html}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # 1. OVERVIEW
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>1. Overview</h2>
    <p>StructSolve is an open-source, browser-based structural engineering application
    built with Python and Streamlit. It provides three core modules:</p>
    <ul>
        <li><strong>Beam Analysis</strong> — Continuous beams using the slope-deflection method</li>
        <li><strong>Frame Analysis</strong> — Plane frames (sway and non-sway) using the matrix stiffness method</li>
        <li><strong>RC Design</strong> — Reinforced concrete design to BS 8110-1:1997</li>
    </ul>
    <h3>Key Capabilities</h3>
    <ul>
        <li>Ordinate-driven SFD and BMD with labelled critical points</li>
        <li>Full step-by-step workings (FEMs, SDEs, equilibrium)</li>
        <li>Cantilever spans with modified stiffness handling</li>
        <li>PDF report export with diagrams and calculation summaries</li>
        <li>Verified against 52 benchmark problems (Kassimali textbook)</li>
    </ul>
    <h3>Technology Stack</h3>
    <table class="doc-table">
        <tr><th>Component</th><th>Technology</th></tr>
        <tr><td>Language</td><td>Python 3.11+</td></tr>
        <tr><td>Web Framework</td><td>Streamlit</td></tr>
        <tr><td>Numerical</td><td>NumPy</td></tr>
        <tr><td>Plotting</td><td>Matplotlib</td></tr>
        <tr><td>PDF Generation</td><td>ReportLab</td></tr>
        <tr><td>Data Tables</td><td>Pandas</td></tr>
    </table>
    """)

    # ══════════════════════════════════════════════════════════════
    # 2. ARCHITECTURE
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>2. System Architecture</h2>
    <p>The application follows a modular architecture with clear separation between
    solver engines, UI logic, and export functionality.</p>
    <table class="doc-table">
        <tr><th>File</th><th>Role</th><th>Lines</th></tr>
        <tr><td>beam_solver.py</td><td>Beam analysis engine (SDE method, critical points)</td><td>~540</td></tr>
        <tr><td>frame_solver.py</td><td>Frame analysis engine (matrix stiffness)</td><td>~900</td></tr>
        <tr><td>bs8110.py</td><td>RC design calculations to BS 8110</td><td>~400</td></tr>
        <tr><td>combine.py</td><td>Streamlit UI, plotting, workings display</td><td>~1850</td></tr>
        <tr><td>pdf_export.py</td><td>PDF report generation</td><td>~900</td></tr>
        <tr><td>homepage.py</td><td>Landing page module</td><td>~360</td></tr>
        <tr><td>test_kassimali.py</td><td>Verification test suite (52 checks)</td><td>~750</td></tr>
    </table>
    <h3>Data Flow</h3>
    <pre>User Input (Streamlit UI)
    ↓
BeamSolver.solve_continuous_beam()  or  FrameSolver.solve()
    ↓  returns: thetas, fems
Back-substitution → end moments (M_ab, M_ba)
    ↓
BeamSolver.get_diagram_data()      → x, V, M arrays
BeamSolver.get_critical_points()   → labelled ordinates
    ↓
plot_sfd_bmd() / beam_workings()   → display in Streamlit
    ↓
pdf_export.export_*_pdf()          → downloadable PDF</pre>
    """)

    # ══════════════════════════════════════════════════════════════
    # 3. MODULE REFERENCE
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>3. Module Reference</h2>

    <h3>3.1 beam_solver.py — Beam Analysis Engine</h3>
    <p>Core solver class <code>BeamSolver</code> implementing the slope-deflection method
    for continuous beams.</p>
    <table class="doc-table">
        <tr><th>Method</th><th>Description</th></tr>
        <tr><td>solve_continuous_beam()</td><td>Main solver. Assembles SDE matrix, handles cantilevers with modified stiffness (3EI/L), solves for joint rotations.</td></tr>
        <tr><td>beam_fixed_end_moments()</td><td>Computes FEMs for Point, UDL, UDL-P, UVL-P loads on a fixed-fixed beam.</td></tr>
        <tr><td>get_diagram_data()</td><td>Generates V(x) and M(x) arrays using equilibrium-based section cuts at 200 points per span.</td></tr>
        <tr><td>get_critical_points()</td><td>Analytically computes all critical ordinates: supports, point loads, zero-shear (M max/min), contraflexure (M=0).</td></tr>
        <tr><td>_eval_vm()</td><td>Evaluates V and M at any single point using equilibrium.</td></tr>
        <tr><td>_cantilever_root_moment()</td><td>Static moment at cantilever root by statics.</td></tr>
    </table>

    <h3>Parameters for solve_continuous_beam()</h3>
    <table class="doc-table">
        <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
        <tr><td>n</td><td>int</td><td>Number of joints (nodes) = spans + 1</td></tr>
        <tr><td>spans</td><td>list[dict]</td><td>{'L': float, 'EI': float} for each span</td></tr>
        <tr><td>support_types</td><td>list[str]</td><td>Fixed | Roller | Pinned | Free (length n)</td></tr>
        <tr><td>span_loads</td><td>dict</td><td>{span_idx: [load_dicts]} with type, mag, pos, end</td></tr>
        <tr><td>sway_corrections</td><td>dict</td><td>{span_idx: delta} chord rotation from settlement</td></tr>
        <tr><td>prescribed_rotations</td><td>dict</td><td>{node_idx: theta_rad}</td></tr>
    </table>
    <p><strong>Returns:</strong></p>
    <ul>
        <li><code>thetas</code> — ndarray of joint rotations (patched for cantilevers)</li>
        <li><code>fems</code> — list of [M_ab, M_ba] (patched for cantilever back-substitution)</li>
    </ul>

    <h3>3.2 frame_solver.py — Frame Analysis Engine</h3>
    <p>Class <code>FrameSolver</code> implementing the matrix stiffness (direct stiffness)
    method for plane frames with optional sway.</p>
    <ul>
        <li>Arbitrary plane frame geometry with inclined members</li>
        <li>Fixed, pinned, and roller supports</li>
        <li>Sway analysis with story-level side-sway DOFs</li>
        <li>Point loads, UDLs on any member</li>
        <li>End moment, shear, and axial force output</li>
    </ul>

    <h3>3.3 bs8110.py — RC Design (BS 8110-1:1997)</h3>
    <table class="doc-table">
        <tr><th>Function</th><th>Clause</th><th>Description</th></tr>
        <tr><td>beam_flexural_design()</td><td>3.4.4</td><td>Singly/doubly reinforced rectangular beam design</td></tr>
        <tr><td>beam_flexural_design_T()</td><td>3.4.4</td><td>Flanged (T/L) beam flexural design</td></tr>
        <tr><td>beam_shear_design()</td><td>3.4.5</td><td>Shear reinforcement design with v_c tables</td></tr>
        <tr><td>deflection_check()</td><td>3.4.6</td><td>Span/depth check with modification factors</td></tr>
        <tr><td>design_continuous_beam()</td><td>—</td><td>Auto-design all spans from beam analysis output</td></tr>
    </table>

    <h3>3.4 combine.py — Application Interface</h3>
    <table class="doc-table">
        <tr><th>Function</th><th>Description</th></tr>
        <tr><td>beam_page()</td><td>Full beam analysis UI (input, solve, diagrams, workings, design)</td></tr>
        <tr><td>frame_page()</td><td>Frame analysis UI with geometry builder and BMD overlay</td></tr>
        <tr><td>design_page()</td><td>Standalone RC design page</td></tr>
        <tr><td>plot_sfd_bmd()</td><td>Ordinate-driven SFD/BMD plotting with critical point labels</td></tr>
        <tr><td>draw_beam_system()</td><td>Visual beam diagram with supports and loads</td></tr>
        <tr><td>beam_workings()</td><td>Step-by-step SDE workings display</td></tr>
    </table>

    <h3>3.5 pdf_export.py — PDF Report Generation</h3>
    <table class="doc-table">
        <tr><th>Function</th><th>Description</th></tr>
        <tr><td>export_beam_analysis_pdf()</td><td>Full beam report with diagrams, ordinates, support actions</td></tr>
        <tr><td>export_beam_design_pdf()</td><td>RC design report linked to beam analysis</td></tr>
        <tr><td>export_frame_pdf()</td><td>Frame analysis report with member forces</td></tr>
        <tr><td>export_rc_design_pdf()</td><td>Standalone RC design report</td></tr>
        <tr><td>_sfd_bmd_fig()</td><td>Matplotlib figure builder with ordinate annotations</td></tr>
    </table>

    <h3>3.6 homepage.py — Landing Page</h3>
    <p>Professional homepage with feature cards, workflow steps, audience sections,
    and navigation buttons that route to each module.</p>
    """)

    # ══════════════════════════════════════════════════════════════
    # 4. ANALYSIS METHODS
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>4. Analysis Methods</h2>

    <h3>4.1 Slope-Deflection Method</h3>
    <p>For each span, the end moments are expressed as:</p>
    <pre>M_ij = FEM_ij + (2EI/L)(2θ_i + θ_j − 3Δ/L)
M_ji = FEM_ji + (2EI/L)(2θ_j + θ_i − 3Δ/L)</pre>
    <p>where:</p>
    <ul>
        <li><code>FEM_ij, FEM_ji</code> = fixed-end moments from applied loads</li>
        <li><code>θ_i, θ_j</code> = joint rotations (unknowns)</li>
        <li><code>Δ</code> = relative settlement (chord rotation correction)</li>
        <li><code>E, I, L</code> = material and geometric properties</li>
    </ul>
    <p>Joint equilibrium (ΣM = 0 at each free-rotation node) yields a system of
    linear equations solved by NumPy's <code>linalg.solve()</code>.</p>

    <h3>4.2 Cantilever Span Handling</h3>
    <p>For spans with a free end, the solver uses <strong>modified stiffness</strong>
    at the root joint:</p>
    <pre>Standard stiffness:  4EI/L  (far end fixed)
Modified stiffness:  3EI/L  (far end free)

Modified SDE:
  M_near = (3EI/L)·θ_root + FEM_near − FEM_far/2 − M_static

where M_static = cantilever root moment from applied loads (by statics)</pre>
    <p>The far-end rotation θ_D is condensed out analytically using M_DC = 0,
    eliminating one unknown per cantilever span from the system.</p>

    <h3>4.3 Critical Ordinate Computation</h3>
    <p>The <code>get_critical_points()</code> method finds ordinate values at all
    locations needed for exam-quality SFD/BMD diagrams:</p>
    <table class="doc-table">
        <tr><th>Critical Point</th><th>How Found</th><th>Label</th></tr>
        <tr><td>Span start/end</td><td>x = 0, x = L</td><td>Support</td></tr>
        <tr><td>Point load locations</td><td>x = a (load position)</td><td>Point load</td></tr>
        <tr><td>Partial load boundaries</td><td>x = a, x = b</td><td>Load boundary</td></tr>
        <tr><td>Zero shear (V = 0)</td><td>Analytical: x = x₁ + V₁/w_net</td><td>V=0 (M max) ★</td></tr>
        <tr><td>Contraflexure (M = 0)</td><td>Bisection (50 iterations)</td><td>M=0 ○</td></tr>
    </table>
    """)

    # ══════════════════════════════════════════════════════════════
    # 5. RC DESIGN
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>5. RC Design Methods (BS 8110)</h2>
    <p>All RC design follows BS 8110-1:1997 with the following assumptions:</p>
    <ul>
        <li>Rectangular stress block (clause 3.4.4.4)</li>
        <li>Maximum neutral axis depth: x/d ≤ 0.5 (moment redistribution ≤ 10%)</li>
        <li>Concrete: C25, C30, C32, or C40 (f_cu in N/mm²)</li>
        <li>Steel: f_y = 460 N/mm² (high-yield) or 250 N/mm² (mild)</li>
    </ul>

    <h3>Flexural Design Procedure</h3>
    <pre>1. K = M / (b·d²·f_cu)
2. If K ≤ K' = 0.156 → singly reinforced
     z = d[0.5 + √(0.25 − K/0.9)]  (capped at 0.95d)
     A_s = M / (0.87·f_y·z)
3. If K > K' → doubly reinforced
     A_s' = (K − K')·f_cu·b·d² / (0.87·f_y·(d − d'))
     A_s  = K'·f_cu·b·d² / (0.87·f_y·z') + A_s'</pre>

    <h3>Shear Design (Cl. 3.4.5)</h3>
    <pre>v = V / (b·d)
v_c = 0.79·(100·A_s/(b·d))^(1/3)·(400/d)^(1/4) / γ_m
If v ≤ v_c/2:       no shear links required
If v_c/2 &lt; v ≤ v_c: minimum links  A_sv/s = 0.4·b/(0.87·f_yv)
If v &gt; v_c:          design links   A_sv/s = b·(v − v_c)/(0.87·f_yv)</pre>

    <h3>Deflection Check (Cl. 3.4.6)</h3>
    <pre>Basic span/depth ratio (Table 3.9):
  Cantilever: 7    Simply supported: 20    Continuous: 26

Tension steel modification factor (Table 3.10):
  f_s = 2·f_y·A_s,req / (3·A_s,prov) × β_b
  MF_t = 0.55 + (477 − f_s) / (120·(0.9 + M/(b·d²)))

Allowable L/d = basic ratio × MF_t × MF_c
Check: actual L/d ≤ allowable L/d</pre>
    """)

    # ══════════════════════════════════════════════════════════════
    # 6. SIGN CONVENTIONS
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>6. Sign Conventions</h2>
    <table class="doc-table">
        <tr><th>Quantity</th><th>Positive Direction</th><th>Notes</th></tr>
        <tr><td>End moment M_ij</td><td>Clockwise on near end (i)</td><td>Standard SDE convention</td></tr>
        <tr><td>FEM (UDL)</td><td>Hogging = negative at near end</td><td>FEM_AB = −wL²/12</td></tr>
        <tr><td>Shear force V</td><td>Upward on left face</td><td>V at x=0 = left reaction</td></tr>
        <tr><td>Bending moment M</td><td>Sagging positive (solver)</td><td>Negated for display (hogging ↑)</td></tr>
        <tr><td>Joint rotation θ</td><td>Clockwise positive</td><td>Consistent with SDE</td></tr>
        <tr><td>Loads</td><td>Downward positive</td><td>Gravity convention</td></tr>
    </table>
    <p><strong>Display Convention:</strong> In the BMD plot, the solver's sagging-positive
    moments are negated so that hogging plots upward and sagging plots downward
    (inverted y-axis). This matches the standard engineering drawing convention.</p>
    """)

    # ══════════════════════════════════════════════════════════════
    # 7. INPUT/OUTPUT
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>7. Input / Output Reference</h2>

    <h3>Load Types</h3>
    <table class="doc-table">
        <tr><th>Type</th><th>Parameters</th><th>Description</th></tr>
        <tr><td>Point</td><td>mag, pos</td><td>Concentrated load at distance 'pos' from left end</td></tr>
        <tr><td>UDL</td><td>mag</td><td>Uniform load over entire span</td></tr>
        <tr><td>UDL-P</td><td>mag, pos, end</td><td>Partial UDL from 'pos' to 'end'</td></tr>
        <tr><td>UVL-P</td><td>mag, pos, end, shape</td><td>Triangular load; shape = start_zero or end_zero</td></tr>
        <tr><td>Moment</td><td>mag, pos</td><td>Applied moment at distance 'pos'</td></tr>
    </table>

    <h3>Support Types</h3>
    <table class="doc-table">
        <tr><th>Type</th><th>θ</th><th>Δ</th><th>M</th><th>Description</th></tr>
        <tr><td>Fixed</td><td>= 0</td><td>= 0</td><td>≠ 0</td><td>Fully restrained</td></tr>
        <tr><td>Roller</td><td>≠ 0</td><td>= 0</td><td>= 0</td><td>Free rotation, ΣM = 0</td></tr>
        <tr><td>Pinned</td><td>≠ 0</td><td>= 0</td><td>= 0</td><td>Same as Roller for beams</td></tr>
        <tr><td>Free</td><td>≠ 0</td><td>≠ 0</td><td>= 0</td><td>Cantilever tip (M = V = 0)</td></tr>
    </table>

    <h3>Units</h3>
    <table class="doc-table">
        <tr><th>Quantity</th><th>Unit</th></tr>
        <tr><td>Length (geometry)</td><td>m</td></tr>
        <tr><td>Force / load</td><td>kN, kN/m</td></tr>
        <tr><td>Moment</td><td>kNm</td></tr>
        <tr><td>Stress</td><td>N/mm²</td></tr>
        <tr><td>Section dimensions</td><td>mm</td></tr>
        <tr><td>Reinforcement area</td><td>mm²</td></tr>
    </table>
    """)

    # ══════════════════════════════════════════════════════════════
    # 8. VERIFICATION
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>8. Verification &amp; Testing</h2>
    <p>The solver is verified against problems from Aslam Kassimali's
    <em>Structural Analysis</em> textbook using an automated test suite
    (<code>test_kassimali.py</code>).</p>

    <h3>Test Summary: 52/52 checks passed ✓</h3>
    <table class="doc-table">
        <tr><th>Category</th><th>#</th><th>Description</th></tr>
        <tr><td>Beam — basic</td><td>6</td><td>Fixed-fixed beams with UDL and point loads</td></tr>
        <tr><td>Beam — propped cantilever</td><td>4</td><td>Fixed-Free with various loads</td></tr>
        <tr><td>Beam — multi-span</td><td>8</td><td>Two/three-span with mixed supports and EI</td></tr>
        <tr><td>Beam — pure cantilever</td><td>2</td><td>Free-tip cantilever</td></tr>
        <tr><td>Frame — non-sway</td><td>6</td><td>Portal frames without side-sway</td></tr>
        <tr><td>Frame — sway</td><td>8</td><td>Lateral loads, sway with fixed/pinned bases</td></tr>
        <tr><td>Frame — multi-bay</td><td>4</td><td>Two-bay symmetric frame</td></tr>
        <tr><td>Frame — L-frame</td><td>2</td><td>Non-rectangular frame</td></tr>
        <tr><td>Diagram checks</td><td>12</td><td>SFD/BMD values at known locations</td></tr>
    </table>

    <h3>Verification Criteria</h3>
    <ul>
        <li>End moments match textbook values within 0.5% tolerance</li>
        <li>Joint equilibrium: ΣM = 0 at every free-rotation node</li>
        <li>Symmetry checks for symmetric structures</li>
        <li>Boundary conditions: M = 0 at pinned/roller/free ends</li>
        <li>Shear and moment values at known points (midspan, supports)</li>
    </ul>

    <h3>Running the Tests</h3>
    <pre>python test_kassimali.py</pre>
    """)

    # ══════════════════════════════════════════════════════════════
    # 9. DEPLOYMENT
    # ══════════════════════════════════════════════════════════════
    section("""
    <h2>9. Deployment</h2>

    <h3>Installation &amp; Run</h3>
    <pre># Clone the repository
git clone https://github.com/hholaitan01/structsolve.git
cd structsolve

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run combine.py</pre>
    <p>The application opens at <code>http://localhost:8501</code>. No database
    or external service is required.</p>

    <h3>Requirements</h3>
    <pre>streamlit
numpy
matplotlib
reportlab
pandas</pre>

    <h3>Project Structure</h3>
    <pre>structsolve/
├── combine.py          # Main Streamlit app
├── homepage.py         # Landing page
├── docs_page.py        # Technical documentation
├── beam_solver.py      # Beam analysis engine
├── frame_solver.py     # Frame analysis engine
├── bs8110.py           # RC design (BS 8110)
├── pdf_export.py       # PDF report generation
├── generate_docs.py    # PDF documentation generator
├── test_kassimali.py   # Verification tests
├── requirements.txt    # Python dependencies
├── setup_and_run.bat   # Windows launcher
└── README.md           # Project readme</pre>
    """)
