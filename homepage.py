"""
StructSolve — Homepage Module
Professional, minimalist engineering homepage
"""
import streamlit as st


def homepage():
    """Render the StructSolve homepage."""

    # ── Custom CSS for light, professional homepage ────────────────────
    st.markdown("""
    <style>
    .home-hero {
        text-align: center;
        padding: 3.5rem 1rem 2.5rem;
        max-width: 780px;
        margin: 0 auto 2rem;
    }
    .home-hero h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800;
        font-size: 2.8rem !important;
        color: #E8EAF0 !important;
        letter-spacing: -1.5px;
        margin: 0 0 .6rem !important;
    }
    .home-hero .tagline {
        font-size: 1.15rem;
        color: #B0B8CC;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }
    .home-hero .methods {
        font-size: .82rem;
        color: #6B7394;
        letter-spacing: .5px;
    }
    .home-hero .methods span { color: #4ECDC4; }

    /* Feature cards */
    .feat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.2rem;
        max-width: 960px;
        margin: 0 auto 2.5rem;
    }
    .feat-card {
        background: #1A1D27;
        border: 1px solid #2A2E40;
        border-radius: 10px;
        padding: 1.4rem 1.3rem;
        transition: border-color .2s;
    }
    .feat-card:hover { border-color: #4ECDC4; }
    .feat-card .icon { font-size: 1.6rem; margin-bottom: .6rem; }
    .feat-card h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700;
        font-size: 1rem !important;
        color: #E8EAF0 !important;
        margin: 0 0 .5rem !important;
    }
    .feat-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .feat-card li {
        color: #8B92A8;
        font-size: .78rem;
        line-height: 1.8;
        padding-left: 1rem;
        position: relative;
    }
    .feat-card li::before {
        content: '–';
        position: absolute;
        left: 0;
        color: #4ECDC4;
    }

    /* Section styling */
    .home-section {
        max-width: 780px;
        margin: 0 auto 2.5rem;
        padding: 0 .5rem;
    }
    .home-section h2 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700;
        font-size: 1.25rem !important;
        color: #E8EAF0 !important;
        margin: 0 0 1rem !important;
        padding-bottom: .5rem;
        border-bottom: 2px solid #2A2E40;
    }
    .home-section p, .home-section li {
        color: #8B92A8;
        font-size: .82rem;
        line-height: 1.8;
    }
    .home-section ul {
        padding-left: 1.2rem;
        margin: .5rem 0;
    }

    /* Workflow steps */
    .workflow {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: .8rem;
    }
    .wf-step {
        background: #1A1D27;
        border: 1px solid #2A2E40;
        border-radius: 8px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .wf-step .num {
        display: inline-block;
        width: 28px; height: 28px;
        line-height: 28px;
        background: #4ECDC4;
        color: #0F1117;
        font-weight: 700;
        font-size: .8rem;
        border-radius: 50%;
        margin-bottom: .6rem;
    }
    .wf-step .label {
        color: #E8EAF0;
        font-size: .82rem;
        font-weight: 600;
    }
    .wf-step .desc {
        color: #6B7394;
        font-size: .72rem;
        margin-top: .3rem;
    }

    /* Audience grid */
    .aud-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: .8rem;
    }
    .aud-col h3 {
        font-size: .9rem !important;
        color: #E8EAF0 !important;
        margin: 0 0 .5rem !important;
        font-weight: 600;
    }
    .aud-col li {
        color: #8B92A8;
        font-size: .78rem;
        line-height: 1.8;
    }
    .aud-col.not h3 { color: #FF6B6B !important; }

    /* Footer */
    .home-footer {
        max-width: 780px;
        margin: 3rem auto 1rem;
        padding: 1.2rem 0;
        border-top: 1px solid #2A2E40;
        text-align: center;
        color: #6B7394;
        font-size: .72rem;
        line-height: 2;
    }
    .home-footer a {
        color: #4ECDC4;
        text-decoration: none;
    }
    .home-footer a:hover { text-decoration: underline; }

    /* Arrow between steps */
    .wf-arrow {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # HERO SECTION
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="home-hero">
        <h1>🏗️ StructSolve</h1>
        <div class="tagline">
            Structural Analysis &amp; RC Design — Done Right
        </div>
        <p style="color:#B0B8CC; font-size:.92rem; margin-bottom:1.5rem;">
            Analyze beams, frames, and reinforced concrete members using
            textbook-verified methods directly in your browser.
        </p>
        <div class="methods">
            <span>Slope-Deflection</span> · <span>Matrix Stiffness</span> · <span>BS 8110 RC Design</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA navigation buttons
    _cta, c1, c2, c3, _cta2 = st.columns([0.8, 1, 1, 1, 0.8])
    with c1:
        if st.button("🔩  Beam Analysis", use_container_width=True, type="primary"):
            st.session_state["_nav_target"] = "🔩 Beam Analysis"
            st.rerun()
    with c2:
        if st.button("🏛️  Frame Analysis", use_container_width=True, type="primary"):
            st.session_state["_nav_target"] = "🏛️ Frame Analysis"
            st.rerun()
    with c3:
        if st.button("🧱  RC Design", use_container_width=True, type="primary"):
            st.session_state["_nav_target"] = "🧱 RC Design"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # CORE FEATURES
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="icon">🔩</div>
            <h3>Beam Analysis</h3>
            <ul>
                <li>Simply supported and continuous beams</li>
                <li>Point loads, UDLs, partial loads</li>
                <li>Ordinate-driven SFD &amp; BMD with labelled critical points</li>
                <li>Full slope-deflection workings</li>
            </ul>
        </div>
        <div class="feat-card">
            <div class="icon">🏛️</div>
            <h3>Frame Analysis</h3>
            <ul>
                <li>Plane frames with and without sway</li>
                <li>Fixed, hinged, and roller supports</li>
                <li>Matrix stiffness method solver</li>
                <li>BMD overlay on frame geometry</li>
            </ul>
        </div>
        <div class="feat-card">
            <div class="icon">🧱</div>
            <h3>RC Design (BS 8110)</h3>
            <ul>
                <li>Flexural design (Cl. 3.4.4)</li>
                <li>Shear checks (Cl. 3.4.5)</li>
                <li>Deflection checks (Cl. 3.4.6)</li>
                <li>Singly &amp; doubly reinforced sections</li>
            </ul>
        </div>
        <div class="feat-card">
            <div class="icon">📄</div>
            <h3>Reports &amp; Export</h3>
            <ul>
                <li>Structural diagrams (SFD, BMD)</li>
                <li>Step-by-step calculation summaries</li>
                <li>Critical ordinates table</li>
                <li>Professional PDF export</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TRUST & CREDIBILITY
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="home-section">
        <h2>Built on Verified Engineering Principles</h2>
        <ul>
            <li>Classical structural analysis methods from standard textbooks</li>
            <li>Verified against 52 solved benchmark problems (Kassimali)</li>
            <li>Linear elastic theory with consistent sign conventions</li>
            <li>Transparent assumptions — no black-box calculations</li>
            <li>Full workings shown at every step for learning and checking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # WORKFLOW
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="home-section">
        <h2>How It Works</h2>
        <div class="workflow">
            <div class="wf-step">
                <div class="num">1</div>
                <div class="label">Define</div>
                <div class="desc">Geometry, supports, material properties</div>
            </div>
            <div class="wf-step">
                <div class="num">2</div>
                <div class="label">Load</div>
                <div class="desc">Point loads, UDLs, applied moments</div>
            </div>
            <div class="wf-step">
                <div class="num">3</div>
                <div class="label">Solve</div>
                <div class="desc">View diagrams, workings, ordinates</div>
            </div>
            <div class="wf-step">
                <div class="num">4</div>
                <div class="label">Export</div>
                <div class="desc">Design to BS 8110 &amp; export PDF</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # AUDIENCE
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="home-section">
        <h2>Audience</h2>
        <div class="aud-grid">
            <div class="aud-col">
                <h3>Who StructSolve Is For</h3>
                <ul>
                    <li>Civil &amp; structural engineering students</li>
                    <li>Practicing engineers (preliminary analysis)</li>
                    <li>Structural analysis educators</li>
                    <li>Anyone learning the slope-deflection method</li>
                </ul>
            </div>
            <div class="aud-col not">
                <h3>What StructSolve Is Not</h3>
                <ul>
                    <li>Nonlinear or plastic analysis software</li>
                    <li>Full building code compliance checker</li>
                    <li>Final construction approval tool</li>
                    <li>Substitute for engineering judgement</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════
    _, fc, _ = st.columns([1.5, 1, 1.5])
    with fc:
        if st.button("📖  View Documentation", use_container_width=True):
            st.session_state["_nav_target"] = "📖 Documentation"
            st.rerun()

    st.markdown("""
    <div class="home-footer">
        <a href="https://github.com/hholaitan01/structsolve" target="_blank">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/hholaitan01/structsolve/issues" target="_blank">Report an Issue</a>
        <br>
        StructSolve · BS 8110-1:1997 · Open Source
    </div>
    """, unsafe_allow_html=True)
