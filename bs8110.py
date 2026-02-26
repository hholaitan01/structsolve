# design/bs8110.py
# BS 8110 Reinforced Concrete Beam Design
#
# (Full file – corrected bar selection logic)

# -------------------------------------------------
# REINFORCEMENT BAR DATABASE
# -------------------------------------------------
BARS = {
    "Y8": 50,
    "Y10": 79,
    "Y12": 113,
    "Y16": 201,
    "Y20": 314,
    "Y25": 491,
    "Y32": 804
}


# -------------------------------------------------
# BAR SELECTION (EXAM / OFFICE SAFE)
# -------------------------------------------------
def select_bars(As_req):
    """
    Select a practical reinforcement arrangement that
    satisfies the required steel area As_req.

    Design rules enforced:
    - Minimum 2 bars in tension
    - Maximum 6 bars in one layer
    - Prefer moderate diameters (Y16–Y25)
    - Discourage very small or very large bars
    """

    solutions = []

    for bar, area in BARS.items():
        dia = int(bar[1:])
        n = int(-(-As_req // area))   # ceiling
        As_prov = n * area

        penalty = 0

        # -------------------------------------------------
        # DETAILING & PRACTICE RULES
        # -------------------------------------------------

        # Rule 1: No single bar
        if n < 2:
            penalty += 10_000

        # Rule 2: Too many bars → congestion
        if n > 6:
            penalty += (n - 6) * 1_000

        # Rule 3: Discourage very small bars in beams
        if dia < 12:
            penalty += 2_000

        # Rule 4: Discourage very large bars
        if dia > 25:
            penalty += 1_000

        # Store solution
        solutions.append((penalty, dia, bar, n, As_prov))

    # -------------------------------------------------
    # SORTING PRIORITY
    # -------------------------------------------------
    solutions.sort(
        key=lambda x: (
            x[0],              # detailing quality
            abs(x[3] - 3),     # prefer ~3 bars
            x[4] - As_req      # minimise excess steel
        )
    )

    _, _, bar, n, As_prov = solutions[0]
    return bar, n, As_prov


# -------------------------------------------------
# FLEXURAL DESIGN (BS 8110)
# -------------------------------------------------
def beam_flexural_design(Mu, b, d, dp, fcu, fy):
    Mu *= 1e6
    z = 0.95 * d

    K = Mu / (fcu * b * d**2)
    K_lim = 0.156
    M_lim = K_lim * fcu * b * d**2

    if K <= K_lim:
        As_req = Mu / (0.87 * fy * z)
        bar, n, As_prov = select_bars(As_req)

        return {
            "type": "singly",
            "As_req": As_req,
            "As_prov": As_prov,
            "tension_bars": f"{n}Ø{bar[1:]} bottom"
        }

    else:
        As1 = M_lim / (0.87 * fy * z)
        M_ex = Mu - M_lim
        Fsc = M_ex / (d - dp)
        Asc_req = Fsc / (0.87 * fy)
        As_total = As1 + Asc_req

        bar_t, n_t, As_t = select_bars(As_total)
        bar_c, n_c, As_c = select_bars(Asc_req)

        return {
            "type": "doubly",
            "As_req": As_total,
            "As_prov": As_t,
            "Asc_req": Asc_req,
            "Asc_prov": As_c,
            "tension_bars": f"{n_t}Ø{bar_t[1:]} bottom",
            "compression_bars": f"{n_c}Ø{bar_c[1:]} top"
        }


# -------------------------------------------------
# SHEAR DESIGN (BS 8110)
# -------------------------------------------------
def beam_shear_design(Vu, b, d, fcu, As):
    Vu *= 1000
    rho = As / (b * d) * 100

    vc = 0.79 * (100 * rho * fcu) ** (1 / 3) / 1000
    vc = min(vc, 0.8)

    v = Vu / (b * d)
    Vc = vc * b * d

    if v <= vc:
        return {
            "status": "Concrete shear capacity adequate",
            "vc": vc,
            "v": v,
            "links": "Provide minimum links only"
        }

    Vus = Vu - Vc
    fyv = 460
    z = 0.9 * d

    Asv_per_s = Vus / (0.87 * fyv * z)
    link_bar = "Y10"
    Asv = 2 * BARS[link_bar]

    sv = Asv / Asv_per_s
    sv_max = min(0.75 * d, 300)
    sv = min(sv, sv_max)

    return {
        "status": "Shear reinforcement required",
        "vc": vc,
        "v": v,
        "links": f"2Ø{link_bar[1:]} @ {int(sv)} mm c/c"
    }


# -------------------------------------------------
# DETAILING CHECKS
# -------------------------------------------------
def beam_detailing_check(b, cover, agg_size, bars):
    results = []

    for dia, n in bars:
        min_clear_spacing = max(dia, agg_size + 5, 20)
        available_width = b - 2 * cover
        required_width = n * dia + (n - 1) * min_clear_spacing

        if required_width <= available_width:
            results.append(
                f"✔ {n}Ø{dia}: spacing OK "
                f"(required {required_width:.0f} mm, "
                f"available {available_width:.0f} mm)"
            )
        else:
            results.append(
                f"✖ {n}Ø{dia}: spacing FAIL "
                f"(required {required_width:.0f} mm, "
                f"available {available_width:.0f} mm)"
            )

    return results


# -------------------------------------------------
# AUTO DESIGN FROM ANALYSIS RESULTS
# -------------------------------------------------
def auto_design_from_beam_analysis(
    moments,
    shears,
    b,
    d,
    dp,
    fcu,
    fy
):
    Mu_sagging = moments["sagging"]
    Mu_hogging = abs(moments["hogging"])
    Vu = max(abs(shears["left"]), abs(shears["right"]))

    sagging_design = beam_flexural_design(
        Mu_sagging, b, d, dp, fcu, fy
    )

    hogging_design = beam_flexural_design(
        Mu_hogging, b, d, dp, fcu, fy
    )

    shear_design = beam_shear_design(
        Vu, b, d, fcu, sagging_design["As_req"]
    )

    return {
        "flexure": {
            "sagging": sagging_design,
            "hogging": hogging_design
        },
        "shear": shear_design
    }
def beam_flexural_design_T(
    Mu,
    bf, hf,
    bw, d, dp,
    fcu, fy
):
    """
    Flexural design of a reinforced concrete T-beam
    in accordance with BS 8110.

    Parameters:
        Mu  : Ultimate bending moment (kNm)
        bf  : Effective flange width (mm)
        hf  : Flange thickness (mm)
        bw  : Web width (mm)
        d   : Effective depth (mm)
        dp  : Compression steel depth d' (mm)
        fcu : Concrete cube strength (N/mm²)
        fy  : Steel yield strength (N/mm²)

    Returns:
        Dictionary describing section behaviour and reinforcement
    """

    Mu *= 1e6  # kNm → Nmm

    # -------------------------------------------------
    # STEP 1: CHECK IF NA IS IN FLANGE
    # -------------------------------------------------
    M_flange = 0.45 * fcu * bf * hf * (d - 0.5 * hf)

    # -------------------------------------------------
    # CASE 1: RECTANGULAR BEHAVIOUR (NA IN FLANGE)
    # -------------------------------------------------
    if Mu <= M_flange:
        # Treat as rectangular beam of width bf
        z = 0.95 * d
        As_req = Mu / (0.87 * fy * z)

        bar, n, As_prov = select_bars(As_req)

        return {
            "section": "rectangular (flange)",
            "type": "singly",
            "As_req": As_req,
            "As_prov": As_prov,
            "tension_bars": f"{n}Ø{bar[1:]} bottom"
        }

    # -------------------------------------------------
    # CASE 2: TRUE T-BEAM (NA IN WEB)
    # -------------------------------------------------
    # Moment resisted by flange
    M1 = M_flange

    # Remaining moment to be resisted by web
    M2 = Mu - M1

    # -------------------------------------------------
    # STEEL REQUIRED FOR FLANGE BLOCK
    # -------------------------------------------------
    Fs1 = 0.45 * fcu * bf * hf
    As1 = Fs1 / (0.87 * fy)

    # -------------------------------------------------
    # WEB CONTRIBUTION (RECTANGULAR bw)
    # -------------------------------------------------
    z2 = 0.95 * (d - 0.5 * hf)
    As2 = M2 / (0.87 * fy * z2)

    As_total = As1 + As2

    # -------------------------------------------------
    # CHECK SINGLY / DOUBLY REINFORCED
    # -------------------------------------------------
    K = Mu / (fcu * bw * d**2)
    K_lim = 0.156

    if K <= K_lim:
        bar, n, As_prov = select_bars(As_total)

        return {
            "section": "T-beam",
            "type": "singly",
            "As_req": As_total,
            "As_prov": As_prov,
            "tension_bars": f"{n}Ø{bar[1:]} bottom"
        }

    # -------------------------------------------------
    # DOUBLY REINFORCED T-BEAM
    # -------------------------------------------------
    M_lim = K_lim * fcu * bw * d**2
    M_ex = Mu - M_lim

    z = 0.95 * d
    As_lim = M_lim / (0.87 * fy * z)

    Fsc = M_ex / (d - dp)
    Asc_req = Fsc / (0.87 * fy)

    As_total = As_lim + Asc_req

    bar_t, n_t, As_t = select_bars(As_total)
    bar_c, n_c, As_c = select_bars(Asc_req)

    return {
        "section": "T-beam",
        "type": "doubly",
        "As_req": As_total,
        "As_prov": As_t,
        "Asc_req": Asc_req,
        "Asc_prov": As_c,
        "tension_bars": f"{n_t}Ø{bar_t[1:]} bottom",
        "compression_bars": f"{n_c}Ø{bar_c[1:]} top"
    }
def design_continuous_beam(
    moments,
    shears,
    section,
    materials
):
    """
    End-to-end RC design of a continuous beam using BS 8110.

    Parameters:
        moments : dict
            {
              "sagging": {
                  "AB": Mu_AB,     # kNm
                  "BC": Mu_BC
              },
              "hogging": {
                  "B": Mu_B        # kNm (negative)
              }
            }

        shears : dict
            {
              "B_left": Vu1,       # kN
              "B_right": Vu2
            }

        section : dict
            {
              "type": "T-beam" or "rectangular",
              "bf": flange width (mm),
              "hf": flange thickness (mm),
              "bw": web width (mm),
              "d": effective depth (mm),
              "dp": compression steel depth (mm)
            }

        materials : dict
            {
              "fcu": concrete strength (N/mm²),
              "fy": steel strength (N/mm²)
            }

    Returns:
        dict containing flexural and shear design results
    """

    fcu = materials["fcu"]
    fy = materials["fy"]

    bf = section.get("bf")
    hf = section.get("hf")
    bw = section["bw"]
    d = section["d"]
    dp = section["dp"]

    results = {
        "flexure": {},
        "shear": {}
    }

    # -------------------------------------------------
    # SAGGING MOMENTS (MID-SPANS)
    # -------------------------------------------------
    for span, Mu in moments["sagging"].items():

        # Sagging → compression in flange → T-beam if available
        if section["type"] == "T-beam":
            flex = beam_flexural_design_T(
                Mu=Mu,
                bf=bf,
                hf=hf,
                bw=bw,
                d=d,
                dp=dp,
                fcu=fcu,
                fy=fy
            )
        else:
            flex = beam_flexural_design(
                Mu=Mu,
                b=bw,
                d=d,
                dp=dp,
                fcu=fcu,
                fy=fy
            )

        results["flexure"][f"sagging_{span}"] = flex

    # -------------------------------------------------
    # HOGGING MOMENT (INTERNAL SUPPORT)
    # -------------------------------------------------
    Mu_hog = abs(moments["hogging"]["B"])

    # Hogging → compression in web → rectangular section
    hogging_flex = beam_flexural_design(
        Mu=Mu_hog,
        b=bw,
        d=d,
        dp=dp,
        fcu=fcu,
        fy=fy
    )

    results["flexure"]["hogging_B"] = hogging_flex

    # -------------------------------------------------
    # SHEAR DESIGN (GOVERNING)
    # -------------------------------------------------
    Vu = max(abs(shears["B_left"]), abs(shears["B_right"]))

    # Conservative practice: use sagging tension steel
    As_for_shear = max(
        flex["As_req"]
        for flex in results["flexure"].values()
    )

    shear = beam_shear_design(
        Vu=Vu,
        b=bw,
        d=d,
        fcu=fcu,
        As=As_for_shear
    )

    results["shear"] = shear

    return results
def deflection_check(L, d, beam_type, fy, As_req, As_prov):
    """
    BS 8110 span/depth deflection check

    Parameters:
        L        : Effective span (m)
        d        : Effective depth (mm)
        beam_type: "simply" or "continuous"
        fy       : Steel yield strength (N/mm²)
        As_req   : Required tension steel (mm²)
        As_prov  : Provided tension steel (mm²)

    Returns:
        dict with deflection check results
    """

    # -------------------------------------------------
    # BASIC SPAN/DEPTH LIMITS (BS 8110)
    # -------------------------------------------------
    if beam_type == "simply":
        basic_limit = 20
    else:
        basic_limit = 26

    # -------------------------------------------------
    # MODIFICATION FACTOR FOR TENSION STEEL
    # (Simplified BS 8110 approach)
    # -------------------------------------------------
    steel_ratio = As_prov / As_req if As_req > 0 else 1.0
    mf = min(2.0, max(0.55, steel_ratio))

    allowable = basic_limit * mf
    actual = (L * 1000) / d

    status = "PASS" if actual <= allowable else "FAIL"

    return {
        "L": L,
        "d": d,
        "actual": actual,
        "allowable": allowable,
        "status": status
    }

def format_design_summary(
    wu,
    moments,
    flexure,
    shear,
    deflection
):
    """
    Produces an exam / office-ready design summary.
    """

    lines = []

    lines.append("Final Design Summary\n")

    if wu is not None:
        lines.append("Ultimate load")
        lines.append(f"wu = {wu:.1f} kN/m\n")

    lines.append("Ultimate moments")
    for k, v in moments.items():
        lines.append(f"{k}: {v:.1f} kNm")
    lines.append("")

    lines.append("Flexural reinforcement")
    for zone, data in flexure.items():
        lines.append(f"{zone}: {data['tension_bars']}")
    lines.append("")

    lines.append("Shear reinforcement")
    lines.append(shear["links"])
    lines.append("")

    lines.append("Deflection check")
    actual    = deflection.get("actual",    deflection.get("actual_Ld",    0))
    allowable = deflection.get("allowable", deflection.get("allowable_Ld", 0))
    lines.append(
        f"Actual L/d = {actual:.1f}, "
        f"Allowable L/d = {allowable:.1f} → "
        f"{deflection['status']}"
    )

    return "\n".join(lines)
