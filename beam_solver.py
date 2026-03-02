import numpy as np

# -------------------------------------------------
# CORE BEAM SOLVER
# -------------------------------------------------

class BeamSolver:

    @staticmethod
    def beam_fixed_end_moments(L, load):
        """
        Returns (M_ab, M_ba) for a FIXED-FIXED span.
        Sign convention: Mba > 0 for a standard UDL (hogging-positive at far end).

        Uses exact integration formulas from standard FEM tables:
          FEM_AB = -(1/L²) ∫ w(x)·x·(L-x)² dx
          FEM_BA = +(1/L²) ∫ w(x)·x²·(L-x) dx
        """
        t = load["type"]

        # POINT LOAD  — Pab²/L², Pa²b/L²
        if t == "Point":
            P = load["mag"]; a = load["pos"]; b = L - a
            return -P*a*b**2/L**2,  P*a**2*b/L**2

        # FULL UDL  — wL²/12
        if t == "UDL":
            w = load["mag"]
            return -w*L**2/12,  w*L**2/12

        # PARTIAL UDL (exact integration)
        #   ∫_a^b x(L-x)² dx = [L²x²/2 - 2Lx³/3 + x⁴/4]_a^b
        #   ∫_a^b x²(L-x) dx = [Lx³/3 - x⁴/4]_a^b
        if t == "UDL-P":
            w = load["mag"]; a = load["pos"]; b = load["end"]
            F1 = lambda x: L**2*x**2/2 - 2*L*x**3/3 + x**4/4
            F2 = lambda x: L*x**3/3 - x**4/4
            return -w/L**2 * (F1(b) - F1(a)),  w/L**2 * (F2(b) - F2(a))

        # PARTIAL UVL — TRIANGULAR (exact integration)
        if t == "UVL-P":
            w = load["mag"]; a = load["pos"]; b = load["end"]; c = b - a
            if c <= 0:
                return 0.0, 0.0
            shape = load.get("shape", "start_zero")
            if shape == "start_zero":
                # w(x) = w·(x-a)/c  from a to b  (0 at a, w at b)
                # ∫_a^b (x-a)·x·(L-x)² dx  antiderivative:
                G1 = lambda x: (L**2*x**3/3 - L*x**4/2 + x**5/5
                                - a*L**2*x**2/2 + 2*a*L*x**3/3 - a*x**4/4)
                # ∫_a^b (x-a)·x²·(L-x) dx  antiderivative:
                G2 = lambda x: (L*x**4/4 - x**5/5 - a*L*x**3/3 + a*x**4/4)
            else:
                # w(x) = w·(b-x)/c  from a to b  (w at a, 0 at b)
                # ∫_a^b (b-x)·x·(L-x)² dx  antiderivative:
                G1 = lambda x: (b*L**2*x**2/2 - 2*b*L*x**3/3 + b*x**4/4
                                - L**2*x**3/3 + L*x**4/2 - x**5/5)
                # ∫_a^b (b-x)·x²·(L-x) dx  antiderivative:
                G2 = lambda x: (b*L*x**3/3 - b*x**4/4 - L*x**4/4 + x**5/5)
            return (-w/(c*L**2) * (G1(b) - G1(a)),
                     w/(c*L**2) * (G2(b) - G2(a)))

        # APPLIED MOMENT  — Mb(2a-b)/L², Ma(2b-a)/L²
        if t == "Moment":
            M = load["mag"]; a = load["pos"]; b = L - a
            return -M*b*(2*a - b)/L**2,  M*a*(2*b - a)/L**2

        return 0.0, 0.0

    # -------------------------------------------------
    # CANTILEVER ROOT MOMENT  (statically determinate)
    # -------------------------------------------------
    @staticmethod
    def _cantilever_root_moment(L, loads):
        """
        Moment at the ROOT (near end) of a determinate cantilever whose
        TIP (far end) is free (M = V = 0 at tip).

        Uses the same sign convention as beam_fixed_end_moments:
        positive = hogging at the root end.
        For a downward gravity UDL the result is positive.
        """
        M = 0.0
        for ld in loads:
            t = ld["type"]
            if t == "UDL":
                M += ld["mag"] * L**2 / 2.0
            elif t == "Point":
                M += ld["mag"] * ld["pos"]          # pos = distance from ROOT
            elif t == "UDL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; l = b - a
                if l > 0:
                    M += w * l * (a + l/2)
            elif t == "UVL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; l = b - a
                if l > 0:
                    R  = 0.5 * w * l
                    xc = (a + 2*l/3 if ld.get("shape", "start_zero") == "start_zero"
                          else a + l/3)
                    M += R * xc
        return M   # positive = hogging at root

    # -------------------------------------------------
    # CONTINUOUS BEAM SOLVER  (slope-deflection method)
    # -------------------------------------------------
    @staticmethod
    def solve_continuous_beam(n, spans, support_types, span_loads,
                              sway_corrections: dict | None = None,
                              prescribed_rotations: dict | None = None):
        """
        Solve a continuous beam by the slope-deflection method.

        CANTILEVER HANDLING
        -------------------
        Any span whose far node has support_type == 'Free' is a statically
        determinate cantilever and is handled in three steps:

          1.  M_root is computed from loads by statics.
          2.  M_root is applied as an external moment at the junction node
              so the adjacent indeterminate spans see the correct condition.
          3.  The cantilever span is excluded from the SDE unknowns.

        After solving, the fems and thetas for cantilever spans are patched
        so that combine.py's standard back-substitution formula gives exact
        statics values:
            M_ab_cant = -M_root   (hogging at root)
            M_ba_cant =  0        (free tip)

        This fixes:
          * Pure fixed-free cantilevers  (was giving wL^2/6 instead of wL^2/2)
          * Cantilever overhangs at end of continuous beam (wrong root moment)

        Parameters
        ----------
        n                    : int  -- number of nodes (= spans + 1)
        spans                : list -- [{'L': float, 'EI': float}, ...]
        support_types        : list -- n strings: Fixed|Cantilever|Roller|Pinned|Free
        span_loads           : dict -- {span_index: [load_dicts]}
        sway_corrections     : dict -- {span_index: delta_m}  chord rotation from settlement
        prescribed_rotations : dict -- {node_index: theta_rad}

        Returns
        -------
        thetas : ndarray (n,)       -- joint rotations (patched for cantilever tips)
        fems   : list [[M_ab,M_ba]] -- fixed-end moments (patched for cantilever spans)
        """
        if sway_corrections      is None: sway_corrections      = {}
        if prescribed_rotations  is None: prescribed_rotations  = {}

        # --- Step 1: Identify cantilever spans and pre-solve by statics ----
        # cant[i] = M_root (positive = hogging at near end, same sign as Mba for UDL)
        cant = {}
        for i in range(n - 1):
            if support_types[i + 1] == "Free":
                cant[i] = BeamSolver._cantilever_root_moment(
                    spans[i]['L'], span_loads.get(i, []))

        # --- Step 2: Compute FEMs for all spans ----------------------------
        fems = []
        for i in range(n - 1):
            L  = spans[i]['L']
            m1 = m2 = 0.0
            for ld in span_loads.get(i, []):
                a, b = BeamSolver.beam_fixed_end_moments(L, ld)
                m1 += a; m2 += b
            fems.append([m1, m2])

        # --- Step 3: Assemble the SDE system --------------------------------
        A = np.zeros((n, n))
        B = np.zeros(n)

        for i in range(n):
            stype = support_types[i]

            # Fixed / Cantilever supports: theta = 0 (or prescribed value)
            if stype in ("Fixed", "Cantilever"):
                A[i, i] = 1.0
                B[i]    = prescribed_rotations.get(i, 0.0)
                continue

            # Free end of a cantilever span: placeholder row
            # (theta back-computed and then zeroed in Step 5)
            if stype == "Free":
                A[i, i] = 1.0
                continue

            # Free-rotation node (Roller / Pinned): enforce ΣM_i = 0
            delta_left  = sway_corrections.get(i - 1, 0.0) if i > 0     else 0.0
            delta_right = sway_corrections.get(i,     0.0) if i < n - 1 else 0.0

            # Left span (i-1): use modified stiffness if it is a cantilever
            if i > 0:
                L, ei = spans[i-1]['L'], spans[i-1]['EI']
                if (i - 1) not in cant:
                    # Normal span: near-end stiffness on i
                    A[i, i-1] += 2*ei/L
                    A[i, i]   += 4*ei/L
                    B[i] -= fems[i-1][1]
                    B[i] -= (2*ei/L) * (-3*delta_left/L)
                else:
                    # Left span is a cantilever (tip at i-1, root at i):
                    # i is the root → use modified stiffness 3EI/L
                    # Modified SDE: M_root = (3EI/L)*θ_root + FEM_far - FEM_tip/2 - cant
                    # (far = near-end of root = fems[i-1][1], tip = fems[i-1][0])
                    A[i, i] += 3*ei/L
                    fem_mod = fems[i-1][1] - fems[i-1][0]/2
                    B[i] -= fem_mod
                    B[i] += cant[i-1]  # static moment opposes FEM sign

            # Right span (i): use modified stiffness if it is a cantilever
            if i < n - 1:
                L, ei = spans[i]['L'], spans[i]['EI']
                if i not in cant:
                    # Normal span: near-end stiffness on i
                    A[i, i]   += 4*ei/L
                    A[i, i+1] += 2*ei/L
                    B[i] -= fems[i][0]
                    B[i] -= (2*ei/L) * (-3*delta_right/L)
                else:
                    # Right span is a cantilever (root at i, tip at i+1):
                    # i is the root → use modified stiffness 3EI/L
                    # Modified SDE: M_near = (3EI/L)*θ_root + FEM_near - FEM_far/2 - cant
                    # (near = fems[i][0], far/tip = fems[i][1])
                    A[i, i] += 3*ei/L
                    fem_mod = fems[i][0] - fems[i][1]/2
                    B[i] -= fem_mod
                    B[i] += cant[i]  # static moment opposes FEM sign

        # Guard: ensure no zero rows
        for i in range(n):
            if np.all(A[i, :] == 0.0):
                A[i, i] = 1.0

        # --- Step 4: Solve --------------------------------------------------
        thetas = np.linalg.solve(A, B)

        # --- Step 5: Patch fems and thetas for cantilever spans -------------
        #
        # combine.py back-substitutes using:
        #   m_ab = fems[i][0] + (2*EI/L)*(2*thetas[i] + thetas[i+1])
        #   m_ba = fems[i][1] + (2*EI/L)*(2*thetas[i+1] + thetas[i])
        #
        # Target: m_ab = -M_root,  m_ba = 0
        # Setting thetas[i+1] = 0 and patching fems:
        #   fems[i][0]_new = -M_root - (4*EI/L)*thetas[i]
        #   fems[i][1]_new =         - (2*EI/L)*thetas[i]
        #
        # Verification (k = EI/L, theta_root = thetas[i]):
        #   m_ab = (-M_root - 4k*tr) + 2k*(2*tr + 0) = -M_root  checkmark
        #   m_ba = (-2k*tr)           + 2k*(0   + tr) = 0        checkmark
        #
        for i, M_root in cant.items():
            L, ei  = spans[i]['L'], spans[i]['EI']
            k      = ei / L           # EI/L
            tr     = thetas[i]        # root rotation (correct from SDE)
            fems[i][0] = -M_root - 4*k*tr
            fems[i][1] =         - 2*k*tr
            thetas[i + 1] = 0.0      # tip theta placeholder

        return thetas, fems

    # -------------------------------------------------
    # DIAGRAM DATA
    # -------------------------------------------------
    @staticmethod
    def get_diagram_data(member, m_ab, m_ba, loads, n_points=200):
        """
        Compute shear force and bending moment diagrams.

        Uses reaction-based equilibrium:
          1. Compute simply-supported near-end reaction Vs from loads
          2. True reaction V0 = Vs - (m_ab + m_ba) / L
          3. March along beam:  V(x) = V0 - cumulative load
                                M(x) = m_ab + V0·x - cumulative load moment

        Sign convention:  sagging = positive moment,
                          upward shear (left of cut) = positive.
        """
        L = member["L"]
        if L < 1e-12:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        x = np.linspace(0, L, n_points)

        # ── Step 1: Simply-supported near-end reaction ──────────
        Vs = 0.0
        for ld in loads:
            t = ld["type"]
            if t == "UDL":
                Vs += ld["mag"] * L / 2.0
            elif t == "Point":
                P = ld["mag"]; a = ld["pos"]
                Vs += P * (L - a) / L
            elif t == "UDL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    Vs += w * c * (L - (a + c / 2.0)) / L
            elif t == "UVL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    R = 0.5 * w * c
                    shape = ld.get("shape", "start_zero")
                    xb = a + 2*c/3.0 if shape == "start_zero" else a + c/3.0
                    Vs += R * (L - xb) / L

        # ── Step 2: Adjust for end moments ──────────────────────
        V0 = Vs - (m_ab + m_ba) / L

        # ── Step 3: Build V(x) and M(x) by free-body equilibrium
        v = np.zeros(n_points)
        m = np.zeros(n_points)

        for i, xi in enumerate(x):
            Vi = V0
            Mi = m_ab + V0 * xi

            for ld in loads:
                t = ld["type"]

                if t == "Point":
                    P = ld["mag"]; a = ld["pos"]
                    if xi > a:
                        Vi -= P
                        Mi -= P * (xi - a)

                elif t == "UDL":
                    w = ld["mag"]
                    Vi -= w * xi
                    Mi -= w * xi**2 / 2.0

                elif t == "UDL-P":
                    w = ld["mag"]; a = ld["pos"]; b = ld["end"]
                    if xi > a:
                        c = min(xi, b) - a
                        Vi -= w * c
                        Mi -= w * c * (xi - a - c / 2.0)

                elif t == "UVL-P":
                    w = ld["mag"]; a = ld["pos"]; b = ld["end"]; cl = b - a
                    if xi > a and cl > 0:
                        d_loc = min(xi, b) - a
                        shape = ld.get("shape", "start_zero")
                        if shape == "start_zero":
                            # linearly increasing from 0 at a to w at b
                            w_at_d = w * d_loc / cl
                            Vi -= 0.5 * w_at_d * d_loc
                            Mi -= w_at_d * d_loc**2 / 6.0
                        else:
                            # linearly decreasing from w at a to 0 at b
                            w_at_d = w * (cl - d_loc) / cl
                            Vi -= (w + w_at_d) * d_loc / 2.0
                            Mi -= d_loc**2 * (2*w + w_at_d) / 6.0

            v[i] = Vi
            m[i] = Mi

        return x, v, m

    # ------------------------------------------------------------------ #
    #  CRITICAL ORDINATE HELPERS                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _eval_vm(x_pt, V0, m_ab, loads):
        """Evaluate V and M at a single point x_pt using equilibrium."""
        Vi = V0
        Mi = m_ab + V0 * x_pt
        for ld in loads:
            t = ld["type"]
            if t == "Point":
                P = ld["mag"]; a = ld["pos"]
                if x_pt > a + 1e-12:
                    Vi -= P
                    Mi -= P * (x_pt - a)
            elif t == "UDL":
                w = ld["mag"]
                Vi -= w * x_pt
                Mi -= w * x_pt**2 / 2.0
            elif t == "UDL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]
                if x_pt > a:
                    c = min(x_pt, b) - a
                    Vi -= w * c
                    Mi -= w * c * (x_pt - a - c / 2.0)
            elif t == "UVL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; cl = b - a
                if x_pt > a and cl > 0:
                    d_loc = min(x_pt, b) - a
                    shape = ld.get("shape", "start_zero")
                    if shape == "start_zero":
                        w_at_d = w * d_loc / cl
                        Vi -= 0.5 * w_at_d * d_loc
                        Mi -= w_at_d * d_loc**2 / 6.0
                    else:
                        w_at_d = w * (cl - d_loc) / cl
                        Vi -= (w + w_at_d) * d_loc / 2.0
                        Mi -= d_loc**2 * (2*w + w_at_d) / 6.0
        return Vi, Mi

    @staticmethod
    def get_critical_points(member, m_ab, m_ba, loads):
        """
        Compute critical ordinates for SFD/BMD labelling.

        Returns a list of dicts:
          [{"x": float, "V": float, "M": float, "label": str}, ...]

        Critical points include:
          - Span start/end (supports)
          - Point load positions (left & right of discontinuity)
          - Zero-shear locations (where moment is max/min)
          - Contraflexure points (M = 0 between known ordinates)
        """
        L = member["L"]
        if L < 1e-12:
            return [{"x": 0.0, "V": 0.0, "M": 0.0, "label": "start"}]

        # ── Step 1: Compute V0 (same as get_diagram_data) ──
        Vs = 0.0
        for ld in loads:
            t = ld["type"]
            if t == "UDL":
                Vs += ld["mag"] * L / 2.0
            elif t == "Point":
                P = ld["mag"]; a = ld["pos"]
                Vs += P * (L - a) / L
            elif t == "UDL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    Vs += w * c * (L - (a + c / 2.0)) / L
            elif t == "UVL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    R = 0.5 * w * c
                    shape = ld.get("shape", "start_zero")
                    xb = a + 2*c/3.0 if shape == "start_zero" else a + c/3.0
                    Vs += R * (L - xb) / L
        V0 = Vs - (m_ab + m_ba) / L

        ev = BeamSolver._eval_vm

        # ── Collect candidate x-positions ──
        xs = {0.0, L}  # always include start and end
        # Point load positions (just before and after)
        EPS = 1e-9
        for ld in loads:
            if ld["type"] == "Point":
                a = ld["pos"]
                if 0 < a < L:
                    xs.add(a - EPS)
                    xs.add(a)
                    xs.add(a + EPS)
            elif ld["type"] in ("UDL-P", "UVL-P"):
                a = ld["pos"]; b = ld["end"]
                if a > 0:   xs.add(a)
                if b < L:   xs.add(b)

        # ── Evaluate V, M at all candidate points ──
        raw_pts = sorted(xs)
        evaluated = []
        for xp in raw_pts:
            Vi, Mi = ev(xp, V0, m_ab, loads)
            evaluated.append({"x": xp, "V": Vi, "M": Mi})

        # ── Find zero-shear points (V=0) between consecutive evaluated points ──
        zero_shear = []
        for j in range(len(evaluated) - 1):
            p1, p2 = evaluated[j], evaluated[j+1]
            v1, v2 = p1["V"], p2["V"]
            x1, x2 = p1["x"], p2["x"]
            if abs(x2 - x1) < 1e-12:
                continue
            # Check for sign change or zero crossing
            if v1 * v2 < 0:
                # Find zero crossing — depends on loading type
                # Under UDL: V(x) is linear → V(x₁) + slope*(x - x₁) = 0
                # Check if there's a UDL active in this region
                w_net = 0.0
                for ld in loads:
                    if ld["type"] == "UDL":
                        w_net += ld["mag"]
                    elif ld["type"] == "UDL-P":
                        w = ld["mag"]; a = ld["pos"]; b = ld["end"]
                        if a <= x1 and x2 <= b:
                            w_net += w
                    elif ld["type"] == "UVL-P":
                        # Approximate with average
                        w = ld["mag"]; a = ld["pos"]; b = ld["end"]; cl = b - a
                        if a <= x1 and x2 <= b and cl > 0:
                            shape = ld.get("shape", "start_zero")
                            xm = (x1 + x2) / 2
                            d = xm - a
                            if shape == "start_zero":
                                w_net += w * d / cl
                            else:
                                w_net += w * (cl - d) / cl

                if abs(w_net) > 1e-12:
                    # Linear V: V(x) = v1 - w_net*(x - x1)
                    # V=0 → x = x1 + v1/w_net
                    x_zero = x1 + v1 / w_net
                    if x1 < x_zero < x2:
                        Vz, Mz = ev(x_zero, V0, m_ab, loads)
                        zero_shear.append({"x": x_zero, "V": Vz, "M": Mz})
                else:
                    # Linear interpolation
                    frac = abs(v1) / (abs(v1) + abs(v2))
                    x_zero = x1 + frac * (x2 - x1)
                    Vz, Mz = ev(x_zero, V0, m_ab, loads)
                    zero_shear.append({"x": x_zero, "V": Vz, "M": Mz})

        # ── Find contraflexure points (M=0) ──
        all_for_contra = sorted(evaluated + zero_shear, key=lambda p: p["x"])
        contraflexure = []
        for j in range(len(all_for_contra) - 1):
            p1, p2 = all_for_contra[j], all_for_contra[j+1]
            m1_v, m2_v = p1["M"], p2["M"]
            x1, x2 = p1["x"], p2["x"]
            if abs(x2 - x1) < 1e-12:
                continue
            if m1_v * m2_v < 0:
                # Bisection to find M=0 (works for parabolic and linear)
                xa, xb = x1, x2
                for _ in range(50):
                    xm = (xa + xb) / 2.0
                    _, Mm = ev(xm, V0, m_ab, loads)
                    if abs(Mm) < 1e-10:
                        break
                    if Mm * m1_v > 0:
                        xa = xm
                    else:
                        xb = xm
                xm = (xa + xb) / 2.0
                Vcf, Mcf = ev(xm, V0, m_ab, loads)
                contraflexure.append({"x": xm, "V": Vcf, "M": Mcf})

        # ── Merge and label ──
        result = []
        for p in evaluated:
            lbl = ""
            if abs(p["x"]) < 1e-9:
                lbl = "start"
            elif abs(p["x"] - L) < 1e-9:
                lbl = "end"
            else:
                # Check if it's a point load location
                for ld in loads:
                    if ld["type"] == "Point" and abs(p["x"] - ld["pos"]) < 1e-6:
                        lbl = "point_load"
                        break
                if not lbl:
                    lbl = "load_boundary"
            result.append({**p, "label": lbl})

        for p in zero_shear:
            result.append({**p, "label": "V=0 (M_max)"})
        for p in contraflexure:
            result.append({**p, "label": "M=0 (contra)"})

        # Sort and deduplicate (merge points within tolerance)
        result.sort(key=lambda p: p["x"])
        merged = []
        for p in result:
            if merged and abs(p["x"] - merged[-1]["x"]) < 1e-6:
                # Keep the one with a more informative label
                if "V=0" in p["label"] or "M=0" in p["label"]:
                    merged[-1] = p
                continue
            merged.append(p)

        return merged
