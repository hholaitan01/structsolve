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
        """
        t = load["type"]

        # POINT LOAD
        if t == "Point":
            P = load["mag"]; a = load["pos"]; b = L - a
            return -P*a*b**2/L**2,  P*a**2*b/L**2

        # FULL UDL
        if t == "UDL":
            w = load["mag"]
            return -w*L**2/12,  w*L**2/12

        # PARTIAL UDL
        if t == "UDL-P":
            w = load["mag"]; a = load["pos"]; b = load["end"]; l = b - a
            R = w*l; x = a + l/2
            return -R*(L-x)**2*x/L**2,  R*x**2*(L-x)/L**2

        # PARTIAL UVL (TRIANGULAR)
        if t == "UVL-P":
            w = load["mag"]; a = load["pos"]; b = load["end"]; l = b - a
            R = 0.5*w*l
            x = a + 2*l/3 if load["shape"] == "start_zero" else a + l/3
            return -R*(L-x)**2*x/L**2,  R*x**2*(L-x)/L**2

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
                              sway_corrections=None, prescribed_rotations=None):
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

            # Left span (i-1): skip if it is a cantilever span
            if i > 0 and (i - 1) not in cant:
                L, ei = spans[i-1]['L'], spans[i-1]['EI']
                A[i, i-1] += 2*ei/L
                A[i, i]   += 4*ei/L
                B[i] -= fems[i-1][1]
                B[i] -= (2*ei/L) * (-3*delta_left/L)   # settlement correction

            # Right span (i): skip if it is a cantilever span
            if i < n - 1 and i not in cant:
                L, ei = spans[i]['L'], spans[i]['EI']
                A[i, i]   += 4*ei/L
                A[i, i+1] += 2*ei/L
                B[i] -= fems[i][0]
                B[i] -= (2*ei/L) * (-3*delta_right/L)  # settlement correction

            # External moment from an adjacent cantilever overhang.
            # Span i (to the right of node i) is a cantilever:
            #   Root moment M_root acts on node i.
            #   Equilibrium: ΣM_SDE + M_root = 0  =>  B[i] -= M_root
            if i in cant:
                B[i] -= cant[i]

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

        L = member["L"]

        x = np.linspace(0, L, n_points)
        v = np.zeros(n_points)
        m = np.zeros(n_points)

        # End moments (linear interpolation base)
        m += m_ab * (1 - x/L) + m_ba * (x/L)

        for ld in loads:

            # FULL UDL
            if ld["type"] == "UDL":
                w = ld["mag"]
                v += w * (L/2 - x)
                m += w * x * (L - x) / 2

            # POINT
            elif ld["type"] == "Point":
                P = ld["mag"]; a = ld["pos"]
                for i, xi in enumerate(x):
                    if xi < a:
                        v[i] += P * (L - a) / L
                        m[i] += P * xi * (L - a) / L
                    else:
                        v[i] -= P * a / L
                        m[i] += P * a * (L - xi) / L

            # PARTIAL UDL
            elif ld["type"] == "UDL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]
                for i, xi in enumerate(x):
                    if a <= xi <= b:
                        v[i] += w * (b - xi)
                        m[i] += w * (xi - a) * (b - xi) / 2

            # PARTIAL UVL
            elif ld["type"] == "UVL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; l = b - a
                for i, xi in enumerate(x):
                    if a <= xi <= b:
                        wx = (w*(xi-a)/l if ld["shape"] == "start_zero"
                              else w*(b-xi)/l)
                        v[i] += wx * (b - xi)
                        m[i] += wx * (xi - a) * (b - xi) / 2

        # Integrate shear -> moment
        dx = x[1] - x[0]
        for i in range(1, len(x)):
            m[i] = m[i-1] + v[i-1] * dx

        # Add end moments
        m += np.linspace(m_ab, m_ba, len(x))
        return x, v, m
