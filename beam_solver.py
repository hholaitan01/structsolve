import numpy as np

# -------------------------------------------------
# CORE BEAM SOLVER
# -------------------------------------------------

class BeamSolver:

    @staticmethod
    def beam_fixed_end_moments(L, load):
        """
        Returns (M_ab, M_ba) in kNm
        """
        t = load["type"]

        # -------------------------
        # POINT LOAD
        # -------------------------
        if t == "Point":
            P = load["mag"]
            a = load["pos"]
            b = L - a
            Mab = -P * a * b**2 / L**2
            Mba =  P * a**2 * b / L**2
            return Mab, Mba

        # -------------------------
        # FULL UDL
        # -------------------------
        if t == "UDL":
            w = load["mag"]
            Mab = -w * L**2 / 12
            Mba =  w * L**2 / 12
            return Mab, Mba

        # -------------------------
        # PARTIAL UDL
        # -------------------------
        if t == "UDL-P":
            w = load["mag"]
            a = load["pos"]
            b = load["end"]
            l = b - a

            R = w * l
            x = a + l / 2

            Mab = -R * (L - x) * (L - x) / L
            Mba =  R * x * x / L
            return Mab, Mba

        # -------------------------
        # PARTIAL UVL (TRIANGULAR)
        # -------------------------
        if t == "UVL-P":
            w = load["mag"]
            a = load["pos"]
            b = load["end"]
            l = b - a

            R = 0.5 * w * l

            if load["shape"] == "start_zero":
                x = a + 2 * l / 3
            else:
                x = a + l / 3

            Mab = -R * (L - x) * (L - x) / L
            Mba =  R * x * x / L
            return Mab, Mba

        return 0.0, 0.0

    # -------------------------------------------------
    # CONTINUOUS BEAM STIFFNESS SOLVER (UNCHANGED)
    # -------------------------------------------------
    @staticmethod
    def solve_continuous_beam(n, spans, support_types, span_loads,
                              sway_corrections=None, prescribed_rotations=None):
        """
        Solves a continuous beam by the slope-deflection method.
        sway_corrections : dict {span_index: Δ (m)} — settlement of near end minus far end
        prescribed_rotations : dict {joint_index: θ (rad)} — imposed rotation at fixed joints
        """
        if sway_corrections is None:
            sway_corrections = {}
        if prescribed_rotations is None:
            prescribed_rotations = {}

        A = np.zeros((n, n))
        B = np.zeros(n)
        fems = []

        for i in range(n - 1):
            L = spans[i]['L']
            tm1, tm2 = 0.0, 0.0
            for ld in span_loads[i]:
                m1, m2 = BeamSolver.beam_fixed_end_moments(L, ld)
                tm1 += m1
                tm2 += m2
            fems.append([tm1, tm2])

        for i in range(n):
            if support_types[i] in ("Fixed", "Cantilever"):
                A[i, i] = 1.0
                # Prescribed rotation overrides zero (e.g. imposed end rotation)
                B[i] = prescribed_rotations.get(i, 0.0)
            else:
                delta_left  = sway_corrections.get(i - 1, 0.0) if i > 0     else 0.0
                delta_right = sway_corrections.get(i,     0.0) if i < n - 1 else 0.0

                if i > 0:
                    L, ei = spans[i - 1]['L'], spans[i - 1]['EI']
                    A[i, i - 1] += 2 * ei / L
                    A[i, i]     += 4 * ei / L
                    B[i] -= fems[i - 1][1]
                    B[i] -= (2 * ei / L) * (-3 * delta_left / L)   # settlement correction

                if i < n - 1:
                    L, ei = spans[i]['L'], spans[i]['EI']
                    A[i, i]     += 4 * ei / L
                    A[i, i + 1] += 2 * ei / L
                    B[i] -= fems[i][0]
                    B[i] -= (2 * ei / L) * (-3 * delta_right / L)  # settlement correction

        thetas = np.linalg.solve(A, B)
        return thetas, fems

    # -------------------------------------------------
    # UPGRADED DIAGRAM SOLVER (NEW)
    # -------------------------------------------------
    @staticmethod
    def get_diagram_data(member, m_ab, m_ba, loads, n_points=200):

        L = member["L"]

        # -------------------------------------------------
        # DISCRETISATION (ALWAYS INITIALISED)
        # -------------------------------------------------
        x = np.linspace(0, L, n_points)
        v = np.zeros(n_points)
        m = np.zeros(n_points)

        # -------------------------------------------------
        # END MOMENTS (LINEAR INTERPOLATION)
        # -------------------------------------------------
        m += m_ab * (1 - x / L) + m_ba * (x / L)

        # -------------------------------------------------
        # LOAD CONTRIBUTIONS
        # -------------------------------------------------
        for ld in loads:

            # FULL UDL
            if ld["type"] == "UDL":
                w = ld["mag"]
                v += w * (L/2 - x)
                m += w * x * (L - x) / 2

            # POINT
            elif ld["type"] == "Point":
                P = ld["mag"]
                a = ld["pos"]
                for i, xi in enumerate(x):
                    if xi < a:
                        v[i] += P * (L - a) / L
                        m[i] += P * xi * (L - a) / L
                    else:
                        v[i] -= P * a / L
                        m[i] += P * a * (L - xi) / L

            # PARTIAL UDL
            elif ld["type"] == "UDL-P":
                w = ld["mag"]
                a = ld["pos"]
                b = ld["end"]
                for i, xi in enumerate(x):
                    if a <= xi <= b:
                        v[i] += w * (b - xi)
                        m[i] += w * (xi - a) * (b - xi) / 2

            # PARTIAL UVL
            elif ld["type"] == "UVL-P":
                w = ld["mag"]
                a = ld["pos"]
                b = ld["end"]
                l = b - a

                for i, xi in enumerate(x):
                    if a <= xi <= b:
                        if ld["shape"] == "start_zero":
                            wx = w * (xi - a) / l
                        else:
                            wx = w * (b - xi) / l

                        v[i] += wx * (b - xi)
                        m[i] += wx * (xi - a) * (b - xi) / 2

            
        # ---------------------------------------------
        # INTEGRATE SHEAR → MOMENT
        # ---------------------------------------------
        dx = x[1] - x[0]
        for i in range(1, len(x)):
            m[i] = m[i - 1] + v[i - 1] * dx

        # ---------------------------------------------
        # ADD END MOMENTS
        # ---------------------------------------------
        m += np.linspace(m_ab, m_ba, len(x))
        return x, v, m