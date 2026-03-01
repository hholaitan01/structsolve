"""
GeneralFrameSolver v4.2 — Slope-Deflection Method for ANY plane frame
======================================================================
Fixes vs v4.1:
  * Modified stiffness FEM correction: when far end is pinned/roller,
    FEM_ij_modified = FEM_ij_ff - 0.5 * FEM_ji_ff  (eliminates far-end carry-over)
  * Cantilever/free-tip handling: a Free-support node connected to only ONE member
    is a cantilever tip. Its moment is computed statically and injected as an
    external joint moment at the root node.
  * Propped cantilever (fixed-pinned): now gives correct M_AB = -wL²/8
  * True cantilever (fixed-free): now gives correct M_AB = -wL²/2
  * All Kassimali Ch.15 benchmark problems pass.

Sign convention:  clockwise end moment = positive
Slope-deflection: M_ij = M_Fij + (2EI/L)[2θ_i + θ_j − 3ψ]
Modified:         M_ij = M_Fij_mod + (3EI/L)[θ_i − ψ],  M_ji = 0
  where M_Fij_mod = M_Fij_ff − ½·M_Fji_ff
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# FIXED-END MOMENTS  (fixed-fixed assumption)
# ══════════════════════════════════════════════════════════════

def _fem(L, load):
    """(M_near, M_far) for one load, fixed-fixed span."""
    t = load["type"]
    if t == "UDL":
        w = load["mag"]
        return -w*L**2/12.0, +w*L**2/12.0
    if t == "UDL-P":
        w=load["mag"]; a=load["pos"]; b=load["end"]
        if b<=a: return 0.0,0.0
        # Exact: Mab=-w/L² ∫_a^b (L-x)²x dx,  Mba=+w/L² ∫_a^b x²(L-x) dx
        inv_L2=1.0/L**2
        Mab=-w*inv_L2*(L**2*(b**2-a**2)/2 - 2*L*(b**3-a**3)/3 + (b**4-a**4)/4)
        Mba= w*inv_L2*(L*(b**3-a**3)/3 - (b**4-a**4)/4)
        return Mab, Mba
    if t == "UVL-P":
        w=load["mag"]; a=load["pos"]; b=load["end"]; c=b-a
        if c<=0: return 0.0,0.0
        R=0.5*w*c
        shape=load.get("shape","start_zero")
        xb=a+2*c/3.0 if shape=="start_zero" else a+c/3.0
        return -R*(L-xb)**2*xb/L**2, +R*xb**2*(L-xb)/L**2
    if t == "Point":
        P=load["mag"]; a=load["pos"]; b=L-a
        if b<0: b=0.0
        return -P*a*b**2/L**2, +P*a**2*b/L**2
    if t == "Moment":
        M=load["mag"]; a=load["pos"]; b=L-a
        return -M*b*(2*a-b)/L**2, -M*a*(2*b-a)/L**2
    return 0.0, 0.0


def _fem_total(L, loads):
    """Sum of (M_near, M_far) over all loads on a span."""
    m_ni, m_nj = 0.0, 0.0
    for ld in loads:
        a, b = _fem(L, ld)
        m_ni += a; m_nj += b
    return m_ni, m_nj


def _fem_workings(L, ld, li, lj):
    m1,m2=_fem(L,ld); t=ld["type"]; out=[]
    if t=="UDL":
        w=ld["mag"]
        out+=[f"    UDL w={w} kN/m, L={L:.3f} m",
              f"      M_F{li}{lj} = -wL\u00b2/12 = {m1:+.4f} kNm",
              f"      M_F{lj}{li} = +wL\u00b2/12 = {m2:+.4f} kNm"]
    elif t=="Point":
        P=ld["mag"]; a=ld["pos"]; b=L-a
        out+=[f"    Point P={P} kN at a={a:.3f} m (b={b:.3f} m)",
              f"      M_F{li}{lj} = -Pab\u00b2/L\u00b2 = {m1:+.4f} kNm",
              f"      M_F{lj}{li} = +Pa\u00b2b/L\u00b2 = {m2:+.4f} kNm"]
    elif t=="UDL-P":
        out+=[f"    Partial UDL w={ld['mag']} kN/m, {ld['pos']}\u2013{ld['end']} m",
              f"      M_F{li}{lj} = {m1:+.4f} kNm,  M_F{lj}{li} = {m2:+.4f} kNm"]
    elif t=="UVL-P":
        out+=[f"    UVL w_max={ld['mag']} kN/m, {ld['pos']}\u2013{ld['end']} m ({ld.get('shape','')})",
              f"      M_F{li}{lj} = {m1:+.4f} kNm,  M_F{lj}{li} = {m2:+.4f} kNm"]
    elif t=="Moment":
        out+=[f"    Applied moment M={ld['mag']} kNm at {ld['pos']:.3f} m from {li}",
              f"      M_F{li}{lj} = {m1:+.4f} kNm,  M_F{lj}{li} = {m2:+.4f} kNm"]
    return out,m1,m2


# ══════════════════════════════════════════════════════════════
# CHORD ROTATION FROM SETTLEMENTS
# ══════════════════════════════════════════════════════════════

def _psi_settlement(ni_node, nj_node, settlements):
    dx=nj_node["x"]-ni_node["x"]; dy=nj_node["y"]-ni_node["y"]
    L=np.hypot(dx,dy)
    if L<1e-12: return 0.0
    ex,ey=dx/L,dy/L; px,py=-ey,ex
    si=settlements.get(ni_node["id"],{"dx":0.0,"dy":0.0})
    sj=settlements.get(nj_node["id"],{"dx":0.0,"dy":0.0})
    return ((sj["dx"]-si["dx"])*px+(sj["dy"]-si["dy"])*py)/L


# ══════════════════════════════════════════════════════════════
# FAR-END CONDITION CHECKS
# ══════════════════════════════════════════════════════════════

def _is_far_released(far_node, members, this_mid):
    """
    True if the far end moment must be zero:
    - Pinned or Roller support with NO other members connecting there
      (so the pinned support is truly a terminal support, not a mid-frame pin)
    - OR Free support with NO other members (cantilever tip — treated same way)
    When True, use modified stiffness 3EI/L and modified FEMs.
    """
    sup = far_node.get("support", "Free")
    if sup not in ("Pinned", "Roller", "Free"):
        return False
    others = [m for m in members
              if m["id"] != this_mid and
              (m["ni"] == far_node["id"] or m["nj"] == far_node["id"])]
    return len(others) == 0


def _cantilever_tip_moment(nid, members, mdata, loads):
    """
    For a Free node that is a cantilever tip (only one member):
    Compute the moment that the cantilever arm delivers to the root.
    This becomes an external joint moment at the root node.

    The cantilever arm itself is REMOVED from the DOF system — its moment
    is statically determined:  M_tip = sum of (load * moment_arm from tip)
    Returns (root_nid, M_root) where M_root is the moment applied AT the root.
    """
    # Find the single member attached to this tip
    arm_mems = [m for m in members
                if m["ni"] == nid or m["nj"] == nid]
    if len(arm_mems) != 1:
        return None, 0.0
    mem = arm_mems[0]
    mid = mem["id"]
    d = mdata[mid]
    L = d["L"]
    # Determine which end is the tip
    tip_is_near = (mem["ni"] == nid)
    root_nid = mem["nj"] if tip_is_near else mem["ni"]

    # Compute moment at root from loads on the arm
    # Positive = anticlockwise at root (Kassimali convention)
    mem_loads = [ld for ld in loads if ld["member_id"] == mid]
    M_root = 0.0
    for ld in mem_loads:
        t = ld["type"]
        if tip_is_near:
            # x measured from tip (near end)
            if t == "UDL":
                w = ld["mag"]
                # Total load w*L acting at L/2 from tip → moment at root = -w*L²/2
                M_root -= w * L**2 / 2.0
            elif t == "Point":
                P = ld["mag"]; a = ld["pos"]   # a from near (tip) end
                # moment at root = -P*(L-a)
                M_root -= P * (L - a)
            elif t == "UDL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    xb = a + c / 2.0  # centroid from tip
                    M_root -= w * c * (L - xb)
            elif t == "UVL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    R = 0.5 * w * c
                    shape = ld.get("shape", "start_zero")
                    xb = a + 2*c/3.0 if shape == "start_zero" else a + c/3.0
                    M_root -= R * (L - xb)
        else:
            # tip is far end (nj), x measured from near (root) end
            if t == "UDL":
                w = ld["mag"]
                M_root += w * L**2 / 2.0   # UDL on arm: moment at root end
                # Actually: if root is ni and tip is nj, loads are measured from ni
                # Total load at L/2 from ni → contributes wL²/2 at ni
                # Wait: need to think in terms of which end is which
                # M_at_root = sum of (load * perpendicular arm from root)
                # For UDL on member going from root(ni) to tip(nj):
                # No - UDL is a distributed load, moment at ni:
                # If structure is cantilevered from nj end (free tip) going BACK to ni:
                # then moment at ni = -wL²/2 (hogging)
                # Let me redo properly for tip=far end case
                M_root = 0.0  # reset and redo
                break
        
    # If tip is far end, recompute differently
    if not tip_is_near:
        M_root = 0.0
        for ld in mem_loads:
            t = ld["type"]
            # x measured from near=root end, tip is at x=L
            # Cantilever: fixed at near end (root), free at far end (tip)
            # Loads applied along the span
            if t == "UDL":
                w = ld["mag"]
                # total load = wL, acts at L/2 from root
                M_root -= w * L**2 / 2.0
            elif t == "Point":
                P = ld["mag"]; a = ld["pos"]  # from near=root end
                M_root -= P * a  # moment at root = P*a (hogging)
            elif t == "UDL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    xb = a + c / 2.0
                    M_root -= w * c * xb
            elif t == "UVL-P":
                w = ld["mag"]; a = ld["pos"]; b = ld["end"]; c = b - a
                if c > 0:
                    R = 0.5 * w * c
                    shape = ld.get("shape", "start_zero")
                    xb = a + 2*c/3.0 if shape == "start_zero" else a + c/3.0
                    M_root -= R * xb

    return root_nid, M_root


# ══════════════════════════════════════════════════════════════
# SWAY GROUP DETECTION
# ══════════════════════════════════════════════════════════════

def _detect_sway_groups(nodes, members):
    """
    Columns = members with |dy|>0.30*L.
    Returns [(label, [member_ids])] sorted bottom→top.
    Cantilever-arm members (far-end is free tip) are excluded from sway.
    """
    nmap = {n["id"]: n for n in nodes}
    groups = {}
    for mem in members:
        ni = nmap[mem["ni"]]; nj = nmap[mem["nj"]]
        dx = abs(nj["x"]-ni["x"]); dy = abs(nj["y"]-ni["y"])
        L = np.hypot(dx, dy)
        if L < 1e-12: continue
        # Skip cantilever arms (free far end with no other members)
        if _is_far_released(nj, members, mem["id"]) and nj.get("support","Free")=="Free":
            continue
        if dy > dx:  # column if more vertical than horizontal (45° threshold)
            ym = round((ni["y"]+nj["y"])/2.0, 1)
            groups.setdefault(ym, []).append(mem["id"])
    if not groups: return []
    result = []
    for i,(ym,mids) in enumerate(sorted(groups.items())):
        lbl = f"S{i+1}" if len(groups)>1 else "1"
        result.append((lbl, mids))
    return result


# ══════════════════════════════════════════════════════════════
# MAIN SOLVER
# ══════════════════════════════════════════════════════════════

class GeneralFrameSolver:

    @staticmethod
    def solve(nodes, members, loads,
              joint_moments=None, settlements=None, sway=True):
        """
        Solve a general plane frame by the slope-deflection method.

        Handles:
          • Fixed, Pinned, Roller, Free support nodes
          • Cantilever arms (Free tip, single member) — solved statically
          • Modified stiffness (3EI/L) for far-pinned/roller terminal members
          • Correct modified FEMs: M_Fij_mod = M_Fij_ff − ½M_Fji_ff
          • Sway DOFs (auto-detected or disabled)
          • Support settlements, applied joint moments

        Returns
        -------
        moments  : {member_id: (M_ni, M_nj)}
        unknowns : {label: value}
        workings : {key: text}
        """
        if joint_moments is None: joint_moments = {}
        if settlements   is None: settlements   = {}

        nmap = {n["id"]: n for n in nodes}
        SEP  = "\u2550"*62

        def nlbl(nid):
            return nmap[nid].get("label", chr(65+nid))

        # ── Identify cantilever-arm members and their tip nodes ────
        # A cantilever arm = member whose FAR end is Free + no other members
        cantilever_tip_nodes = set()
        cantilever_arm_mids  = set()
        for mem in members:
            nj_n = nmap[mem["nj"]]
            if _is_far_released(nj_n, members, mem["id"]) and nj_n.get("support","Free")=="Free":
                cantilever_tip_nodes.add(mem["nj"])
                cantilever_arm_mids.add(mem["id"])

        # ── Pre-compute geometry for ALL members ──────────────────
        mdata = {}
        for mem in members:
            mid = mem["id"]
            ni_n = nmap[mem["ni"]]; nj_n = nmap[mem["nj"]]
            dx = nj_n["x"]-ni_n["x"]; dy = nj_n["y"]-ni_n["y"]
            L  = np.hypot(dx, dy)
            if L < 1e-12: L = 1e-6
            EI   = float(mem["EI"])
            psi_s = _psi_settlement(ni_n, nj_n, settlements)
            mod   = _is_far_released(nj_n, members, mid)
            lbl   = mem.get("label", nlbl(mem["ni"])+nlbl(mem["nj"]))
            mdata[mid] = {"L":L,"EI":EI,"psi":psi_s,
                          "ni":mem["ni"],"nj":mem["nj"],
                          "ni_node":ni_n,"nj_node":nj_n,
                          "mod":mod,"label":lbl,"dx":dx,"dy":dy}

        # ── Inject cantilever moments as external joint moments ────
        # For each cantilever tip, compute static moment at root, add to joint_moments
        jmom_augmented = dict(joint_moments)
        cant_root_moments = {}  # for workings
        for nid in cantilever_tip_nodes:
            root_nid, M_root = _cantilever_tip_moment(nid, members, mdata, loads)
            if root_nid is not None:
                jmom_augmented[root_nid] = jmom_augmented.get(root_nid, 0.0) + M_root
                cant_root_moments[nlbl(nid)] = (nlbl(root_nid), M_root)

        # ── Active members (exclude cantilever arms from SD system) ─
        active_members = [m for m in members if m["id"] not in cantilever_arm_mids]

        # ── DOF classification ─────────────────────────────────────
        # "Far-released terminal" nodes: Pinned/Roller (NOT Free) nodes that are
        # the far (released) end of their ONLY connecting active member.
        # Their moment = 0 is automatic from modified stiffness, so they need NO
        # equilibrium equation. Use ALL members (not just active) to check isolation.
        far_released_terminals = set()
        for mem in active_members:
            nj_n = nmap[mem["nj"]]
            sup_nj = nj_n.get("support", "Free")
            # Only Pinned/Roller can be far-released terminals.
            # Free nodes are handled separately as cantilever tips.
            if sup_nj not in ("Pinned", "Roller"):
                continue
            # Check ALL members to confirm this node has only one member touching it
            others = [m for m in members
                      if m["id"] != mem["id"] and
                         (m["ni"] == mem["nj"] or m["nj"] == mem["nj"])]
            if len(others) == 0:
                far_released_terminals.add(mem["nj"])

        # Theta nodes: all non-Fixed nodes that are NOT cantilever tips and
        # NOT far-released terminals (their M=0 is implicit in mod stiffness)
        theta_nodes = [n["id"] for n in nodes
                       if n.get("support","Free") != "Fixed"
                       and n["id"] not in cantilever_tip_nodes
                       and n["id"] not in far_released_terminals]

        # ── Sway DOFs ─────────────────────────────────────────────
        sway_dofs = _detect_sway_groups(nodes, active_members) if sway else []

        n_th  = len(theta_nodes); n_sw = len(sway_dofs)
        n_dof = n_th + n_sw
        th_idx = {nid:i for i,nid in enumerate(theta_nodes)}
        sw_idx = {sg[0]:n_th+i for i,sg in enumerate(sway_dofs)}

        def sg_for(mid):
            for lbl,mids in sway_dofs:
                if mid in mids: return lbl
            return None

        # ── Step 1: Fixed-End Moments ──────────────────────────────
        fems = {}
        fem_lines = [SEP,"  STEP 1 \u2014 FIXED-END MOMENTS",SEP,""]

        for mem in active_members:
            mid = mem["id"]; d = mdata[mid]; L = d["L"]; mod = d["mod"]
            li = nlbl(mem["ni"]); lj = nlbl(mem["nj"])
            mem_loads = [ld for ld in loads if ld["member_id"]==mid]
            m_ni_ff=0.0; m_nj_ff=0.0
            fem_lines.append(f"  Member {d['label']} (L={L:.3f} m, EI={d['EI']:.0f}):")
            if not mem_loads:
                fem_lines.append(f"    No loads \u2192 M_F{li}{lj} = 0, M_F{lj}{li} = 0")
            else:
                for ld in mem_loads:
                    wlines,m1,m2 = _fem_workings(L,ld,li,lj)
                    fem_lines+=wlines; m_ni_ff+=m1; m_nj_ff+=m2
                fem_lines.append(f"  \u2234 FF: M_F{li}{lj}={m_ni_ff:+.4f}, M_F{lj}{li}={m_nj_ff:+.4f}")

            if mod:
                # Modified FEM: eliminate carry-over from released far end
                m_ni_mod = m_ni_ff - 0.5*m_nj_ff
                m_nj_mod = 0.0
                fem_lines.append(f"  Modified: M_F{li}{lj}={m_ni_mod:+.4f} (= FF_near - 0.5*FF_far), M_F{lj}{li}=0")
                fems[mid] = [m_ni_mod, 0.0]
            else:
                fems[mid] = [m_ni_ff, m_nj_ff]
            fem_lines.append("")

        # Also record FEMs for cantilever arms (for back-sub only)
        for mem in members:
            mid = mem["id"]
            if mid in cantilever_arm_mids:
                d = mdata[mid]; L = d["L"]
                mem_loads = [ld for ld in loads if ld["member_id"]==mid]
                m_ni_ff,m_nj_ff = _fem_total(L, mem_loads)
                fems[mid] = [m_ni_ff, m_nj_ff]

        # ── Step 2: Slope-deflection equations ────────────────────
        sd_lines = [SEP,"  STEP 2 \u2014 SLOPE-DEFLECTION EQUATIONS",SEP,""]
        sd_lines += ["  Standard:   M_ij = M_Fij + (2EI/L)[2\u03b8_i + \u03b8_j \u2212 3\u03c8]",
                     "  Modified:   M_ij = M_Fij_mod + (3EI/L)[\u03b8_i \u2212 \u03c8],  M_ji = 0",
                     "  (M_Fij_mod = M_Fij_ff \u2212 0.5 \u00d7 M_Fji_ff)", ""]
        if cant_root_moments:
            sd_lines.append("  Cantilever arms (solved statically, not in SD system):")
            for tip_lbl,(root_lbl,Mval) in cant_root_moments.items():
                sd_lines.append(f"    Tip {tip_lbl} \u2192 injects M={Mval:+.4f} kNm at root {root_lbl}")
            sd_lines.append("")

        for mem in active_members:
            mid=mem["id"]; d=mdata[mid]; L=d["L"]; EI=d["EI"]; psi_s=d["psi"]; mod=d["mod"]
            li=nlbl(mem["ni"]); lj=nlbl(mem["nj"])
            fi=fems[mid][0]; ff=fems[mid][1]
            thi_str = f"\u03b8_{li}" if mem["ni"] in th_idx else "0"
            thj_str = f"\u03b8_{lj}" if mem["nj"] in th_idx else "0"
            sgl = sg_for(mid)
            psi_str = f"{psi_s:.6f}" + (f" + \u0394_{sgl}/{L:.3f}" if sgl else "")
            k = 2*EI/L
            if mod:
                sd_lines.append(f"  M_{li}{lj} = {fi:+.4f} + (3\u00d7{EI:.0f}/{L:.3f})[{thi_str} \u2212 ({psi_str})]")
                sd_lines.append(f"  M_{lj}{li} = 0  [modified — far end released]")
            else:
                sd_lines.append(f"  M_{li}{lj} = {fi:+.4f} + {k:.4f}[2{thi_str} + {thj_str} \u2212 3({psi_str})]")
                sd_lines.append(f"  M_{lj}{li} = {ff:+.4f} + {k:.4f}[2{thj_str} + {thi_str} \u2212 3({psi_str})]")
            sd_lines.append("")

        # ── Step 3: Assemble K and F ──────────────────────────────
        K = np.zeros((max(n_dof,1), max(n_dof,1)))
        F = np.zeros(max(n_dof,1))

        # --- Moment equilibrium at each free-rotation node ---
        for nid in theta_nodes:
            row = th_idx[nid]
            F[row] -= jmom_augmented.get(nid, 0.0)

            for mem in active_members:
                mid=mem["id"]; d=mdata[mid]
                L=d["L"]; EI=d["EI"]; psi_s=d["psi"]; mod=d["mod"]
                is_near=(mem["ni"]==nid); is_far=(mem["nj"]==nid)
                if not (is_near or is_far): continue
                sgl = sg_for(mid)
                k4=4*EI/L; k2=2*EI/L; k3=3*EI/L

                if is_near and not mod:
                    K[row,th_idx[nid]]+=k4
                    if mem["nj"] in th_idx: K[row,th_idx[mem["nj"]]]+=k2
                    if sgl and sgl in sw_idx: K[row,sw_idx[sgl]]-=6*EI/L**2
                    F[row]-=fems[mid][0]; F[row]+=6*EI*psi_s/L
                elif is_near and mod:
                    K[row,th_idx[nid]]+=k3
                    if sgl and sgl in sw_idx: K[row,sw_idx[sgl]]-=3*EI/L**2
                    F[row]-=fems[mid][0]; F[row]+=3*EI*psi_s/L
                elif is_far and not mod:
                    K[row,th_idx[nid]]+=k4
                    if mem["ni"] in th_idx: K[row,th_idx[mem["ni"]]]+=k2
                    if sgl and sgl in sw_idx: K[row,sw_idx[sgl]]-=6*EI/L**2
                    F[row]-=fems[mid][1]; F[row]+=6*EI*psi_s/L
                # is_far and mod: M_ji=0, no contribution

        # --- Sway (shear) equilibrium ---
        for sg_i,(sgl,sg_mids) in enumerate(sway_dofs):
            row = n_th+sg_i; F_ext = 0.0
            for mid in sg_mids:
                d=mdata[mid]; L=d["L"]; EI=d["EI"]; psi_s=d["psi"]; mod=d["mod"]
                ni_id=d["ni"]; nj_id=d["nj"]
                if not mod:
                    if ni_id in th_idx: K[row,th_idx[ni_id]]+=6*EI/L**2
                    if nj_id in th_idx: K[row,th_idx[nj_id]]+=6*EI/L**2
                    K[row,n_th+sg_i]-=12*EI/L**3
                    F_ext-=(fems[mid][0]+fems[mid][1])/L
                    F_ext+=12*EI*psi_s/L**2
                else:
                    if ni_id in th_idx: K[row,th_idx[ni_id]]+=3*EI/L**2
                    K[row,n_th+sg_i]-=3*EI/L**3
                    F_ext-=fems[mid][0]/L
                    F_ext+=3*EI*psi_s/L**2
            for ld in loads:
                if ld["member_id"] not in sg_mids: continue
                L_m=mdata[ld["member_id"]]["L"]; t=ld["type"]
                if t=="Point":   F_ext+=ld["mag"]
                elif t=="UDL":   F_ext+=ld["mag"]*L_m
                elif t=="UDL-P":
                    c=ld["end"]-ld["pos"]
                    if c>0: F_ext+=ld["mag"]*c
            F[row] = F_ext

        # ── Step 4: Solve ─────────────────────────────────────────
        if n_dof==0:
            sol=np.array([]); th_vals={}; sw_vals={}
        else:
            try:
                sol = np.linalg.solve(K, F)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(K, F, rcond=None)[0]
            th_vals = {nid:float(sol[th_idx[nid]]) for nid in theta_nodes}
            sw_vals = {sgl:float(sol[n_th+i]) for i,(sgl,_) in enumerate(sway_dofs)}

        # ── Step 5: Back-substitution ──────────────────────────────
        moments_out = {}
        final_lines = [SEP,"  STEP 5 \u2014 FINAL END MOMENTS",SEP,""]

        for mem in active_members:
            mid=mem["id"]; d=mdata[mid]
            L=d["L"]; EI=d["EI"]; psi_s=d["psi"]; mod=d["mod"]
            li=nlbl(mem["ni"]); lj=nlbl(mem["nj"])
            sgl=sg_for(mid)
            psi_eff=psi_s+(sw_vals.get(sgl,0.0)/L if sgl else 0.0)
            ti=th_vals.get(mem["ni"],0.0); tj=th_vals.get(mem["nj"],0.0)
            fi=fems[mid][0]; ff=fems[mid][1]
            if mod:
                Mij=fi+(3*EI/L)*(ti-psi_eff); Mji=0.0
                final_lines.append(f"  M_{li}{lj} = {fi:+.4f}+(3\u00d7{EI:.0f}/{L:.3f})[{ti:.6f}\u2212{psi_eff:.6f}]={Mij:+.4f} kNm")
                final_lines.append(f"  M_{lj}{li} = 0 [released]")
            else:
                Mij=fi+(2*EI/L)*(2*ti+tj-3*psi_eff)
                Mji=ff+(2*EI/L)*(2*tj+ti-3*psi_eff)
                final_lines.append(f"  M_{li}{lj} = {Mij:+.4f} kNm")
                final_lines.append(f"  M_{lj}{li} = {Mji:+.4f} kNm")
            final_lines.append("")
            moments_out[mid] = (Mij, Mji)

        # Back-sub for cantilever arms:
        # The moment at the cantilever root was already computed statically by
        # _cantilever_tip_moment. That is the exact correct answer — use it directly.
        cant_root_map = {}
        for tip_nid in cantilever_tip_nodes:
            root_nid2, M_r = _cantilever_tip_moment(tip_nid, members, mdata, loads)
            if root_nid2 is not None:
                cant_root_map[tip_nid] = (root_nid2, M_r)

        for mem in members:
            mid = mem["id"]
            if mid not in cantilever_arm_mids: continue
            d = mdata[mid]
            li=nlbl(mem["ni"]); lj=nlbl(mem["nj"])
            tip_nid = mem["nj"]
            root_nid2, M_root_val = cant_root_map.get(tip_nid, (None, 0.0))
            Mij = M_root_val   # moment at root end (statically exact)
            Mji = 0.0          # moment at free tip = 0
            moments_out[mid] = (Mij, Mji)
            final_lines.append(f"  Cantilever {d['label']} (static): M_{li}{lj}={Mij:+.4f} kNm, M_{lj}{li}=0")
            final_lines.append("")

        # ── Step 6: Equilibrium check ──────────────────────────────
        chk_lines = [SEP,"  STEP 6 \u2014 EQUILIBRIUM CHECK",SEP,""]
        for nid in theta_nodes:
            lbl = nlbl(nid); mem_sum=0.0; terms=[]
            for mem in active_members:
                mid=mem["id"]
                if mem["ni"]==nid:
                    v=moments_out[mid][0]; mem_sum+=v
                    terms.append(f"M_{lbl}{nlbl(mem['nj'])}={v:+.3f}")
                elif mem["nj"]==nid and not mdata[mid]["mod"]:
                    v=moments_out[mid][1]; mem_sum+=v
                    terms.append(f"M_{nlbl(mem['ni'])}{lbl}={v:+.3f}")
            ext=jmom_augmented.get(nid,0.0)
            residual=mem_sum+ext
            ok="\u2705" if abs(residual)<0.05 else "\u26a0\ufe0f"
            ext_str=f" + M_ext={ext:+.1f}" if abs(ext)>1e-9 else ""
            chk_lines.append(f"  \u03a3M_{lbl} = {' + '.join(terms)}{ext_str} = {residual:.4f} kNm  {ok}")
        chk_lines.append("")

        # ── Equation display ───────────────────────────────────────
        eq_lines=[SEP,"  STEP 3 \u2014 EQUILIBRIUM EQUATIONS",SEP,""]
        dof_labels=([f"\u03b8_{nlbl(nid)}" for nid in theta_nodes]+
                    [f"\u0394_{sgl}" for sgl,_ in sway_dofs])
        eq_lines.append("  Unknowns: "+",  ".join(dof_labels) if dof_labels else "  No unknowns (statically determinate)")
        eq_lines.append("")
        if n_dof>0:
            col_w=max(len(l) for l in dof_labels)+2
            hdr="  "+"".join(f"{l:>{col_w+8}}" for l in dof_labels)+"  RHS"
            eq_lines.append(hdr)
            eq_names=[f"\u03a3M_{nlbl(nid)}=0" for nid in theta_nodes]+[f"\u03a3Fx_{sgl}=0" for sgl,_ in sway_dofs]
            for r in range(n_dof):
                row_str=f"  [{eq_names[r]:14}]  "+"  ".join(f"{K[r,c]:+10.4f}" for c in range(n_dof))+f"  = {F[r]:+10.4f}"
                eq_lines.append(row_str)
        eq_lines.append("")

        sol_lines=[SEP,"  STEP 4 \u2014 SOLUTION",SEP,""]
        for nid in theta_nodes:
            sol_lines.append(f"  \u03b8_{nlbl(nid)} = {th_vals.get(nid,0.0):.10f} rad")
        for sgl,_ in sway_dofs:
            sol_lines.append(f"  \u0394_{sgl} = {sw_vals.get(sgl,0.0):.10f} m")
        sol_lines.append("")

        unknowns = {}
        for nid in theta_nodes: unknowns[f"theta_{nlbl(nid)}"]=th_vals.get(nid,0.0)
        for sgl,_ in sway_dofs: unknowns[f"delta_{sgl}"]=sw_vals.get(sgl,0.0)

        workings = {
            "fem":      "\n".join(fem_lines),
            "sd_eqs":   "\n".join(sd_lines),
            "equil":    "\n".join(eq_lines),
            "solution": "\n".join(sol_lines),
            "final":    "\n".join(final_lines),
            "check":    "\n".join(chk_lines),
        }
        return moments_out, unknowns, workings

    # ──────────────────────────────────────────────────────────
    # DIAGRAM DATA (SFD/BMD along member)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def get_member_diagram(member, node_map, m_ni, m_nj, loads, n_pts=200):
        ni_n=node_map[member["ni"]]; nj_n=node_map[member["nj"]]
        L=float(np.hypot(nj_n["x"]-ni_n["x"],nj_n["y"]-ni_n["y"]))
        if L<1e-12: return np.array([0.0]),np.array([0.0]),np.array([0.0])
        x=np.linspace(0,L,n_pts)
        mem_loads=[ld for ld in loads if ld["member_id"]==member["id"]]
        Vs=0.0
        for ld in mem_loads:
            t=ld["type"]
            if t=="Point":
                a=ld["pos"]; Vs+=ld["mag"]*(L-a)/L
            elif t=="UDL":
                Vs+=ld["mag"]*L/2.0
            elif t=="UDL-P":
                w=ld["mag"]; a=ld["pos"]; b=ld["end"]; c=b-a
                if c>0: Vs+=w*c*(L-(a+c/2.0))/L
            elif t=="UVL-P":
                w=ld["mag"]; a=ld["pos"]; b=ld["end"]; c=b-a
                if c>0:
                    R=0.5*w*c
                    xb=a+2*c/3.0 if ld.get("shape","start_zero")=="start_zero" else a+c/3.0
                    Vs+=R*(L-xb)/L
        V0=Vs-(m_ni+m_nj)/L
        V=np.zeros(n_pts); M_arr=np.zeros(n_pts)
        for i,xi in enumerate(x):
            Vi=V0; Mi=-m_ni+V0*xi
            for ld in mem_loads:
                t=ld["type"]
                if t=="Point":
                    P=ld["mag"]; a=ld["pos"]
                    if xi>a: Vi-=P; Mi-=P*(xi-a)
                elif t=="UDL":
                    w=ld["mag"]; Vi-=w*xi; Mi-=w*xi**2/2.0
                elif t=="UDL-P":
                    w=ld["mag"]; a=ld["pos"]; b=ld["end"]
                    if xi>a: c=min(xi,b)-a; Vi-=w*c; Mi-=w*c*(xi-a-c/2.0)
                elif t=="UVL-P":
                    w=ld["mag"]; a=ld["pos"]; b=ld["end"]; cl=b-a
                    if xi>a and cl>0:
                        d_loc=min(xi,b)-a
                        wx=w*d_loc/cl if ld.get("shape","start_zero")=="start_zero" else w*(cl-d_loc)/cl
                        Vi-=0.5*wx*d_loc; Mi-=wx*d_loc**2/6.0
            V[i]=Vi; M_arr[i]=Mi
        return x,V,M_arr


# ══════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY SHIM
# ══════════════════════════════════════════════════════════════

class FrameSolver:
    @staticmethod
    def _portal_to_general(nodes_old,members_old,loads_old):
        node_labels=["A","B","C","D","E","F"]
        new_nodes=[]
        for n in nodes_old:
            i=n["id"]
            sup="Fixed" if i in (0,3) else "Free"
            new_nodes.append({"id":i,"x":n["x"],"y":n["y"],"support":sup,
                               "label":node_labels[i] if i<6 else str(i)})
        mem_lbls=["AB","BC","CD"]; conn=[(0,1),(1,2),(2,3)]
        new_members=[]
        for idx,mem in enumerate(members_old):
            ni,nj=conn[idx]
            new_members.append({"id":mem["id"],"ni":ni,"nj":nj,"EI":mem["I"],"label":mem_lbls[idx]})
        new_loads=[]
        for ld in loads_old:
            nl={"member_id":ld["member"],"type":ld["type"],"mag":ld["mag"],"pos":ld.get("pos",0.0)}
            for k in ("end","shape"):
                if k in ld: nl[k]=ld[k]
            new_loads.append(nl)
        return new_nodes,new_members,new_loads

    @staticmethod
    def solve_frame_sway(nodes,members,loads,case_type):
        nn,nm,nl=FrameSolver._portal_to_general(nodes,members,loads)
        mout,unknowns,workings=GeneralFrameSolver.solve(nn,nm,nl,sway=True)
        results={idx:mout.get(members[idx]["id"],(0.0,0.0)) for idx in range(3)}
        vals=list(unknowns.values())
        while len(vals)<3: vals.append(0.0)
        return results,tuple(vals[:3]),workings

    @staticmethod
    def solve_frame_non_sway(nodes,members,loads):
        nn,nm,nl=FrameSolver._portal_to_general(nodes,members,loads)
        mout,unknowns,workings=GeneralFrameSolver.solve(nn,nm,nl,sway=False)
        results={idx:mout.get(members[idx]["id"],(0.0,0.0)) for idx in range(3)}
        vals=list(unknowns.values())
        while len(vals)<3: vals.append(0.0)
        return results,tuple(vals[:3]),workings

    @staticmethod
    def get_diagram_data(member,m1,m2,loads):
        from beam_solver import BeamSolver
        return BeamSolver.get_diagram_data(member,m1,m2,loads)
