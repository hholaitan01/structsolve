"""
GeneralFrameSolver v4.1 — Slope Deflection Method for ANY plane frame
======================================================================
Key improvements in v4.1:
  * Correct sway detection: only truly vertical/inclined members form sway groups
  * Correct shear-equilibrium equation for sway frames
  * Free cantilever tip: theta is unknown, M=0 at free tip enforced naturally
  * Modified stiffness 3EI/L for far-pinned/roller members
  * Full 6-step algebraic workings output
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# FIXED-END MOMENT TABLE
# ══════════════════════════════════════════════════════════════

def _fem(L, load):
    t = load["type"]
    if t == "UDL":
        w = load["mag"]
        return -w*L**2/12.0, +w*L**2/12.0
    if t == "UDL-P":
        w=load["mag"]; a=load["pos"]; b=load["end"]; c=b-a
        if c<=0: return 0.0,0.0
        R=w*c; xb=a+c/2.0
        return -R*(L-xb)**2*xb/L**2, +R*xb**2*(L-xb)/L**2
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
    return 0.0,0.0


def _fem_workings(L, ld, li, lj):
    m1,m2=_fem(L,ld); t=ld["type"]; out=[]
    if t=="UDL":
        w=ld["mag"]
        out+=[f"    UDL w={w} kN/m, L={L:.3f} m",
              f"      M_F{li}{lj} = -wL\u00b2/12 = -{w}\u00d7{L:.3f}\u00b2/12 = {m1:+.4f} kNm",
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
    delta_perp=((sj["dx"]-si["dx"])*px+(sj["dy"]-si["dy"])*py)
    return delta_perp/L


# ══════════════════════════════════════════════════════════════
# MODIFIED STIFFNESS
# ══════════════════════════════════════════════════════════════

def _is_far_pinned(far_node, members, this_mid):
    sup=far_node.get("support","Free")
    if sup not in ("Pinned","Roller"): return False
    others=[m for m in members if m["id"]!=this_mid and
            (m["ni"]==far_node["id"] or m["nj"]==far_node["id"])]
    return len(others)==0


# ══════════════════════════════════════════════════════════════
# SWAY GROUP DETECTION  (fixed in v4.1)
# ══════════════════════════════════════════════════════════════

def _detect_sway_groups(nodes, members):
    """
    A column is a member whose vertical component |dy| > 0.30*L.
    This correctly excludes horizontal beams from sway groups.
    Returns [(label, [member_ids])] sorted bottom->top.
    """
    nmap={n["id"]:n for n in nodes}
    groups={}
    for mem in members:
        ni=nmap[mem["ni"]]; nj=nmap[mem["nj"]]
        dx=abs(nj["x"]-ni["x"]); dy=abs(nj["y"]-ni["y"])
        L=np.hypot(dx,dy)
        if L<1e-12: continue
        if dy > 0.30*L:   # column criterion: >30% vertical
            ym=round((ni["y"]+nj["y"])/2.0,1)
            groups.setdefault(ym,[]).append(mem["id"])
    if not groups: return []
    result=[]
    for i,(ym,mids) in enumerate(sorted(groups.items())):
        lbl=f"S{i+1}" if len(groups)>1 else "1"
        result.append((lbl,mids))
    return result


# ══════════════════════════════════════════════════════════════
# MAIN SOLVER
# ══════════════════════════════════════════════════════════════

class GeneralFrameSolver:

    @staticmethod
    def solve(nodes, members, loads,
              joint_moments=None, settlements=None, sway=True):
        """
        Solve a general plane frame by slope-deflection method.
        
        Parameters
        ----------
        nodes   : list of {id, x, y, support, label}
                  support: "Fixed"|"Pinned"|"Roller"|"Free"
        members : list of {id, ni, nj, EI, label}
        loads   : list of {member_id, type, mag, pos, [end], [shape]}
        joint_moments : {node_id: moment_kNm}  (+ve anticlockwise)
        settlements   : {node_id: {dx, dy}}  (m)
        sway    : True = auto-detect sway DOFs; False = no sway

        Returns
        -------
        moments  : {member_id: (M_ni, M_nj)}
        unknowns : {label: value}
        workings : {key: text}
        """
        if joint_moments is None: joint_moments={}
        if settlements   is None: settlements={}

        nmap={n["id"]:n for n in nodes}
        SEP="\u2550"*62

        def nlbl(nid):
            return nmap[nid].get("label", chr(65+nid))

        # ── DOF classification ────────────────────────────────
        # theta_nodes: all non-Fixed nodes (θ unknown)
        theta_nodes=[n["id"] for n in nodes
                     if n.get("support","Free") != "Fixed"]

        # ── Sway DOFs ─────────────────────────────────────────
        sway_dofs=_detect_sway_groups(nodes,members) if sway else []

        n_th=len(theta_nodes); n_sw=len(sway_dofs)
        n_dof=n_th+n_sw
        th_idx={nid:i for i,nid in enumerate(theta_nodes)}
        sw_idx={sg[0]:n_th+i for i,sg in enumerate(sway_dofs)}

        def sg_for(mid):
            for lbl,mids in sway_dofs:
                if mid in mids: return lbl
            return None

        # ── Pre-compute member geometry ───────────────────────
        mdata={}
        for mem in members:
            mid=mem["id"]
            ni_n=nmap[mem["ni"]]; nj_n=nmap[mem["nj"]]
            dx=nj_n["x"]-ni_n["x"]; dy=nj_n["y"]-ni_n["y"]
            L=np.hypot(dx,dy)
            if L<1e-12: L=1e-6
            EI=float(mem["EI"])
            psi_s=_psi_settlement(ni_n,nj_n,settlements)
            mod=_is_far_pinned(nj_n,members,mid)
            lbl=mem.get("label", nlbl(mem["ni"])+nlbl(mem["nj"]))
            mdata[mid]={"L":L,"EI":EI,"psi":psi_s,
                        "ni":mem["ni"],"nj":mem["nj"],
                        "ni_node":ni_n,"nj_node":nj_n,
                        "mod":mod,"label":lbl,"dx":dx,"dy":dy}

        # ── Step 1: Fixed-End Moments ─────────────────────────
        fems={}
        fem_lines=[SEP,"  STEP 1 \u2014 FIXED-END MOMENTS",SEP,""]
        for mem in members:
            mid=mem["id"]; d=mdata[mid]; L=d["L"]
            li=nlbl(mem["ni"]); lj=nlbl(mem["nj"])
            mem_loads=[ld for ld in loads if ld["member_id"]==mid]
            m_ni=0.0; m_nj=0.0
            fem_lines.append(f"  Member {d['label']} (L = {L:.3f} m, EI = {d['EI']:.0f}):")
            if not mem_loads:
                fem_lines.append(f"    No loads \u2192 M_F{li}{lj} = 0,  M_F{lj}{li} = 0")
            else:
                for ld in mem_loads:
                    wlines,m1,m2=_fem_workings(L,ld,li,lj)
                    fem_lines+=wlines; m_ni+=m1; m_nj+=m2
                fem_lines.append(f"  \u2234 Total M_F{li}{lj} = {m_ni:+.4f} kNm,  M_F{lj}{li} = {m_nj:+.4f} kNm")
            fem_lines.append("")
            fems[mid]=[m_ni,m_nj]

        # ── Step 2: Slope-deflection equations ────────────────
        sd_lines=[SEP,"  STEP 2 \u2014 SLOPE-DEFLECTION EQUATIONS",SEP,""]
        sd_lines+=["  Standard:  M_ij = M_Fij + (2EI/L)[2\u03b8_i + \u03b8_j \u2212 3\u03c8]",
                   "  Modified (far-pinned): M_ij = M_Fij + (3EI/L)[\u03b8_i \u2212 \u03c8],  M_ji = 0",""]
        for mem in members:
            mid=mem["id"]; d=mdata[mid]; L=d["L"]; EI=d["EI"]; ps=d["psi"]; mod=d["mod"]
            li=nlbl(mem["ni"]); lj=nlbl(mem["nj"])
            fi=fems[mid][0]; ff=fems[mid][1]
            thi_str=f"\u03b8_{li}" if mem["ni"] in th_idx else "0"
            thj_str=f"\u03b8_{lj}" if mem["nj"] in th_idx else "0"
            sgl=sg_for(mid)
            psi_str=f"{ps:.6f}"+(f" + \u0394_{sgl}/{L:.3f}" if sgl else "")
            k=2*EI/L
            if mod:
                sd_lines.append(f"  M_{li}{lj} = {fi:+.4f} + (3\u00d7{EI:.0f}/{L:.3f})[{thi_str} \u2212 ({psi_str})]  [modified stiffness]")
                sd_lines.append(f"  M_{lj}{li} = 0  [far end released]")
            else:
                sd_lines.append(f"  M_{li}{lj} = {fi:+.4f} + {k:.4f}[2{thi_str} + {thj_str} \u2212 3({psi_str})]")
                sd_lines.append(f"  M_{lj}{li} = {ff:+.4f} + {k:.4f}[2{thj_str} + {thi_str} \u2212 3({psi_str})]")
            sd_lines.append("")

        # ── Step 3: Assemble stiffness matrix ─────────────────
        K=np.zeros((max(n_dof,1),max(n_dof,1))); F=np.zeros(max(n_dof,1))

        # --- Moment equilibrium at each free-rotation node ---
        for nid in theta_nodes:
            row=th_idx[nid]
            F[row]-=joint_moments.get(nid,0.0)

            for mem in members:
                mid=mem["id"]; d=mdata[mid]
                L=d["L"]; EI=d["EI"]; psi_s=d["psi"]; mod=d["mod"]
                is_near=(mem["ni"]==nid); is_far=(mem["nj"]==nid)
                if not (is_near or is_far): continue
                sgl=sg_for(mid)
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

        # --- Sway (shear) equilibrium: ΣH = 0 for each storey ---
        # For each column: column shear = (M_ni + M_nj) / h
        # Substituting SD eqs:
        #   M_ni+M_nj = FEM_ni+FEM_nj + (6EI/L)(θ_i+θ_j) - 12EI*Δ/L² + 12EI*ψ_s/L
        # So contribution of one column to ΣH = (M_ni+M_nj)/L :
        #   (6EI/L²)(θ_i+θ_j) - (12EI/L³)Δ = RHS
        for sg_i,(sgl,sg_mids) in enumerate(sway_dofs):
            row=n_th+sg_i; F_ext=0.0
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
                    # modified: only M_ni term, M_nj=0
                    if ni_id in th_idx: K[row,th_idx[ni_id]]+=3*EI/L**2
                    K[row,n_th+sg_i]-=3*EI/L**3
                    F_ext-=fems[mid][0]/L
                    F_ext+=3*EI*psi_s/L**2

            # External lateral loads on columns in this storey
            for ld in loads:
                if ld["member_id"] not in sg_mids: continue
                L_m=mdata[ld["member_id"]]["L"]
                t=ld["type"]
                if t=="Point":   F_ext+=ld["mag"]
                elif t=="UDL":   F_ext+=ld["mag"]*L_m
                elif t=="UDL-P":
                    c=ld["end"]-ld["pos"]
                    if c>0: F_ext+=ld["mag"]*c
            F[row]=F_ext

        # ── Step 4: Solve ─────────────────────────────────────
        if n_dof==0:
            sol=np.array([]); th_vals={}; sw_vals={}
        else:
            try:
                sol=np.linalg.solve(K,F)
            except np.linalg.LinAlgError:
                sol=np.linalg.lstsq(K,F,rcond=None)[0]
            th_vals={nid:float(sol[th_idx[nid]]) for nid in theta_nodes}
            sw_vals={sgl:float(sol[n_th+i]) for i,(sgl,_) in enumerate(sway_dofs)}

        # ── Step 5: Back-substitution ──────────────────────────
        moments_out={}
        final_lines=[SEP,"  STEP 5 \u2014 FINAL END MOMENTS (back-substitution)",SEP,""]
        for mem in members:
            mid=mem["id"]; d=mdata[mid]
            L=d["L"]; EI=d["EI"]; psi_s=d["psi"]; mod=d["mod"]
            li=nlbl(mem["ni"]); lj=nlbl(mem["nj"])
            sgl=sg_for(mid)
            psi_eff=psi_s+(sw_vals.get(sgl,0.0)/L if sgl else 0.0)
            ti=th_vals.get(mem["ni"],0.0); tj=th_vals.get(mem["nj"],0.0)
            fi=fems[mid][0]; ff=fems[mid][1]
            if mod:
                Mij=fi+(3*EI/L)*(ti-psi_eff); Mji=0.0
                final_lines.append(f"  M_{li}{lj} = {fi:+.4f} + (3\u00d7{EI:.0f}/{L:.4f})[{ti:.8f} \u2212 {psi_eff:.8f}] = {Mij:+.4f} kNm")
                final_lines.append(f"  M_{lj}{li} = 0  [released]")
            else:
                Mij=fi+(2*EI/L)*(2*ti+tj-3*psi_eff)
                Mji=ff+(2*EI/L)*(2*tj+ti-3*psi_eff)
                final_lines.append(f"  M_{li}{lj} = {fi:+.4f} + (2\u00d7{EI:.0f}/{L:.4f})[2\u00d7{ti:.8f} + {tj:.8f} \u2212 3\u00d7{psi_eff:.8f}] = {Mij:+.4f} kNm")
                final_lines.append(f"  M_{lj}{li} = {ff:+.4f} + (2\u00d7{EI:.0f}/{L:.4f})[2\u00d7{tj:.8f} + {ti:.8f} \u2212 3\u00d7{psi_eff:.8f}] = {Mji:+.4f} kNm")
            final_lines.append("")
            moments_out[mid]=(Mij,Mji)

        # ── Step 6: Equilibrium check ──────────────────────────
        chk_lines=[SEP,"  STEP 6 \u2014 EQUILIBRIUM CHECK",SEP,""]
        for nid in theta_nodes:
            lbl=nlbl(nid)
            mem_sum=0.0; terms=[]
            for mem in members:
                mid=mem["id"]
                if mem["ni"]==nid:
                    v=moments_out[mid][0]; mem_sum+=v
                    terms.append(f"M_{lbl}{nlbl(mem['nj'])}={v:+.3f}")
                elif mem["nj"]==nid and not mdata[mid]["mod"]:
                    v=moments_out[mid][1]; mem_sum+=v
                    terms.append(f"M_{nlbl(mem['ni'])}{lbl}={v:+.3f}")
            ext=joint_moments.get(nid,0.0)
            residual=mem_sum+ext
            ok="\u2705" if abs(residual)<0.05 else "\u26a0\ufe0f"
            ext_str=f" + M_ext={ext:+.1f}" if abs(ext)>1e-9 else ""
            chk_lines.append(f"  \u03a3M_{lbl} = {' + '.join(terms)}{ext_str} = {residual:.4f} kNm  {ok}")
        chk_lines.append("")

        # ── Equation display ───────────────────────────────────
        eq_lines=[SEP,"  STEP 3 \u2014 EQUILIBRIUM EQUATIONS",SEP,""]
        dof_labels=([f"\u03b8_{nlbl(nid)}" for nid in theta_nodes]+
                    [f"\u0394_{sgl}" for sgl,_ in sway_dofs])
        eq_lines.append("  Unknowns: "+",  ".join(dof_labels))
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

        unknowns={}
        for nid in theta_nodes: unknowns[f"theta_{nlbl(nid)}"]=th_vals.get(nid,0.0)
        for sgl,_ in sway_dofs: unknowns[f"delta_{sgl}"]=sw_vals.get(sgl,0.0)

        workings={
            "fem":      "\n".join(fem_lines),
            "sd_eqs":   "\n".join(sd_lines),
            "equil":    "\n".join(eq_lines),
            "solution": "\n".join(sol_lines),
            "final":    "\n".join(final_lines),
            "check":    "\n".join(chk_lines),
        }
        return moments_out, unknowns, workings

    # ──────────────────────────────────────────────────────────
    # DIAGRAM DATA
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def get_member_diagram(member, node_map, m_ni, m_nj, loads, n_pts=200):
        """Returns (x, V, M) arrays along member local axis (x=0 at ni)."""
        ni_n=node_map[member["ni"]]; nj_n=node_map[member["nj"]]
        L=float(np.hypot(nj_n["x"]-ni_n["x"], nj_n["y"]-ni_n["y"]))
        if L<1e-12: return np.array([0.0]),np.array([0.0]),np.array([0.0])
        x=np.linspace(0,L,n_pts)
        mem_loads=[ld for ld in loads if ld["member_id"]==member["id"]]

        # Reaction at near end from loads (for ΣM=0 about far end)
        Vs=0.0
        for ld in mem_loads:
            t=ld["type"]
            if t=="Point":
                a=ld["pos"]; b=L-a
                Vs+=ld["mag"]*b/L
            elif t=="UDL":
                Vs+=ld["mag"]*L/2.0
            elif t=="UDL-P":
                w=ld["mag"]; a=ld["pos"]; b=ld["end"]; c=b-a
                if c>0: Vs+=w*c*(L-(a+c/2.0))/L
            elif t=="UVL-P":
                w=ld["mag"]; a=ld["pos"]; b=ld["end"]; c=b-a
                if c>0:
                    R=0.5*w*c
                    shape=ld.get("shape","start_zero")
                    xb=a+2*c/3.0 if shape=="start_zero" else a+c/3.0
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
                        shape=ld.get("shape","start_zero")
                        wx=w*d_loc/cl if shape=="start_zero" else w*(cl-d_loc)/cl
                        Vi-=0.5*wx*d_loc; Mi-=wx*d_loc**2/6.0
            V[i]=Vi; M_arr[i]=Mi
        return x,V,M_arr


# ══════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY SHIM
# ══════════════════════════════════════════════════════════════

class FrameSolver:
    """Backward-compatible wrapper for old combine.py calls."""

    @staticmethod
    def _portal_to_general(nodes_old, members_old, loads_old):
        node_labels=["A","B","C","D","E","F"]
        new_nodes=[]
        for n in nodes_old:
            i=n["id"]
            sup="Fixed" if i in (0,3) else "Free"
            new_nodes.append({"id":i,"x":n["x"],"y":n["y"],
                               "support":sup,"label":node_labels[i] if i<6 else str(i)})
        mem_lbls=["AB","BC","CD"]; conn=[(0,1),(1,2),(2,3)]
        new_members=[]
        for idx,mem in enumerate(members_old):
            ni,nj=conn[idx]
            new_members.append({"id":mem["id"],"ni":ni,"nj":nj,
                                 "EI":mem["I"],"label":mem_lbls[idx]})
        new_loads=[]
        for ld in loads_old:
            nl={"member_id":ld["member"],"type":ld["type"],
                "mag":ld["mag"],"pos":ld.get("pos",0.0)}
            for k in ("end","shape"):
                if k in ld: nl[k]=ld[k]
            new_loads.append(nl)
        return new_nodes,new_members,new_loads

    @staticmethod
    def solve_frame_sway(nodes, members, loads, case_type):
        nn,nm,nl=FrameSolver._portal_to_general(nodes,members,loads)
        mout,unknowns,workings=GeneralFrameSolver.solve(nn,nm,nl,sway=True)
        results={idx:mout.get(members[idx]["id"],(0.0,0.0)) for idx in range(3)}
        vals=list(unknowns.values())
        while len(vals)<3: vals.append(0.0)
        return results,tuple(vals[:3]),workings

    @staticmethod
    def solve_frame_non_sway(nodes, members, loads):
        nn,nm,nl=FrameSolver._portal_to_general(nodes,members,loads)
        mout,unknowns,workings=GeneralFrameSolver.solve(nn,nm,nl,sway=False)
        results={idx:mout.get(members[idx]["id"],(0.0,0.0)) for idx in range(3)}
        vals=list(unknowns.values())
        while len(vals)<3: vals.append(0.0)
        return results,tuple(vals[:3]),workings

    @staticmethod
    def get_diagram_data(member, m1, m2, loads):
        from beam_solver import BeamSolver
        return BeamSolver.get_diagram_data(member, m1, m2, loads)
