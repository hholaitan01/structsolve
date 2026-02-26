# 🏗️ StructSolve

**A professional structural analysis web app built with Python & Streamlit**

> Continuous beam analysis · General frame analysis · RC beam design — all following **BS 8110-1:1997**

---

## ✨ Features

### 🔩 Beam Analysis
- Continuous beams with any number of spans and support types (Fixed, Pinned, Roller)
- Loads: Full UDL, Partial UDL, UVL (triangular), Point loads
- Support settlements and prescribed rotations
- Full **Slope Deflection Method** workings — step by step
- SFD & BMD diagrams
- RC design integration (analyse then design in one click)

### 🏛️ Frame Analysis *(General Solver — v4.0)*
- **Any plane frame topology** — portal, L-shape, T-shape, multi-bay, multi-storey, propped cantilever
- Any number of nodes and members
- Support types: Fixed, Pinned, Roller, Free (cantilever tip)
- **Applied joint moments** (e.g. 150 kNm at C)
- **Support settlements** at any node (vertical & horizontal)
- **Modified stiffness** — auto-detects far-end pinned/roller members and applies 3EI/L
- **Automatic sway group detection** — single and multi-storey
- Full 6-step workings: FEMs → SD equations → equilibrium matrix → solution → back-substitution → check
- BMD overlay diagram
- 6 built-in quick templates

### 🧱 RC Beam Design (BS 8110)
- Rectangular and T-beam sections
- Simply supported and continuous beams
- Flexural design — Cl. 3.4.4 (singly and doubly reinforced)
- Shear design — Cl. 3.4.5
- Deflection check — Cl. 3.4.6
- Cross-section diagram with reinforcement visualisation
- Full design workings

### 📄 PDF Export
- Professional PDF reports for all three modules
- Includes diagrams, tables, and full step-by-step workings

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/structsolve.git
cd structsolve

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run combine.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit
numpy
matplotlib
pandas
reportlab
scipy
```

See `requirements.txt` for pinned versions.

---

## 🗂️ Project Structure

```
structsolve/
├── combine.py          # Main Streamlit app — all UI pages
├── beam_solver.py      # Continuous beam solver (Slope Deflection Method)
├── frame_solver.py     # General plane frame solver
├── bs8110.py           # BS 8110 RC design calculations
├── pdf_export.py       # ReportLab PDF generation
├── requirements.txt
└── README.md
```

---

## 📐 Theory

All structural analysis uses the **Slope Deflection Method**:

```
M_ij = M_Fij + (2EI/L)[2θᵢ + θⱼ − 3ψ]
```

Where:
- `M_Fij` = Fixed-end moment
- `EI/L` = member stiffness
- `θᵢ, θⱼ` = joint rotations (unknowns)
- `ψ = Δ/L` = chord rotation from sway or settlement

RC design follows **BS 8110-1:1997**:
- Flexure: `K = M/(fcu·b·d²)`, `z = d[0.5 + √(0.25 − K/0.9)]`
- Shear: `v = V/(b·d)` vs `vc` from Table 3.8
- Deflection: basic span/depth ratio × modification factor

---

## 🖼️ Screenshots

| Beam Analysis | Frame Analysis | RC Design |
|---|---|---|
| SFD & BMD with full workings | General frame BMD overlay | Cross-section with reinforcement |

---

## 📋 Supported Frame Types

| Type | Example | Template |
|---|---|---|
| Fixed base portal | Classic 3-member portal | ✅ |
| Pinned base portal | Columns pinned at base | ✅ |
| L-shaped frame | Column + horizontal beam | ✅ |
| 4° indeterminate | Multi-member with settlement | ✅ |
| Multi-bay | Two-bay single storey | ✅ |
| Propped cantilever | Roller-supported column | ✅ |
| Custom | Any topology you define | ✅ |

---

## 🤝 Contributing

Pull requests welcome. For major changes, please open an issue first.

---

## 📄 Licence

MIT — free to use, modify, and distribute.

---

*Built with Python · Streamlit · NumPy · Matplotlib · ReportLab*
