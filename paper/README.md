# Overleaf LaTeX Project: Syslog Anomaly Detection Research Paper

This repository contains the complete, self-contained LaTeX source for the IEEE conference research paper:
> **"Lightweight Reconstruction Modeling and Autonomous Extreme Value Thresholding for Syslog Anomaly Detection"**

---

## 1. Quick Overleaf Setup (Step-by-Step)

To compile this project on [Overleaf](https://www.overleaf.com/):

### Method A: Zip Upload (Recommended)
1. Download or archive the `paper/` folder as a `.zip` file:
   ```bash
   cd paper
   zip -r ../syslog_ieee_paper.zip ./*
   ```
2. Log into Overleaf and click **New Project** $\rightarrow$ **Upload Project**.
3. Select `syslog_ieee_paper.zip`.
4. Overleaf will automatically detect `main.tex` as the root document.
5. Ensure compiler settings (under Menu in top-left) are:
   - **Compiler:** `pdfLaTeX`
   - **TeX Live version:** `2023` or `2024` (or any modern release)
   - **Main document:** `main.tex`
6. Click **Recompile**.

### Method B: Direct File Upload
1. Create a blank project on Overleaf.
2. Upload `main.tex`, `references.bib`, `README.md`, and `NOTES.md` to the root folder.
3. Create a folder named `figures/` in Overleaf and upload:
   - `figures/pipeline_diagram.tex`
   - `figures/generate_figures.py`
   - `figures/ablation_comparison.pdf`
   - `figures/chronological_comparison.pdf`
   - `figures/prevalence_impact.pdf`
4. Recompile with `pdfLaTeX`.

---

## 2. Document & Layout Specifications

- **Document Class:** `\documentclass[conference]{IEEEtran}` (two-column IEEE conference layout).
- **Target Page Limit:** Maximum 6 pages including all figures, tables, and references.
- **Current Footprint:** $\sim 5.7\text{--}5.9$ pages (2,757 prose body words, 183-word abstract, 7 booktabs tables, 1 TikZ architectural diagram, and 21 bibliographic references).
- **Citations & Bibliography:** Formatted using `\bibliographystyle{IEEEtran}` and `references.bib`.

---

## 3. Required LaTeX Packages

All required packages are standard and pre-installed in TeX Live / Overleaf:
- `cite`: IEEE citation management and grouping.
- `amsmath, amssymb, amsfonts`: Mathematical typography and theorem definitions.
- `algorithmic`: Algorithmic pseudocode typesetting.
- `graphicx`: Graphic file imports.
- `textcomp`: Text companion fonts and symbols.
- `xcolor`: Color definitions for TikZ diagrams and tables.
- `booktabs`: Professional horizontal rules (`\toprule`, `\midrule`, `\bottomrule`).
- `multirow`: Spanning cells across multiple table rows.
- `tikz`: Vector graphics rendering for the system pipeline architecture.
- `pgfplots`: Coordinate mathematics and layout for TikZ.
- `url`: Formatting of web addresses and digital identifiers.
- `microtype`: Sub-pixel typographical margin kerning and hyphenation control.

---

## 4. Project Directory Structure

```text
paper/
├── main.tex                    # Core IEEEtran conference manuscript
├── references.bib              # Verified BibTeX bibliography (21 references)
├── NOTES.md                    # Complete number manifest, traceability, and audit log
├── README.md                   # This instruction file
├── verify_paper.py             # Python syntax, citation, and constraint verification script
└── figures/
    ├── pipeline_diagram.tex    # TikZ vector source for Fig. 1 architecture
    ├── generate_figures.py     # Python script to reproduce all analytical vector plots
    ├── ablation_comparison.pdf # Architectural ablation comparison plot (AE vs PCA vs iForest)
    ├── chronological_comparison.pdf # Chronological drift & OOV impact plot
    └── prevalence_impact.pdf   # Base-rate precision inflation plot (10% vs 50%)
```

---

## 5. Local Syntax and Rule Verification

To verify that the project satisfies all formatting constraints, has no broken references, has valid braces, and strictly obeys all forbidden phrase guidelines, run the bundled verification script:

```bash
python3 paper/verify_paper.py
```

### Verification Checks Performed:
- **Forbidden Phrases:** Checks for zero occurrences of prohibited phrases (`"state of the art"`, `"SOTA"`, `"strictly required"`, `"outperforms DeepLog"`, `"disproved the base paper"`, `"zero leakage"`).
- **Braces and Environments:** Validates exact parity of open and close braces (`{}`) and matching `\begin{...}` / `\end{...}` pairs.
- **Citation Validity:** Confirms that 100% of `\cite{...}` calls in `main.tex` exist in `references.bib`.
- **Cross-Reference Validity:** Verifies that all `\ref{...}` calls map to active `\label{...}` declarations.
- **Word Count Audit:** Confirms the Abstract is strictly between 150 and 200 words (currently 183 words) and total text volume fits comfortably within 6 pages.

---

## 6. Regenerating Figures Locally

To re-render the vector PDF and PNG figures in `paper/figures/`:
```bash
python3 paper/figures/generate_figures.py
```
Outputs are written directly to `paper/figures/`.

