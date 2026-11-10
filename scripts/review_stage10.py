"""
scripts/review_stage10.py
==========================
Stage 10 Review: BTP Thesis completeness check.

Validates:
  A) Thesis source files
  B) All 7 chapters present and non-empty
  C) Preliminary pages present
  D) All figures referenced exist
  E) All tables input'd exist
  F) Appendices present
  G) Bibliography shared with paper
  H) Submission readiness (no [FILL] placeholders)

Usage: python scripts/review_stage10.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
issues = []

def check(label, condition, detail="", warn=False):
    if condition:
        print(f"  {PASS} {label}" + (f" — {detail}" if detail else ""))
    elif warn:
        print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        issues.append(label)

THESIS = "thesis/main.tex"
BIB    = "paper/references.bib"

print("\n" + "="*65)
print("  STAGE 10: BTP Thesis Review")
print("="*65)

# ----------------------------------------------------------------
# A: Source files
# ----------------------------------------------------------------
print("\n  [A] Thesis Source Files")
check("thesis/main.tex exists",     os.path.exists(THESIS))
check("paper/references.bib exists (shared)", os.path.exists(BIB))
check("thesis/ directory exists",   os.path.isdir("thesis"))

# ----------------------------------------------------------------
# B: Chapter structure
# ----------------------------------------------------------------
print("\n  [B] Chapter Structure (7 chapters)")
if os.path.exists(THESIS):
    tex = open(THESIS, encoding="utf-8").read()
    chapters = [
        (r"\\chapter\{Introduction\}",             "Ch 1: Introduction"),
        (r"\\chapter\{Literature Review\}",         "Ch 2: Literature Review"),
        (r"\\chapter\{Methodology\}",               "Ch 3: Methodology"),
        (r"\\chapter\{Implementation\}",            "Ch 4: Implementation"),
        (r"\\chapter\{Results and Analysis\}",      "Ch 5: Results & Analysis"),
        (r"\\chapter\{Discussion\}",                "Ch 6: Discussion"),
        (r"\\chapter\{Conclusion and Future Work\}", "Ch 7: Conclusion"),
    ]
    for pattern, label in chapters:
        check(label, bool(re.search(pattern, tex)))

    # Appendices
    check("Appendix A (Column Mapping)",    r"\chapter{NHANES Column Mapping}" in tex)
    check("Appendix B (Full Stats Tables)", r"\chapter{Full Statistical Tables}" in tex)
    check("Appendix C (Additional Figures)",r"\chapter{Additional Figures}" in tex)
    check("Appendix D (Code Repository)",   r"\chapter{Code Repository}" in tex)

# ----------------------------------------------------------------
# C: Preliminary pages
# ----------------------------------------------------------------
print("\n  [C] Preliminary Pages")
if os.path.exists(THESIS):
    prelims = [
        (r"\\chapter\*\{Certificate\}",        "Certificate page"),
        (r"\\chapter\*\{Declaration\}",         "Declaration page"),
        (r"\\chapter\*\{Acknowledgements\}",    "Acknowledgements"),
        (r"\\chapter\*\{Abstract\}",            "Abstract"),
        (r"\\tableofcontents",                  "Table of Contents"),
        (r"\\listoffigures",                    "List of Figures"),
        (r"\\listoftables",                     "List of Tables"),
        (r"\\begin\{titlepage\}",               "Title page"),
    ]
    for pattern, label in prelims:
        check(label, bool(re.search(pattern, tex)))

# ----------------------------------------------------------------
# D: Figures referenced in thesis exist
# ----------------------------------------------------------------
print("\n  [D] Figures Referenced in Thesis")
if os.path.exists(THESIS):
    tex = open(THESIS, encoding="utf-8").read()
    fig_refs = re.findall(r'\\includegraphics\[.*?\]\{(.*?)\}', tex)
    for ref in fig_refs:
        # Paths in thesis are relative to thesis/ — resolve from project root
        path = os.path.normpath(os.path.join("thesis", ref))
        exists = os.path.exists(path)
        check(f"'{os.path.basename(ref)}' exists",
              exists,
              path if exists else "Run run_analysis.py to generate",
              warn=not exists)

# ----------------------------------------------------------------
# E: Tables input'd exist
# ----------------------------------------------------------------
print("\n  [E] Tables Input'd in Thesis")
if os.path.exists(THESIS):
    tex = open(THESIS, encoding="utf-8").read()
    table_refs = re.findall(r'\\input\{(.*?)\}', tex)
    for ref in table_refs:
        path = os.path.normpath(os.path.join("thesis", ref))
        if not path.endswith('.tex'):
            path += '.tex'
        exists = os.path.exists(path)
        check(f"'{os.path.basename(ref)}' exists",
              exists,
              path if exists else "Run run_analysis.py first",
              warn=not exists)

# ----------------------------------------------------------------
# F: Key thesis content
# ----------------------------------------------------------------
print("\n  [F] Key Thesis Content")
if os.path.exists(THESIS):
    tex = open(THESIS, encoding="utf-8").read()
    content_checks = [
        ("explanation equivalence",           "Equivalence metric defined"),
        ("lie budget",                        "Lie budget defined"),
        ("pre-registered",                    "Pre-registration mentioned"),
        ("NHANES",                            "Dataset named"),
        ("\\\\begin{algorithm}",              "Algorithm 1 (explanation-guided)"),
        ("\\\\begin{definition}",             "Formal definition present"),
        ("N\\^\\*",                           "N* notation used"),
        ("7.29",                              "Height lie budget value stated"),
        ("8.96",                              "Weight lie budget value stated"),
        ("\\\\bibliographystyle",             "Bibliography style set"),
        ("\\\\bibliography",                  "Bibliography included"),
        ("\\\\appendix",                      "Appendix section present"),
        ("results/figures/fig1_reporting",    "Fig 1 embedded"),
        ("results/figures/fig2_main_result",  "Fig 2 (main result) embedded"),
        ("results/figures/fig8_config_heatmap","Fig 8 (heatmap) embedded"),
        ("table4_main_results",               "Table 4 (main results) embedded"),
        ("table5_statistical_tests",          "Table 5 (stats) embedded"),
    ]
    for pattern, label in content_checks:
        check(label, bool(re.search(pattern, tex, re.IGNORECASE)))

# ----------------------------------------------------------------
# G: [FILL] placeholder count (submission check)
# ----------------------------------------------------------------
print("\n  [G] Submission Readiness")
if os.path.exists(THESIS):
    tex = open(THESIS, encoding="utf-8").read()
    fills  = tex.count("[FILL]")
    todos  = len(re.findall(r'%.*TODO', tex))
    fills_list = [(i+1, line.strip()) for i, line in enumerate(tex.split('\n'))
                  if '[FILL]' in line]

    check("No [FILL] placeholders (submission ready)",
          fills == 0,
          f"{fills} placeholders remain — fill after Stage 7 results",
          warn=fills > 0)
    check("No TODO blocks (all sections written)",
          todos == 0,
          f"{todos} TODO blocks remain — write per roadmap/10_thesis.md",
          warn=todos > 0)

    # Key submission items
    sub_checks = [
        (os.path.exists("docs/analysis_plan.md"), "Pre-registration doc exists"),
        (os.path.exists("paper/references.bib"),  "Shared bibliography exists"),
        (os.path.exists(THESIS),                  "Thesis main.tex exists"),
    ]
    for cond, label in sub_checks:
        check(label, cond)

    if fills_list:
        print(f"\n  [FILL] locations in thesis:")
        for lineno, line in fills_list[:10]:
            print(f"    Line {lineno:4d}: {line[:70]}")
        if len(fills_list) > 10:
            print(f"    ... and {len(fills_list)-10} more")

# ----------------------------------------------------------------
# H: Thesis vs paper consistency
# ----------------------------------------------------------------
print("\n  [H] Thesis-Paper Consistency")
if os.path.exists(THESIS) and os.path.exists("paper/main.tex"):
    thesis_tex = open(THESIS, encoding="utf-8").read()
    paper_tex  = open("paper/main.tex", encoding="utf-8").read()
    # Both should have same key numbers
    for val, thesis_pat, paper_pat in [
        ("7.29",  "7.29",        "7.29"),
        ("8.96",  "8.96",        "8.96"),
        ("1000",  "1{,}000",     "1000"),   # thesis uses LaTeX thousands sep
        ("0.95",  "0.95",        "0.95"),
    ]:
        in_thesis = thesis_pat in thesis_tex
        in_paper  = paper_pat  in paper_tex
        check(f"Value {val} in both thesis and paper",
              in_thesis and in_paper,
              f"thesis={'yes' if in_thesis else 'NO'}, "
              f"paper={'yes' if in_paper else 'NO'}")

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)

if os.path.exists(THESIS):
    tex = open(THESIS, encoding="utf-8").read()
    fills  = tex.count("[FILL]")
    todos  = len(re.findall(r'%.*TODO', tex))
    n_chars = len(tex)
    n_lines = len(tex.split('\n'))
    print(f"\n  Thesis size: {n_lines} lines / {n_chars//1000}K chars")
    print(f"  Chapters:   7 body + 4 appendices")

if issues:
    print(f"\n  {FAIL} {len(issues)} blocking issue(s):")
    for i in issues:
        print(f"    - {i}")
else:
    print(f"\n  {PASS} All structural checks passed!")

if os.path.exists(THESIS):
    if fills > 0 or todos > 0:
        print(f"\n  Pending writing work:")
        if fills > 0:
            print(f"    - {fills} [FILL] placeholders (need final N* results from Stage 7)")
        if todos > 0:
            print(f"    - {todos} TODO blocks (write sections per roadmap/10_thesis.md)")
    else:
        print(f"\n  Thesis is SUBMISSION READY — all placeholders filled.")
        print(f"  Compile with:")
        print(f"    cd thesis && pdflatex main.tex && bibtex main && pdflatex main.tex")
print()
