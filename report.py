"""Tables, figures and text macros of Section 7 of the paper, generated from ``results/``.

``python -m srdssink.report`` writes into ``generated/``:

  tables/pooled.tex, per_suite.tex, failures.tex, indefinite.tex, escape.tex,
  sensitivity.tex, hvp_budget.tex, mlp.tex, mlp_escape.tex     (\\input by the manuscript)
  macros.tex                                                   (every number quoted in prose)
  figures/profile_cost, history_saddle1000, history_classical, history_cuter1000,
  history_cuter10000, history_andrei3000, hvp_budget, sensitivity   (.pdf, .eps and .png)

Conventions: medians are over solved instances (Table 2); bold marks the best entry of a
row (largest number solved, smallest median); costs are in work units; an unsolved run is
"fail".  The performance profile is that of Dolan and More by cost, with unsolved runs at
infinity.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

from .campaign import METHODS, RESULTS_DIR
from .problems import all_instances

_ORDER = {p.name: k for k, p in enumerate(all_instances())}


def _sorted_problems(names):
    return sorted(names, key=lambda n: (_ORDER.get(n, 10**6), n))

OUT_DIR = os.environ.get("SRDSSINK_GENERATED", "generated")
GROUPS = [("classical", "Classical"), ("cuter1000", "CUTEr $n=1000$"),
          ("cuter10000", "CUTEr $n=10000$"), ("andrei", "Andrei $n=3000$")]
SITES = [("third", r"Third-derivative terms, \eqref{eq:matvec_u} and \eqref{eq:matvec_alpha}"),
         ("Hd", r"Operator product $H\mathbf{d}_{\mathbf{u}}$ in GMRES"),
         ("p-solve", r"Inner solve for $\mathbf{p}$"),
         ("damping", r"Residual in the damping test"),
         ("lanczos", r"Lanczos estimate of $\lambda_{\min}$"),
         ("residual", r"Residual at the first inner iterate (no product needed)"),
         ("scale", r"Scale estimate for $\epsilon$")]


# ------------------------------------------------------------------------------ helpers ----


def _load(name: str):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as fh:
        return json.load(fh)


def _group_of(rec: Dict) -> str:
    if rec["group"] == "cuter":
        return "cuter1000" if rec["n"] == 1000 else "cuter10000"
    return rec["group"]


def _solved(rec: Dict) -> bool:
    return rec["status"] == "converged"


def _median(values: List[float]):
    return float(np.median(values)) if values else None


def _rint(v) -> int:
    """Round half up, so that the tables and the prose quote the same integer."""
    return int(np.floor(float(v) + 0.5))


def _fmt_int(v) -> str:
    return "---" if v is None else f"${_rint(v)}$"


def _fmt_sci(v: float, digits: int = 2) -> str:
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    if float(f"{abs(m):.{digits}f}") >= 10.0:      # rounding carried the mantissa to 10
        e += 1
        m = v / 10 ** e
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def _fmt_sci_signed(v: float, digits: int = 3) -> str:
    s = _fmt_sci(abs(v), digits)
    return s.replace("$", "$+", 1) if v >= 0 else s.replace("$", "$-", 1)


def _bold(s: str) -> str:
    return s.replace("$", "$\\mathbf{", 1)[:-1] + "}$" if s.startswith("$") else f"\\textbf{{{s}}}"


def _row(label: str, cells: List[str]) -> str:
    return label + " & " + " & ".join(cells) + r" \\"


def _macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _num(v: float) -> str:
    """Plain number for prose: integers without decimals, else two decimals."""
    if v is None:
        return "---"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.2f}"


def _sci_prose(v: float, digits: int = 1) -> str:
    """Scientific notation without the enclosing dollars, for use inside math in the prose."""
    return _fmt_sci(v, digits).strip("$")


# ------------------------------------------------------------------------------- tables ----


def table_pooled(main: List[Dict], n_inst: int, out: List[str], macros: List[str]) -> Dict:
    rows = {"solved": {}, "NI": {}, "NFG": {}, "inner": {}, "cost": {}}
    for m in METHODS:
        runs = [r for r in main if r["method"] == m]
        solved = [r for r in runs if _solved(r)]
        rows["solved"][m] = len(solved)
        rows["NI"][m] = _median([r["outer"] for r in solved])
        rows["NFG"][m] = _median([r["nfg"] for r in solved])
        rows["inner"][m] = _median([r["n_hvp"] for r in solved])
        rows["cost"][m] = _median([r["cost"] for r in solved])
    lines = []
    best = max(rows["solved"].values())
    lines.append(_row("solved", [_bold(f"${rows['solved'][m]}/{n_inst}$") if rows["solved"][m] == best
                                 else f"${rows['solved'][m]}/{n_inst}$" for m in METHODS]))
    for key, label in (("NI", "median NI"), ("NFG", "median NFG"), ("inner", "median inner"),
                       ("cost", "median cost")):
        vals = rows[key]
        finite = [v for v in vals.values() if v is not None]
        bmin = min(finite) if finite else None
        cells = []
        for m in METHODS:
            v = vals[m]
            s = _fmt_int(v)
            if v is not None and bmin is not None and abs(v - bmin) < 0.5:
                s = _bold(s)
            cells.append(s)
        lines.append(_row(label, cells))
    out.append("\n".join(lines))
    for m in METHODS:
        tag = m.replace("-", "").replace("SSINK2", "SSINKtwo").replace("LBFGS", "LBFGS")
        macros.append(_macro(f"solved{tag}", rows["solved"][m]))
        for key, label in (("NI", "NI"), ("NFG", "NFG"), ("inner", "Inner"), ("cost", "Cost")):
            v = rows[key][m]
            macros.append(_macro(f"median{label}{tag}", "---" if v is None else _rint(v)))
    others = [rows["solved"][m] for m in ("ARC", "TR-NCG", "RegN", "L-BFGS")]
    macros.append(_macro("solvedOthersMin", min(others)))
    macros.append(_macro("solvedOthersMax", max(others)))
    rank = sorted(rows["solved"].values(), reverse=True)
    macros.append(_macro("solvedRankSSINK", 1 + sum(1 for v in rows["solved"].values() if v > rows["solved"]["SS-INK"])))
    macros.append(_macro("solvedFewerThanSSINK", ", ".join(m for m in METHODS if rows["solved"][m] < rows["solved"]["SS-INK"]) or "none"))
    macros.append(_macro("solvedFewerThanSSINKtwo", ", ".join(m for m in METHODS if rows["solved"][m] < rows["solved"]["SS-INK-2"]) or "none"))
    costs_sorted = sorted((v for v in rows["cost"].values() if v is not None), reverse=True)
    for m, tag in (("SS-INK", "SSINK"), ("SS-INK-2", "SSINKtwo")):
        v = rows["cost"][m]
        macros.append(_macro(f"costRank{tag}", 1 + sum(1 for c in costs_sorted if c > v) if v is not None else "---"))
    lb = rows["cost"]["L-BFGS"]
    for m, tag in (("SS-INK", "SSINK"), ("SS-INK-2", "SSINKtwo")):
        ratio = rows["cost"][m] / lb if (lb and rows["cost"][m]) else None
        macros.append(_macro(f"costRatio{tag}LBFGS", _rint(ratio) if ratio else "---"))
    return rows


def table_per_suite(main: List[Dict], out: List[str], macros: List[str]) -> None:
    lines = []
    for gkey, glabel in GROUPS:
        runs = [r for r in main if _group_of(r) == gkey]
        total = len({r["problem"] for r in runs})
        cells = []
        for m in METHODS:
            k = sum(1 for r in runs if r["method"] == m and _solved(r))
            cells.append(f"${k}/{total}$")
        lines.append(_row(f"{glabel} (${total}$)", cells))
        macros.append(_macro("nInst" + {"classical": "classical", "cuter1000": "cuterK",
                                        "cuter10000": "cuterTenK", "andrei": "andrei"}[gkey], total))
    out.append("\n".join(lines))
    macros.append(_macro("nInstcuter", len({r["problem"] for r in main if r["group"] == "cuter"})))


def table_failures(main: List[Dict], starts: Dict, out: List[str], macros: List[str]) -> None:
    fails = [r for r in main if r["method"] == "SS-INK" and not _solved(r)]
    fails.sort(key=lambda r: r["problem"])
    lines = []
    n_indef = 0
    for r in fails:
        lam = starts[r["problem"]]["lambda_min_0"]
        if lam < 0:
            n_indef += 1
        lines.append(_row(r["problem"].replace("_", "-"),
                          [_fmt_sci_signed(lam), _fmt_sci(starts[r["problem"]]["grad_norm_0"]),
                           _fmt_sci(r["grad_norm_exit"])]))
    out.append("\n".join(lines) if lines else r"(none) & & & \\")
    nostep = [r for r in fails if r["outer"] == 0]
    macros.append(_macro("failNoStepSSINK", ", ".join(r["problem"] for r in nostep) or "none"))
    macros.append(_macro("nFailNoStepSSINK", len(nostep)))
    for r in nostep:
        st = starts.get(r["problem"], {})
        if "lanczos_start_hvp" in st:
            macros.append(_macro("lanczosStartHVP" + _name_tag(r["problem"]), st["lanczos_start_hvp"]))
    fails2 = [r for r in main if r["method"] == "SS-INK-2" and not _solved(r)]
    macros.append(_macro("nFailSSINKtwo", len(fails2)))
    macros.append(_macro("failListSSINKtwo", ", ".join(sorted(r["problem"] for r in fails2)) or "none"))
    macros.append(_macro("failNoStepSSINKtwo", ", ".join(sorted(r["problem"] for r in fails2 if r["outer"] == 0)) or "none"))
    macros.append(_macro("nFailSSINK", len(fails)))
    macros.append(_macro("nFailSSINKindef", n_indef))
    macros.append(_macro("failListSSINK", ", ".join(r["problem"] for r in fails) if fails else "none"))
    # CURLY10 universal-failure statement
    for n in (1000, 10000):
        runs = [r for r in main if r["problem"] == f"CURLY10-{n}"]
        macros.append(_macro(f"curlySolvedCount{'K' if n == 1000 else 'TenK'}",
                             sum(1 for r in runs if _solved(r))))
        sol = [r["method"] for r in runs if _solved(r)]
        macros.append(_macro(f"curlySolvedList{'K' if n == 1000 else 'TenK'}",
                             ", ".join(sol) if sol else "none of the nine methods"))
    tri = [r for r in main if r["problem"] == "TRIDIA-3000" and r["method"] == "SS-INK"]
    if tri:
        macros.append(_macro("tridiaSSINKstatus", "solves" if _solved(tri[0]) else "does not solve"))
        macros.append(_macro("tridiaGradExit", _sci_prose(tri[0]["grad_norm_exit"], 2)))
        macros.append(_macro("tridiaGradStart", _sci_prose(starts["TRIDIA-3000"]["grad_norm_0"], 2)))
        macros.append(_macro("tridiaLamMin", f"{starts['TRIDIA-3000']['lambda_min_0']:.3f}"))
        eps = tri[0].get("eps")
        if eps is not None:
            lam0 = starts["TRIDIA-3000"]["lambda_min_0"]
            macros.append(_macro("tridiaHnorm", _sci_prose(eps / 1e-2, 2)))
            macros.append(_macro("tridiaEps", _sci_prose(eps, 2)))
            macros.append(_macro("tridiaAlphaZero", _sci_prose(2.0 * max(0.0, eps - lam0), 2)))
            macros.append(_macro("tridiaAlphaOverLam", _sci_prose(2.0 * max(0.0, eps - lam0) / lam0, 1)))


def table_indefinite(main: List[Dict], starts: Dict, out: List[str], macros: List[str]) -> None:
    indef = sorted(name for name, s in starts.items() if s["lambda_min_0"] < 0)
    runs = [r for r in main if r["problem"] in indef]
    solved_cells, cost_cells = [], []
    solved_counts, costs = {}, {}
    for m in METHODS:
        rs = [r for r in runs if r["method"] == m]
        sol = [r for r in rs if _solved(r)]
        solved_counts[m] = len(sol)
        costs[m] = _median([r["cost"] for r in sol])
    best = max(solved_counts.values())
    for m in METHODS:
        s = f"${solved_counts[m]}/{len(indef)}$"
        solved_cells.append(_bold(s) if solved_counts[m] == best else s)
    finite = [v for v in costs.values() if v is not None]
    bmin = min(finite) if finite else None
    for m in METHODS:
        v = costs[m]
        s = _fmt_int(v)
        if v is not None and abs(v - bmin) < 0.5:
            s = _bold(s)
        cost_cells.append(s)
    out.append(_row("solved", solved_cells) + "\n" + _row("median cost", cost_cells))
    macros.append(_macro("nIndef", len(indef)))
    macros.append(_macro("indefList", ", ".join(indef)))
    for m, tag in (("SS-INK", "SSINK"), ("SS-INK-2", "SSINKtwo")):
        macros.append(_macro(f"indefSolved{tag}", solved_counts[m]))
        macros.append(_macro(f"indefSolvedList{tag}", ", ".join(sorted(r["problem"] for r in runs if r["method"] == m and _solved(r))) or "none"))
        lb = costs["L-BFGS"]
        ratio = costs[m] / lb if (lb and costs[m]) else None
        macros.append(_macro(f"indefCostRatio{tag}LBFGS", _rint(ratio) if ratio else "---"))
        macros.append(_macro(f"indefMedianCost{tag}", "---" if costs[m] is None else _rint(costs[m])))
    others = [solved_counts[m] for m in METHODS if m not in ("SS-INK", "SS-INK-2") and solved_counts[m] > 0]
    macros.append(_macro("indefSolvedOthersMin", min(others) if others else "---"))


TOL_REL = 1e-6         # relative gradient tolerance of Section 7.1 (default of every solver)
ESCAPE_TOL = 1e-4      # f <= -n + ESCAPE_TOL counts as "reaches f = -n" (Table escape caption)


def table_escape(esc: List[Dict], out: List[str], macros: List[str]) -> None:
    lines = []
    for n in (100, 1000):
        cells = []
        for m in METHODS:
            rs = [r for r in esc if r["problem"] == f"QUARTIC-{n}" and r["method"] == m]
            cells.append(f"${rs[0]['f_exit']:.1f}$" if rs else "---")
        lines.append(_row(f"$n = {n}$", cells))
    out.append("\n".join(lines))
    for n in (100, 1000):
        for m in ("SS-INK", "SS-INK-2", "ARC", "TR-NCG", "PGD"):
            rs = [r for r in esc if r["problem"] == f"QUARTIC-{n}" and r["method"] == m]
            if rs:
                tag = m.replace("-", "").replace("SSINK2", "SSINKtwo")
                macros.append(_macro(f"escape{tag}{'H' if n == 100 else 'K'}", f"{rs[0]['f_exit']:.1f}"))
    # cost at which ARC and TR-NCG reach f = -n (first history entry with f <= -n + 1e-6)
    reach = []
    for n in (100, 1000):
        for m in ("ARC", "TR-NCG"):
            rs = [r for r in esc if r["problem"] == f"QUARTIC-{n}" and r["method"] == m]
            if rs:
                hist = [(c, f) for c, g, f in rs[0]["history"] if f is not None and f <= -n + 1e-6]
                if hist:
                    reach.append(hist[0][0])
    macros.append(_macro("escapeReachMin", _num(min(reach)) if reach else "---"))
    macros.append(_macro("escapeReachMax", _num(max(reach)) if reach else "---"))
    macros.append(_macro("escapeReachCount", len(reach)))
    # which methods reach -n (within ESCAPE_TOL, stated in the caption of Table escape)
    for n in (100, 1000):
        tag = 'H' if n == 100 else 'K'
        rs_n = [r for r in esc if r["problem"] == f"QUARTIC-{n}"]
        reached = [r["method"] for r in rs_n if r["f_exit"] <= -n + ESCAPE_TOL]
        not_reached = [m for m in METHODS if m not in reached]
        macros.append(_macro(f"escapeNotReached{tag}", ", ".join(not_reached) if not_reached else "none"))
        macros.append(_macro(f"escapeReached{tag}", ", ".join(reached) if reached else "none"))
        macros.append(_macro(f"escapeReachedCount{tag}", sum(1 for m in reached if m not in ("SS-INK", "SS-INK-2"))))
        if rs_n:
            least = max(rs_n, key=lambda r: r["f_exit"])
            macros.append(_macro(f"escapeLeast{tag}", least["method"]))
            macros.append(_macro(f"escapeLeastValue{tag}", f"{least['f_exit']:.1f}"))
            ncg = [r for r in rs_n if r["method"] == "NCG"]
            if ncg and len(ncg[0]["history"]) > 2:
                h = ncg[0]["history"]
                macros.append(_macro(f"ncgEscapeFirstF{tag}", f"{h[1][2]:.4f}"))
                per = (h[1][2] - h[-1][2]) / max(1, ncg[0]["outer"] - 1)
                macros.append(_macro(f"ncgEscapePerIter{tag}", _sci_prose(per, 1)))
                macros.append(_macro(f"ncgEscapeIters{tag}", max(0, ncg[0]["outer"] - 1)))
        for m in ("SS-INK", "SS-INK-2", "NCG"):
            rs = [r for r in rs_n if r["method"] == m]
            if rs:
                macros.append(_macro(f"escapeOuter{m.replace('-', '').replace('SSINK2', 'SSINKtwo')}{tag}", rs[0]["outer"]))


def table_sensitivity(sens: List[Dict], out: List[str], macros: List[str]) -> None:
    def cell(rs):
        if not rs:
            return "---"
        r = rs[0]
        return _fmt_int(r["cost"]) if _solved(r) else "fail"
    etas = [0.9, 0.5, 0.1, 1e-3, 1e-6, 1e-10]
    ratios = [1.2, 2.0, 5.0, 20.0]
    eta_cells, ratio_cells = [], []
    eta_costs = []
    for e in etas:
        rs = [r for r in sens if r["params"].get("eta_max") == e]
        eta_cells.append(cell(rs))
        if rs and _solved(rs[0]):
            eta_costs.append(rs[0]["cost"])
    for q in ratios:
        rs = [r for r in sens if r["params"].get("K") == q]
        ratio_cells.append(cell(rs))
    lines = [_row(r"$\eta_{\max}$", ["$0.9$", "$0.5$", "$0.1$", "$10^{-3}$", "$10^{-6}$", "$10^{-10}$"]),
             _row("cost", eta_cells), r"\midrule",
             _row(r"$K/\gamma$", ["$1.2$", "$2$", "$5$", "$20$", "", ""]),
             _row("cost", ratio_cells + ["", ""])]
    out.append("\n".join(lines))
    if eta_costs:
        spread = (max(eta_costs) - min(eta_costs)) / min(eta_costs) * 100
        macros.append(_macro("sensEtaSpreadPct", _rint(spread)))
    macros.append(_macro("sensEtaAllSolved", "yes" if len(eta_costs) == len(etas) else "no"))
    def _eta_tex(e):
        return f"${e:g}$" if e >= 0.1 else f"$10^{{{int(round(np.log10(e)))}}}$"
    attained = []
    for r in sens:
        if "eta_max" in r["params"]:
            attained.append((r["params"]["eta_max"], r.get("forcing_missed", 0) == 0, r.get("forcing_missed", 0), r.get("linear_solves", 0)))
    attained.sort(key=lambda t: -t[0])
    macros.append(_macro("sensEtaAttainedList", ", ".join(_eta_tex(e) for e, a, _, _ in attained if a) or "none"))
    macros.append(_macro("sensEtaMissedList", ", ".join(_eta_tex(e) for e, a, _, _ in attained if not a) or "none"))
    macros.append(_macro("sensEtaMissedCounts", ", ".join(f"{m} of {l}" for e, a, m, l in attained if not a) or "none"))
    macros.append(_macro("sensEtaCountAttained", sum(1 for t in attained if t[1])))
    n_att = sum(1 for t in attained if t[1])
    if n_att == len(attained):
        sent = "In every one of the six runs every forcing term was attained"
    elif n_att == 0:
        sent = ("In none of the six runs was every forcing term attained: the cap was binding in " +
                ", ".join(f"{m} of {l}" for e, a, m, l in attained) + " linear solves for $\\eta_{\\max} = " +
                ", ".join(_eta_tex(e).strip("$") for e, a, m, l in attained) + "$ respectively, so the experiment does not isolate the forcing rule")
    else:
        sent = ("Every forcing term was attained for $\\eta_{\\max} \\in \\{" + ", ".join(_eta_tex(e).strip("$") for e, a, _, _ in attained if a) +
                "\\}$, while for $\\eta_{\\max} \\in \\{" + ", ".join(_eta_tex(e).strip("$") for e, a, _, _ in attained if not a) +
                "\\}$ the cap was binding in " + ", ".join(f"{m} of {l}" for e, a, m, l in attained if not a) + " linear solves respectively")
    macros.append(_macro("sensEtaAttainedSentence", sent))
    # K/gamma statements
    def cost_of(q):
        rs = [r for r in sens if r["params"].get("K") == q]
        return rs[0]["cost"] if rs and _solved(rs[0]) else None
    c12, c2, c5, c20 = cost_of(1.2), cost_of(2.0), cost_of(5.0), cost_of(20.0)
    solved_q = [(q, c) for q, c in zip(ratios, (c12, c2, c5, c20)) if c is not None]
    macros.append(_macro("sensRatioLowest", f"{min(solved_q, key=lambda t: t[1])[0]:g}" if solved_q else "---"))
    macros.append(_macro("sensRatioHighest", f"{max(solved_q, key=lambda t: t[1])[0]:g}" if solved_q else "---"))
    macros.append(_macro("sensRatioFailed", ", ".join(f"{q:g}" for q, c in zip(ratios, (c12, c2, c5, c20)) if c is None) or "none"))
    macros.append(_macro("sensRatioSpreadFactor", f"{max(c for _, c in solved_q) / min(c for _, c in solved_q):.1f}" if solved_q else "---"))
    macros.append(_macro("sensRatioOnePointTwoFactor", f"{c12 / c2:.1f}" if (c12 and c2) else "---"))
    macros.append(_macro("sensRatioFiveFactor", f"{c5 / c2:.1f}" if (c5 and c2) else "---"))
    macros.append(_macro("sensRatioTwentyStatus", "fail" if c20 is None else _num(c20)))
    macros.append(_macro("sensBaseCost", _num(c2) if c2 else "---"))
    macros.append(_macro("sensRatioTwentyFactor", f"{c20 / c2:.1f}" if (c20 and c2) else "---"))
    for q, c, tag in ((1.2, c12, "OnePointTwo"), (2.0, c2, "Two"), (5.0, c5, "Five"), (20.0, c20, "Twenty")):
        macros.append(_macro(f"sensCost{tag}", _num(c) if c else "fail"))
    order = sorted(solved_q, key=lambda t: t[1])
    macros.append(_macro("sensRatioOrder", " < ".join(f"{q:g}" for q, _ in order) if order else "---"))


def table_hvp_budget(main: List[Dict], out: List[str], macros: List[str]) -> None:
    lines = []
    totals = {}
    shares = {}
    for m, tag in (("SS-INK", "SSINK"), ("SS-INK-2", "SSINKtwo")):
        runs = [r for r in main if r["method"] == m]
        pooled = {}
        for r in runs:
            for k, v in r["hvp_by_site"].items():
                pooled[k] = pooled.get(k, 0) + v
        tot = sum(pooled.values())
        totals[m] = tot
        shares[m] = {k: 100.0 * pooled.get(k, 0) / tot for k, _ in SITES}
        macros.append(_macro(f"hvpTotal{tag}", _sci_prose(tot, 2)))
        # per-run share of the third-derivative terms, solved and unsolved
        def share(r):
            t = sum(r["hvp_by_site"].values())
            return 100.0 * r["hvp_by_site"].get("third", 0) / t if t else 0.0
        macros.append(_macro(f"thirdShareSolved{tag}", _rint(_median([share(r) for r in runs if _solved(r)]) or 0)))
        macros.append(_macro(f"thirdShareUnsolved{tag}", _rint(_median([share(r) for r in runs if not _solved(r)]) or 0)))
        hv_per_step = [r["n_hvp"] / r["outer"] for r in runs if _solved(r) and r["outer"] > 0]
        macros.append(_macro(f"hvpPerStep{tag}", _rint(_median(hv_per_step) or 0)))
        largest = max(shares[m].items(), key=lambda kv: kv[1])
        macros.append(_macro(f"largestSite{tag}", dict(SITES)[largest[0]]))
        macros.append(_macro(f"largestShare{tag}", f"{largest[1]:.1f}"))
    for key, label in SITES:
        cells = []
        for m in ("SS-INK", "SS-INK-2"):
            s = shares[m][key]
            if key == "third" and m == "SS-INK-2":
                cells.append("---")
            elif s < 0.05:
                cells.append("$<0.1\\%$")
            else:
                cells.append(f"${s:.1f}\\%$")
        lines.append(_row(label, cells))
    out.append("\n".join(lines))
    for key, _ in SITES:
        macros.append(_macro(f"share{key.replace('-', '').capitalize()}SSINK", f"{shares['SS-INK'][key]:.1f}"))
        macros.append(_macro(f"share{key.replace('-', '').capitalize()}SSINKtwo", f"{shares['SS-INK-2'][key]:.1f}"))


def table_mlp(mlp: List[Dict], out: List[str], macros: List[str]) -> None:
    lines = []
    sizes = sorted({r["n"] for r in mlp if r["kind"] == "mlp"})
    for n in sizes:
        cells = []
        for m in METHODS:
            rs = [r for r in mlp if r["kind"] == "mlp" and r["n"] == n and r["method"] == m]
            cells.append("---" if not rs else (_fmt_int(rs[0]["cost"]) if _solved(rs[0]) else "fail"))
        lines.append(_row(f"$n = {n}$", cells))
    out.append("\n".join(lines))
    # statements of Section 7.9
    for n, tag in zip(sizes, ("A", "B")):
        ss = [r for r in mlp if r["kind"] == "mlp" and r["n"] == n and r["method"] == "SS-INK"]
        ss2 = [r for r in mlp if r["kind"] == "mlp" and r["n"] == n and r["method"] == "SS-INK-2"]
        macros.append(_macro(f"mlpN{tag}", n))
        macros.append(_macro(f"mlpSSINKstatus{tag}", "fails" if ss and not _solved(ss[0]) else "succeeds"))
        macros.append(_macro(f"mlpSSINKtwostatus{tag}", "fails" if ss2 and not _solved(ss2[0]) else "succeeds"))
        if ss:
            macros.append(_macro(f"mlpDtExit{tag}", f"{ss[0]['dt']:.2g}"))
            if "lambda_min_exit" in ss[0]:
                macros.append(_macro(f"mlpLamExit{tag}", _sci_prose(ss[0]["lambda_min_exit"], 2)))
        base = [r for r in mlp if r["kind"] == "mlp" and r["n"] == n and r["method"] not in ("SS-INK", "SS-INK-2")]
        succ = [r["method"] for r in base if _solved(r)]
        macros.append(_macro(f"mlpBaselinesSolved{tag}", len(succ)))
        macros.append(_macro(f"mlpBaselinesFailed{tag}", ", ".join(r["method"] for r in base if not _solved(r)) or "none"))
        neg = [(r["method"], r["lambda_min_exit"]) for r in base if _solved(r) and r.get("lambda_min_exit", 1) < 0]
        macros.append(_macro(f"mlpNegExit{tag}", "; ".join(f"{m} at $\\lambda_{{\\min}} = {_sci_prose(l, 1)}$" for m, l in neg) or "none"))
        macros.append(_macro(f"mlpNegExitSentence{tag}", ("is positive for all of them" if not neg else
                             "is positive except for " + "; ".join(f"{m}, which stops at a point with $\\lambda_{{\\min}} = {_sci_prose(l, 1)}$" for m, l in neg)
                             + " (the gradient test cannot distinguish such a point from a minimizer)")))
    st = {}
    for n in sizes:
        for m in ("SS-INK", "SS-INK-2"):
            rr = [r for r in mlp if r["kind"] == "mlp" and r["n"] == n and r["method"] == m]
            if rr:
                st[(m, n)] = _solved(rr[0])
    if len(st) == 2 * len(sizes) and sizes:
        if not any(st.values()):
            macros.append(_macro("mlpVerdict", "both variants fail within the budget at both sizes"))
            macros.append(_macro("mlpSSINKVerdict", "fails at both sizes"))
        else:
            def desc(m):
                sol = [str(n) for n in sizes if st[(m, n)]]
                return f"solves it at $n = {', '.join(sol)}$ only" if 0 < len(sol) < len(sizes) else ("solves it at both sizes" if sol else "fails at both sizes")
            macros.append(_macro("mlpVerdict", "SS-INK " + desc("SS-INK") + " and SS-INK-2 " + desc("SS-INK-2")))
            macros.append(_macro("mlpSSINKVerdict", desc("SS-INK")))
    trunc = [r for r in mlp if r["kind"] == "mlp_trunc"]
    macros.append(_macro("mlpTruncStatus", "removed" if any(_solved(r) for r in trunc) else "not removed"))
    eps_runs = [r for r in mlp if r["kind"] == "mlp_eps"]
    macros.append(_macro("mlpEpsStatus", "removed" if any(_solved(r) for r in eps_runs) else "not removed"))


def table_mlp_escape(esc: List[Dict], out: List[str], macros: List[str]) -> None:
    lines = []
    sizes = sorted({r["n"] for r in esc})
    for n in sizes:
        cells = []
        for m in METHODS:
            rs = [r for r in esc if r["n"] == n and r["method"] == m]
            if rs:
                dec = rs[0]["f_saddle"] - rs[0]["f_exit"]
                cells.append(f"${dec:.3f}$" if abs(dec) >= 0.01 else _fmt_sci(dec, 1))
            else:
                cells.append("---")
        lines.append(_row(f"$n = {n}$", cells))
    out.append("\n".join(lines))
    for n, tag in zip(sizes, ("A", "B")):
        rs = {r["method"]: r["f_saddle"] - r["f_exit"] for r in esc if r["n"] == n}
        least = min(rs.items(), key=lambda kv: kv[1])[0] if rs else "---"
        macros.append(_macro(f"mlpEscapeLeast{tag}", least))
        fs = next(iter(abs(r["f_saddle"]) for r in esc if r["n"] == n), 1.0)
        thr = 1e-6 * max(1.0, fs)               # an escape is a decrease beyond roundoff
        macros.append(_macro(f"mlpEscapeAll{tag}", "all nine methods" if all(v > thr for v in rs.values()) else
                             "the methods " + ", ".join(m for m, v in rs.items() if v > thr)))
        macros.append(_macro(f"mlpEscapeNone{tag}", ", ".join(m for m, v in rs.items() if v <= thr) or "none"))


# ------------------------------------------------------------------------------ figures ----
#
# All figures are vector graphics (PDF for pdflatex, EPS for latex+dvips) with embedded Type 42
# fonts, sized for a single column of the manuscript (width 3.35 in) or its full text width
# (6.5 in).  The eleven methods are drawn with the Okabe--Ito colour-blind-safe palette combined
# with distinct line styles and markers, so that every curve is identifiable in greyscale; the
# two SS-INK variants are black (solid and dashed).  A PNG preview at 300 dpi is written too.


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.0, "lines.markersize": 3.0, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "grid.color": "0.88", "grid.linewidth": 0.5, "legend.framealpha": 1.0,
        "legend.edgecolor": "0.6", "legend.handlelength": 2.2, "ps.fonttype": 42,
        "pdf.fonttype": 42, "savefig.dpi": 300})
    return plt


# Okabe--Ito palette (colour-blind safe); black is reserved for the SS-INK variants.
_OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
       "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
STYLES = {
    "SS-INK":   ("#000000", "-",  "o"),
    "SS-INK-2": ("#000000", "--", "s"),
    "ARC":      (_OI["blue"], "-", "^"),
    "TR-NCG":   (_OI["vermilion"], "-", "v"),
    "NCG":      (_OI["green"], "-", "D"),
    "RegN":     (_OI["purple"], "-", "P"),
    "L-BFGS":   (_OI["orange"], "-", "X"),
    "Adam":     (_OI["sky"], "-", "<"),
    "PGD":      (_OI["grey"], "-", ">"),
    "TN":       (_OI["blue"], "-.", "h"),
    "INB":      (_OI["vermilion"], "-.", "p"),
}
COLUMN_WIDTH, TEXT_WIDTH = 3.35, 6.5


def _save(fig, name: str) -> None:
    d = os.path.join(OUT_DIR, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), format=ext, bbox_inches="tight", pad_inches=0.02)


def _plot(ax, x, y, method: str, markevery=None, **kw):
    color, ls, marker = STYLES[method]
    ax.plot(x, y, color=color, linestyle=ls, marker=marker, markevery=markevery,
            markerfacecolor="white", markeredgewidth=0.7, label=method, **kw)


def _method_legend(fig_or_ax, ncol: int, **kw):
    handles, labels = [], []
    for m in METHODS:
        color, ls, marker = STYLES[m]
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color=color, linestyle=ls, marker=marker,
                              markerfacecolor="white", markeredgewidth=0.7))
        labels.append(m)
    return fig_or_ax.legend(handles, labels, ncol=ncol, **kw)


def figure_profile(main: List[Dict], macros: List[str]) -> None:
    """Dolan--Mor\'e performance profile by cost; unsolved runs are placed at infinity."""
    plt = _mpl()
    problems = sorted({r["problem"] for r in main})
    cost = {(r["problem"], r["method"]): (r["cost"] if _solved(r) else np.inf) for r in main}
    ratios = {m: [] for m in METHODS}
    for p in problems:
        best = min(cost.get((p, m), np.inf) for m in METHODS)
        for m in METHODS:
            c = cost.get((p, m), np.inf)
            ratios[m].append(c / best if np.isfinite(c) and np.isfinite(best) else np.inf)
    finite = [r for m in METHODS for r in ratios[m] if np.isfinite(r)]
    tmax = max(finite) * 1.05 if finite else 10.0
    taus = np.logspace(0, np.log10(tmax), 600)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.35, 2.9))
    for m in METHODS:
        rho = [np.mean([r <= t for r in ratios[m]]) for t in taus]
        color, ls, marker = STYLES[m]
        ax.step(taus, rho, where="post", color=color, linestyle=ls, label=m)
        ax.plot(taus[::75], np.array(rho)[::75], color=color, linestyle="none", marker=marker,
                markerfacecolor="white", markeredgewidth=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, tmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"cost ratio to the best method, $\tau$")
    ax.set_ylabel(r"fraction of instances solved within $\tau$")
    ax.grid(True)
    _method_legend(ax, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False,
                   handletextpad=0.5, columnspacing=0.9)
    _save(fig, "profile_cost")
    plt.close(fig)


def _history_panel(ax, runs: List[Dict], title: str) -> None:
    for m in METHODS:
        rs = [r for r in runs if r["method"] == m]
        if not rs:
            continue
        h = np.array([(c, g) for c, g, f in rs[0]["history"]], dtype=float)
        x, y = np.maximum(h[:, 0], 1.0), np.maximum(h[:, 1], 1e-300)
        _plot(ax, x, y, m, markevery=max(1, len(x) // 6))
    # solve tolerance of Section 7.1 from the first history point (common starting point)
    g0 = float(runs[0]["history"][0][1])
    ax.axhline(TOL_REL * max(1.0, g0), color="0.3", linestyle=":", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if title:
        ax.set_title(title, pad=2)
    ax.grid(True)


def figure_histories(main: List[Dict]) -> None:
    plt = _mpl()
    runs = [r for r in main if r["problem"] == "QUARTIC-1000"]
    if runs:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.35, 2.9))
        _history_panel(ax, runs, "")
        ax.set_xlabel("cost (work units)")
        ax.set_ylabel(r"$\|\nabla f\|_2$")
        _method_legend(ax, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False,
                       handletextpad=0.5, columnspacing=0.9)
        _save(fig, "history_saddle1000")
        plt.close(fig)
    for gkey, fname in (("classical", "history_classical"), ("cuter1000", "history_cuter1000"),
                        ("cuter10000", "history_cuter10000"), ("andrei", "history_andrei3000")):
        problems = _sorted_problems({r["problem"] for r in main if _group_of(r) == gkey})
        if not problems:
            continue
        ncol = 4 if len(problems) > 6 else 3
        nrow = int(np.ceil(len(problems) / ncol))
        legend_in = 0.42                                    # height reserved for the legend
        height = 1.75 * nrow + legend_in
        fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_WIDTH, height), squeeze=False)
        for k, p in enumerate(problems):
            ax = axes[k // ncol][k % ncol]
            _history_panel(ax, [r for r in main if r["problem"] == p], p)
            if k // ncol == nrow - 1:
                ax.set_xlabel("cost (work units)")
            if k % ncol == 0:
                ax.set_ylabel(r"$\|\nabla f\|_2$")
        for k in range(len(problems), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        _method_legend(fig, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False,
                       handletextpad=0.5, columnspacing=0.9)
        fig.tight_layout(rect=(0, legend_in / height, 1, 1))
        _save(fig, fname)
        plt.close(fig)


def figure_hvp_budget(main: List[Dict]) -> None:
    """Share of the Hessian-vector products by call site (companion of Table hvp_budget)."""
    plt = _mpl()
    variants = ["SS-INK", "SS-INK-2"]
    sites = [k for k, _ in SITES]
    labels = {"third": "third-derivative terms", "Hd": r"operator product $H\mathbf{d}_{\mathbf{u}}$",
              "p-solve": r"solve for $\mathbf{p}$", "damping": "damping-test residuals",
              "lanczos": r"Lanczos estimate of $\lambda_{\min}$", "residual": "residual at the first iterate",
              "scale": r"scale estimate for $\epsilon$"}
    shares = {}
    for m in variants:
        tot = {}
        for r in main:
            if r["method"] == m:
                for k, v in r.get("hvp_by_site", {}).items():
                    tot[k] = tot.get(k, 0) + v
        total = sum(tot.values()) or 1
        shares[m] = [100.0 * tot.get(k, 0) / total for k in sites]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.35, 1.6))
    greys = [str(0.15 + 0.11 * k) for k in range(len(sites))]
    left = np.zeros(len(variants))
    for k, site in enumerate(sites):
        vals = np.array([shares[m][k] for m in variants])
        ax.barh(variants, vals, left=left, color=greys[k], edgecolor="black", linewidth=0.4,
                label=labels[site], height=0.55)
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of Hessian-vector products (%)")
    ax.invert_yaxis()
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.45), frameon=False)
    _save(fig, "hvp_budget")
    plt.close(fig)


def figure_sensitivity(sens: List[Dict]) -> None:
    """Cost of SS-INK against the forcing cap and the gain ratio (companion of Table sensitivity)."""
    plt = _mpl()
    etas = [0.9, 0.5, 0.1, 1e-3, 1e-6, 1e-10]
    ratios = [1.2, 2.0, 5.0, 20.0]

    def cost(key, value):
        rs = [r for r in sens if r["params"].get(key) == value]
        return (rs[0]["cost"] if _solved(rs[0]) else np.nan) if rs else np.nan
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.8, 2.2))
    c_eta = [cost("eta_max", e) for e in etas]
    a1.plot(etas, c_eta, color="black", marker="o", markerfacecolor="white", markeredgewidth=0.7)
    a1.set_xscale("log")
    a1.invert_xaxis()
    a1.set_xlabel(r"forcing cap $\eta_{\max}$")
    a1.set_ylabel("cost (work units)")
    a1.grid(True)
    c_ratio = [cost("K", q) for q in ratios]
    a2.plot(ratios, c_ratio, color="black", marker="s", markerfacecolor="white", markeredgewidth=0.7)
    a2.set_xscale("log")
    a2.set_xticks(ratios)
    a2.set_xticklabels([f"{q:g}" for q in ratios])
    a2.set_xlabel(r"gain ratio $K/\gamma$")
    a2.grid(True)
    for ax, vals in ((a1, c_eta), (a2, c_ratio)):
        if any(np.isnan(v) for v in vals):
            ax.text(0.5, 0.05, "missing points: budget exhausted", transform=ax.transAxes,
                    ha="center", fontsize=6.5)
    fig.tight_layout()
    _save(fig, "sensitivity")
    plt.close(fig)


# ------------------------------------------------------------------------------ prose ----


def prose_macros(main: List[Dict], cosine: List[Dict], starts: Dict, saddle: Dict, macros: List[str]) -> None:
    probs = sorted({r["problem"] for r in main})
    macros.append(_macro("nInstances", len(probs)))
    by = {(r["problem"], r["method"]): r for r in main}
    missing = [(p, m) for p in probs for m in METHODS if (p, m) not in by]
    if missing:
        print(f"warning: {len(missing)} (problem, method) pairs missing from main.json; "
              "prose macros computed on the complete instances only", file=sys.stderr)
        probs = [p for p in probs if all((p, m) in by for m in METHODS)]
    both = [p for p in probs if _solved(by[(p, "SS-INK")]) and _solved(by[(p, "SS-INK-2")])]
    ratios = [by[(p, "SS-INK")]["cost"] / by[(p, "SS-INK-2")]["cost"] for p in both]
    cheaper = sum(1 for q in ratios if q > 1.0)
    macros.append(_macro("bothSolved", len(both)))
    macros.append(_macro("twoCheaperCount", cheaper))
    # linear solves (GMRES calls, retries included) per accepted outer step, Section 7.3
    per_step = lambda r: r["linear_solves"] / max(1, r["outer"])
    more = sum(1 for p in both if per_step(by[(p, "SS-INK-2")]) > per_step(by[(p, "SS-INK")]))
    equal = sum(1 for p in both if per_step(by[(p, "SS-INK-2")]) == per_step(by[(p, "SS-INK")]))
    macros.append(_macro("twoMoreSolvesCount", more))
    macros.append(_macro("twoEqualSolvesCount", equal))
    macros.append(_macro("twoFewerSolvesCount", len(both) - more - equal))
    macros.append(_macro("twoMedianFactor", f"{_median(ratios):.2f}" if ratios else "---"))
    macros.append(_macro("twoFactorMin", f"{min(ratios):.2f}" if ratios else "---"))
    macros.append(_macro("twoFactorMax", f"{max(ratios):.2f}" if ratios else "---"))
    med = _median(ratios)
    if med is None:
        verdict = "no instance is solved by both variants"
    elif med > 1.0:
        verdict = f"omitting them lowers the cost by a median factor of {med:.2f} on the instances both variants solve"
    else:
        verdict = (f"omitting them does not lower the cost: on the instances both variants solve the median of the "
                   f"cost ratio SS-INK/SS-INK-2 is {med:.2f}")
    macros.append(_macro("twoVerdict", verdict))
    only1 = [p for p in probs if _solved(by[(p, "SS-INK")]) and not _solved(by[(p, "SS-INK-2")])]
    only2 = [p for p in probs if _solved(by[(p, "SS-INK-2")]) and not _solved(by[(p, "SS-INK")])]
    macros.append(_macro("onlySSINK", ", ".join(only1) if only1 else "none"))
    macros.append(_macro("onlySSINKtwo", ", ".join(only2) if only2 else "none"))
    macros.append(_macro("sameCount", "the same number of" if len([p for p in probs if _solved(by[(p, 'SS-INK')])]) ==
                         len([p for p in probs if _solved(by[(p, 'SS-INK-2')])]) else "a different number of"))
    # common instances with L-BFGS
    common = [p for p in probs if _solved(by[(p, "SS-INK")]) and _solved(by[(p, "L-BFGS")])]
    rr = [by[(p, "SS-INK")]["cost"] / by[(p, "L-BFGS")]["cost"] for p in common]
    macros.append(_macro("commonLBFGS", len(common)))
    macros.append(_macro("commonRatioLBFGS", _rint(_median(rr)) if rr else "---"))
    # quartic 1000 statements
    q = {m: by.get(("QUARTIC-1000", m)) for m in METHODS}
    if any(v is None for v in q.values()):
        q = None
    if q is not None:
        macros.append(_macro("qNIssink", q["SS-INK"]["outer"]))
        macros.append(_macro("qCostSSINK", _sci_prose(q["SS-INK"]["cost"], 1)))
        macros.append(_macro("qStatusSSINK", "solves it" if _solved(q["SS-INK"]) else "does not solve it"))
        macros.append(_macro("qNIssinktwo", q["SS-INK-2"]["outer"]))
        macros.append(_macro("qCostSSINKtwo", _sci_prose(q["SS-INK-2"]["cost"], 1)))
        macros.append(_macro("qStatusSSINKtwo", "solves it" if _solved(q["SS-INK-2"]) else "does not solve it"))
        oth_all = [q[m] for m in ("ARC", "TR-NCG", "NCG", "RegN", "L-BFGS")]
        oth = [r for r in oth_all if _solved(r)]
        macros.append(_macro("qOthersSolvedList", ", ".join(r["method"] for r in oth) or "none"))
        macros.append(_macro("qOthersUnsolvedList", ", ".join(r["method"] for r in oth_all if not _solved(r)) or "none"))
        macros.append(_macro("qNIothersMin", min(r["outer"] for r in oth) if oth else "---"))
        macros.append(_macro("qNIothersMax", max(r["outer"] for r in oth) if oth else "---"))
        macros.append(_macro("qCostOthersMin", _num(min(r["cost"] for r in oth)) if oth else "---"))
        macros.append(_macro("qCostOthersMax", _num(max(r["cost"] for r in oth)) if oth else "---"))
        macros.append(_macro("qSolvedBy", ", ".join(m for m in METHODS if _solved(q[m])) or "none"))
        macros.append(_macro("qUnsolvedBy", ", ".join(m for m in METHODS if not _solved(q[m])) or "none"))
        uns = [m for m in METHODS if not _solved(q[m])]
        macros.append(_macro("qSolvedSentence", "which all nine methods solve" if not uns else
                             "which " + ", ".join(m for m in METHODS if _solved(q[m])) + " solve and " + ", ".join(uns) + (" does not" if len(uns) == 1 else " do not")))
    # COSINE statements
    cos_status = {}
    for n, tag in ((1000, "K"), (10000, "TenK")):
        r1, r2 = by.get((f"COSINE-{n}", "SS-INK")), by.get((f"COSINE-{n}", "SS-INK-2"))
        if r1 is None or r2 is None:
            continue
        cos_status[("SS-INK", n)], cos_status[("SS-INK-2", n)] = _solved(r1), _solved(r2)
        macros.append(_macro(f"cosSSINK{tag}", "solves" if _solved(r1) else "does not solve"))
        macros.append(_macro(f"cosSSINKtwo{tag}", "solves" if _solved(r2) else "does not solve"))
    if len(cos_status) == 4:
        if not any(cos_status.values()):
            verdict = "Neither variant solves it at either size"
        else:
            parts = []
            for m in ("SS-INK", "SS-INK-2"):
                sol = [str(n) for n in (1000, 10000) if cos_status[(m, n)]]
                parts.append(f"{m} solves it at $n = {', '.join(sol)}$" if sol else f"{m} solves it at neither size")
            verdict = "; ".join(parts)
        macros.append(_macro("cosVerdict", verdict))
    for n, tag in ((1000, "K"), (10000, "TenK")):
        tr = [r for r in cosine if r["problem"] == f"COSINE-{n}"]
        if tr:
            macros.append(_macro(f"cosTrunc{tag}", (f"solves it at a cost of ${int(round(tr[0]['cost']))}$"
                                                   if _solved(tr[0]) else "does not solve it either")))
    # longest run and wall clock
    walls = [r["time"] for r in main]          # oracle wall clock of the run itself
    macros.append(_macro("longestRunSeconds", _rint(max(walls))))
    macros.append(_macro("longestRunInstance", max(main, key=lambda r: r["time"])["problem"] + " (" + max(main, key=lambda r: r["time"])["method"] + ")"))
    macros.append(_macro("anyTimeLimit", "yes" if any(r["status"] == "time" for r in main) else "no"))
    macros.append(_macro("anyIterCap", "yes" if any(r["status"] == "iteration cap" for r in main) else "no"))
    # forcing attained in all main SS-INK runs?
    for m, tag in (("SS-INK", "SSINK"), ("SS-INK-2", "SSINKtwo")):
        rs = [r for r in main if r["method"] == m]
        macros.append(_macro(f"forcingMissedRuns{tag}", sum(1 for r in rs if r.get("forcing_missed", 0) > 0)))
        macros.append(_macro(f"forcingMissedSolves{tag}", sum(r.get("forcing_missed", 0) for r in rs)))
        macros.append(_macro(f"linearSolves{tag}", sum(r.get("linear_solves", 0) for r in rs)))
        macros.append(_macro(f"pZeroRuns{tag}", sum(1 for r in rs if r.get("p_zero", 0) > 0)))
        macros.append(_macro(f"pZeroTotal{tag}", sum(r.get("p_zero", 0) for r in rs)))
    # quartic multiplicity and MLP saddle facts
    macros.append(_macro("quarticMultH", 100))
    macros.append(_macro("quarticMultK", 1000))
    for name, tag in (("MLP-241", "A"), ("MLP-1101", "B")):
        if name in saddle:
            s = saddle[name]
            macros.append(_macro(f"mlpSaddleLam{tag}", f"{s['lambda_min']:.3f}"))
            macros.append(_macro(f"mlpSaddleMult{tag}", s["multiplicity"]))
            macros.append(_macro(f"mlpSaddleSmallest{tag}", _sci_prose(s["smallest_modulus"], 2)))


def metric_macros(metric: Dict, macros: List[str]) -> None:
    """Numbers quoted in Appendix B (results/metric.json, written by srdssink.metric)."""
    sv = metric.get("sv_count", {})
    cg = metric.get("cg", {})
    sizes_sv = sorted(int(k) for k in sv)
    macros.append(_macro("metricSVsizes", ", ".join(str(k) for k in sizes_sv)))
    counts = [sv[str(k)]["count_above"] for k in sizes_sv]
    macros.append(_macro("metricSVcounts", ", ".join(str(c) for c in counts)))
    macros.append(_macro("metricSVmin", min(counts) if counts else "---"))
    macros.append(_macro("metricSVmax", max(counts) if counts else "---"))
    if sizes_sv:
        macros.append(_macro("metricSVnMin", sizes_sv[0] ** 2))
        macros.append(_macro("metricSVnMax", sizes_sv[-1] ** 2))
    sizes = sorted(int(k) for k in cg)
    pd = [k for k in sizes if cg[str(k)]["euclid_pd"] and cg[str(k)]["mh_pd"]
          and cg[str(k)].get("euclid_pd_spectral", True) and cg[str(k)].get("mh_pd_spectral", True)]
    notpd = [k for k in sizes if k not in pd]
    macros.append(_macro("metricCGsizes", ", ".join(str(k) for k in pd)))
    macros.append(_macro("metricCGsizesNotPD", ", ".join(str(k) for k in notpd) if notpd else "none"))
    macros.append(_macro("metricCGcountNotPD", len(notpd)))
    macros.append(_macro("metricCGMh", ", ".join(str(cg[str(k)]["mh_iters"]) for k in pd)))
    macros.append(_macro("metricCGEuclid", ", ".join(str(cg[str(k)]["euclid_iters"]) for k in pd)))
    for k in notpd:
        c = cg[str(k)]
        macros.append(_macro(f"metricLamMinN{_roman(k)}", _sci_prose(c["lambda_min"], 2)))
        macros.append(_macro(f"metricAlphaN{_roman(k)}", _sci_prose(c["alpha"], 2)))
        if "mu_min" in c:
            macros.append(_macro(f"metricMuMinN{_roman(k)}", _sci_prose(c["mu_min"], 2)))
            macros.append(_macro(f"metricAlphaMhN{_roman(k)}", _sci_prose(c["alpha_mh"], 2)))
    p1 = metric.get("p1", {})
    if p1:
        ks = sorted(int(k) for k in p1)
        macros.append(_macro("metricPoneSizes", ", ".join(str(k) for k in ks)))
        macros.append(_macro("metricPoneStiffErr", _sci_prose(max(p1[str(k)]["stiffness_vs_h2L_maxabs"] for k in ks), 1)))
        macros.append(_macro("metricPoneMassErr", _sci_prose(max(p1[str(k)]["lumped_mass_vs_h2_maxrel"] for k in ks), 1)))
    macros.append(_macro("metricSplitErrMax", _sci_prose(max(v["splitting"] for v in metric["splitting_error"].values()), 1)))


def constants_macros(c: Dict, macros: List[str]) -> None:
    """Numbers quoted in Section 2.4 and Section 5.3 (results/constants.json, written by srdssink.constants)."""
    macros.append(_macro("rosenGradNorm", _sci_prose(c["grad_norm_u0"], 2)))
    macros.append(_macro("rosenEps", f"{c['eps']:.2f}"))
    macros.append(_macro("rosenThirdNorm", _sci_prose(c["third_norm_u0"], 2)))
    macros.append(_macro("rosenGainThreshold", _sci_prose(c["gain_threshold_lower_bound"], 2)))


def _join(items: List[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + (" and " if len(items) == 2 else ", and ") + items[-1]


def retry_macros(abl: List[Dict], main: List[Dict], macros: List[str]) -> None:
    """Section 4.3: SS-INK with r_max = 1 against the main campaign (results/retry_ablation.json)."""
    by = {r["problem"]: r for r in main if r["method"] == "SS-INK"}
    order = _sorted_problems([r["problem"] for r in abl])
    rec = {r["problem"]: r for r in abl}
    stall = [p for p in order if _solved(by[p]) and not _solved(rec[p])]
    same = [p for p in order if rec[p]["status"] == by[p]["status"]]
    same_ni = [p for p in same if rec[p]["outer"] == by[p]["outer"]]
    macros.append(_macro("retryInstances", _join(order)))
    macros.append(_macro("retryStallCount", len(stall)))
    macros.append(_macro("retryStallList", _join(stall) if stall else "none"))
    if stall:
        dts = _join([f"${_sci_prose(rec[p]['dt'], 1)}$" for p in stall])
        gs = _join([(f"${rec[p]['grad_norm_exit']:.1f}$" if 1.0 <= abs(rec[p]['grad_norm_exit']) < 1e3
                     else f"${_sci_prose(rec[p]['grad_norm_exit'], 1)}$") for p in stall])
        resp = ", respectively," if len(stall) > 1 else ""
        macros.append(_macro("retryStallSentence",
                             f"the pseudo-time step falls to {dts}{resp} and the gradient norm stays at {gs}"
                             f" until the budget is exhausted"))
    if same:
        tail = (" and so is the number of outer iterations" if same_ni == same else "")
        macros.append(_macro("retryUnaffectedSentence", f"on {_join(same)} the outcome is unchanged{tail}"))
    else:
        macros.append(_macro("retryUnaffectedSentence", "no instance has an unchanged outcome"))


def repeat_macros(rep: List[Dict], main: List[Dict], macros: List[str]) -> None:
    """Section 7.1 (reproducibility): a second run of the main campaign against the first
    (results/repeat.json, written by srdssink.campaign repeat)."""
    first = {(r["problem"], r["method"]): r for r in main}
    pairs = [(r, first[(r["problem"], r["method"])]) for r in rep if (r["problem"], r["method"]) in first]
    identical = [a for a, b in pairs if a["cost"] == b["cost"] and a["status"] == b["status"]
                 and a["outer"] == b["outer"]]
    differ = [(a, b) for a, b in pairs if not (a["cost"] == b["cost"] and a["status"] == b["status"]
                                               and a["outer"] == b["outer"])]
    changed = [(a, b) for a, b in pairs if _solved(a) != _solved(b)]
    rel = [abs(a["cost"] - b["cost"]) / b["cost"] for a, b in differ if b["cost"] > 0]
    macros.append(_macro("repeatRuns", len(pairs)))
    macros.append(_macro("repeatIdentical", len(identical)))
    macros.append(_macro("repeatDiffer", len(differ)))
    macros.append(_macro("repeatMaxRelDiff", f"{100.0 * max(rel):.2f}" if rel else "0"))
    macros.append(_macro("repeatStatusChanges", len(changed)))
    meths = _join(sorted({a["method"] for a, _ in differ}, key=METHODS.index)) if differ else "none"
    macros.append(_macro("repeatDifferMethods", meths))
    if changed:
        lst = _join([f"{a['method']} on {a['problem']} ({b['status']} in the first run, {a['status']} in the second)"
                     for a, b in changed])
        macros.append(_macro("repeatStatusSentence", f"the classification into solved and unsolved changes for {lst}"))
    else:
        macros.append(_macro("repeatStatusSentence", "no run changes from solved to unsolved or back"))


def verify_macros(ver: Dict, macros: List[str]) -> None:
    """Largest derivative-check errors (results/verify.json, written by srdssink.verify)."""
    for key, tag in (("largest_benchmark", "Bench"), ("largest_mlp", "MLP")):
        vals = ver.get(key)
        if vals:
            for v, name in zip(vals, ("Grad", "Hvp", "Sym")):
                macros.append(_macro(f"verify{tag}{name}", _sci_prose(v, 1)))
    macros.append(_macro("verifyStep", f"10^{{{int(round(np.log10(ver.get('step', 1e-5))))}}}"))
    macros.append(_macro("verifyDirections", ver.get("directions", 3)))


def _name_tag(name: str) -> str:
    """Instance name as a LaTeX macro suffix: letters only (FLETCHCR-10000 -> FLETCHCRTenK)."""
    base, _, n = name.partition("-")
    return base.replace("_", "") + {"100": "H", "1000": "K", "10000": "TenK", "3000": "ThreeK",
                                    "256": "TwoFiftySix", "1024": "TenTwentyFour"}.get(n, n)


def _roman(k: int) -> str:
    return {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour", 128: "OneTwoEight"}.get(k, f"N{k}")


# --------------------------------------------------------------------------------- main ----


def build() -> None:
    main = _load("main.json")
    starts = _load("starts.json")
    esc = _load("escape.json") if os.path.exists(os.path.join(RESULTS_DIR, "escape.json")) else []
    sens = _load("sensitivity.json") if os.path.exists(os.path.join(RESULTS_DIR, "sensitivity.json")) else []
    cosine = _load("cosine_trunc.json") if os.path.exists(os.path.join(RESULTS_DIR, "cosine_trunc.json")) else []
    mlp = _load("mlp.json") if os.path.exists(os.path.join(RESULTS_DIR, "mlp.json")) else []
    mlp_esc = _load("mlp_escape.json") if os.path.exists(os.path.join(RESULTS_DIR, "mlp_escape.json")) else []
    saddle = _load("mlp_saddle.json") if os.path.exists(os.path.join(RESULTS_DIR, "mlp_saddle.json")) else {}
    metric = _load("metric.json") if os.path.exists(os.path.join(RESULTS_DIR, "metric.json")) else {}
    ver = _load("verify.json") if os.path.exists(os.path.join(RESULTS_DIR, "verify.json")) else {}
    consts = _load("constants.json") if os.path.exists(os.path.join(RESULTS_DIR, "constants.json")) else {}
    retry = _load("retry_ablation.json") if os.path.exists(os.path.join(RESULTS_DIR, "retry_ablation.json")) else []
    repeat = _load("repeat.json") if os.path.exists(os.path.join(RESULTS_DIR, "repeat.json")) else []
    arp = _load("arpack_repeat.json") if os.path.exists(os.path.join(RESULTS_DIR, "arpack_repeat.json")) else {}
    n_inst = len({r["problem"] for r in main})
    tables: Dict[str, List[str]] = {}
    macros: List[str] = []

    def tab(name):
        tables[name] = []
        return tables[name]

    table_pooled(main, n_inst, tab("pooled"), macros)
    table_per_suite(main, tab("per_suite"), macros)
    table_failures(main, starts, tab("failures"), macros)
    table_indefinite(main, starts, tab("indefinite"), macros)
    if esc:
        table_escape(esc, tab("escape"), macros)
    if sens:
        table_sensitivity(sens, tab("sensitivity"), macros)
    table_hvp_budget(main, tab("hvp_budget"), macros)
    if mlp:
        table_mlp(mlp, tab("mlp"), macros)
    if mlp_esc:
        table_mlp_escape(mlp_esc, tab("mlp_escape"), macros)
    prose_macros(main, cosine, starts, saddle, macros)
    if metric:
        metric_macros(metric, macros)
    if ver:
        verify_macros(ver, macros)
    if consts:
        constants_macros(consts, macros)
    if retry:
        retry_macros(retry, main, macros)
    if repeat:
        repeat_macros(repeat, main, macros)
    if arp:
        macros.append(_macro("arpackRepeats", arp["repeats"]))
        macros.append(_macro("arpackDistinct", arp["n_distinct"]))
        macros.append(_macro("arpackMaxDiff", _sci_prose(arp["max_abs_difference"], 1) if arp["max_abs_difference"] > 0 else "0"))
        macros.append(_macro("arpackInstance", arp["instance"]))

    os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
    for name, lines in tables.items():
        with open(os.path.join(OUT_DIR, "tables", name + ".tex"), "w") as fh:
            fh.write("% generated by srdssink.report from results/*.json; do not edit\n")
            fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "macros.tex"), "w") as fh:
        fh.write("% generated by srdssink.report from results/*.json; do not edit\n")
        fh.write("\n".join(macros) + "\n")
    figure_profile(main, macros)
    figure_histories(main)
    figure_hvp_budget(main)
    if sens:
        figure_sensitivity(sens)
    print(f"wrote {len(tables)} tables, {len(macros)} macros and the figures to {OUT_DIR}/")


if __name__ == "__main__":
    build()
