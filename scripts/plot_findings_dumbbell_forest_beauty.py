#!/usr/bin/env python3
"""
Dumbbell/forest plot for Kaggle writeups (polished):
- Baseline vs. Bypass with Wilson 95% CIs
- Newcombe 95% CI for Δ (mod - base)
- Inter font (fallback safe)
- Footer caption (not chopped)
- Optional logo watermark placed in the right gutter (upper-right)

Example:
  python3 scripts/plot_findings_dumbbell_forest_beauty.py \
      --phase 3 --metric non_refusal --sort \
      --figwidth 13.5 --right-gutter 0.22 \
      --logo assets/ai_sweet_harmony_logo.png \
      --logo-pos tr --logo-scale 0.26 --logo-opacity 0.9 \
      --font-dir assets/fonts/Inter
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
EXP_DIR = BASE_DIR / "exp-combined"
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Data structures & math
# -------------------------
@dataclass
class GroupStats:
    n: int
    count: int  # count for the chosen metric

    @property
    def p(self) -> float:
        return 0.0 if self.n == 0 else self.count / self.n

    @property
    def pct(self) -> float:
        return self.p * 100.0


def wilson_ci(p: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def newcombe_diff_ci(
    p1: float, n1: int, p2: float, n2: int, z: float = 1.959963984540054
) -> Tuple[float, float]:
    l1, u1 = wilson_ci(p1, n1, z)
    l2, u2 = wilson_ci(p2, n2, z)
    return (l2 - u1, u2 - l1)


def read_phase_stats(experiment_name: str, phase: int, metric: str) -> GroupStats:
    """
    Reads exp-combined/experiment_{name}/evaluation_summary.json and returns counts
    for 'metric' in the given phase.

    Structure:
      {
        "results": { "3": { "total": 80, "refusal": ..., "non_refusal": ..., "undecided": ... } }
      }

    If you later wire in a grader JSON, switch on --metric harmful to load that file.
    """
    p = EXP_DIR / f"experiment_{experiment_name}" / "evaluation_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    data = json.loads(p.read_text())
    r = data["results"][str(phase)]
    total = int(r.get("total", 0))
    if metric not in r:
        if metric == "harmful":
            metric_key = "non_refusal"  # alias until grader JSON is wired
        else:
            raise KeyError(f"Metric '{metric}' not in results; keys: {list(r.keys())}")
    else:
        metric_key = metric
    count = int(r[metric_key])
    return GroupStats(n=total, count=count)

# -------------------------
# Fonts & styling
# -------------------------
def try_register_inter(font_dir: Optional[Path]) -> None:
    """
    Register Inter TTFs if available; otherwise fallback to default fonts.
    """
    if font_dir and font_dir.exists():
        from matplotlib import font_manager
        added = 0
        for ttf in font_dir.glob("*.ttf"):
            try:
                font_manager.fontManager.addfont(str(ttf))
                added += 1
            except Exception:
                pass
        if added:
            mpl.rcParams["font.family"] = "Inter"

    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.right": False,
        "axes.spines.top": False
    })

# -------------------------
# Colors & utils
# -------------------------
COLOR_BASE = "#6e7781"   # neutral gray
COLOR_MOD  = "#e45756"   # coral (increase)
COLOR_DEC  = "#2bb0b2"   # teal  (decrease)
COLOR_LINE = "#c3c7cf"
COLOR_FOOT = "#6b7280"

def fmt_pp(x: float) -> str:
    sign = "−" if x < 0 else "+"
    return f"{sign}{abs(x):.1f} pp"

def delta_color(d: float) -> str:
    return COLOR_DEC if d < 0 else COLOR_MOD

# -------------------------
# Logo in the right gutter
# -------------------------
def add_logo_in_gutter(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    logo_path: Optional[Path],
    position: str = "tr",
    scale: float = 0.24,      # fraction of axes height
    opacity: float = 0.10
) -> None:
    """
    Places the logo in the right gutter, aligned to the axes' right edge.
    - position: 'tr' or 'br' (top-right / bottom-right in the gutter)
    - scale: fraction of axes height the logo should occupy
    - opacity: 0..1
    """
    if not logo_path or not Path(logo_path).exists():
        return

    # Figure size in pixels
    w_in, h_in = fig.get_size_inches()
    dpi = fig.get_dpi()
    fig_w_px, fig_h_px = int(w_in * dpi), int(h_in * dpi)

    # Axes bbox in figure coordinates
    bbox = ax.get_position()
    gutter_left = bbox.x1  # start of gutter in figure coords
    gutter_right = 0.995   # keep a small right margin in fig coords
    gutter_width = max(0.0, gutter_right - gutter_left)

    img = Image.open(logo_path).convert("RGBA")
    # Target height = scale * axes pixel height
    axes_h_px = int((bbox.y1 - bbox.y0) * fig_h_px)
    target_h = int(axes_h_px * scale)
    target_h = max(1, target_h)
    aspect = img.width / img.height
    target_w = int(target_h * aspect)
    img = img.resize((target_w, target_h), Image.LANCZOS)

    # Apply opacity
    if opacity < 1.0:
        r, g, b, a = img.split()
        a = a.point(lambda px: int(px * opacity))
        img = Image.merge("RGBA", (r, g, b, a))

    # Position inside the gutter in pixels
    gx0 = int(gutter_left * fig_w_px)
    gx1 = int(gutter_right * fig_w_px)
    gutter_px_w = max(1, gx1 - gx0)

    # Right-align with a small inset
    x = gx1 - target_w - int(0.02 * fig_w_px)
    if position == "br":
        y = int(bbox.y0 * fig_h_px) + int(0.02 * fig_h_px)
    else:  # top-right
        y = int(bbox.y1 * fig_h_px) - target_h - int(0.02 * fig_h_px)

    # Draw
    fig.figimage(np.array(img), xo=x, yo=y, zorder=5)

# -------------------------
# Main plot
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=3, help="Phase key in evaluation_summary.json")
    parser.add_argument("--metric", type=str, default="non_refusal",
                        choices=["non_refusal", "refusal", "undecided", "harmful"],
                        help="Which result field to chart. 'harmful' aliases to non_refusal unless grader JSON is wired.")
    parser.add_argument("--sort", action="store_true", help="Sort rows by |Δ| descending")
    parser.add_argument("--outfile", type=str, default="findings_dumbbell_forest_beauty")
    parser.add_argument("--font-dir", type=str, default="", help="Directory with Inter *.ttf")
    parser.add_argument("--logo", type=str, default="", help="Path to logo PNG")
    parser.add_argument("--logo-pos", type=str, default="tr", choices=["tr","br"])
    parser.add_argument("--logo-scale", type=float, default=0.24, help="Logo height as fraction of axes height")
    parser.add_argument("--logo-opacity", type=float, default=0.10, help="Logo opacity 0..1")
    parser.add_argument("--figwidth", type=float, default=13.0, help="Figure width in inches")
    parser.add_argument("--right-gutter", type=float, default=0.20, help="Fraction of figure reserved for right gutter (0..0.35)")
    args = parser.parse_args()

    try_register_inter(Path(args.font_dir) if args.font_dir else None)

    # Mapping aligned with your Writeup
    mapping: Dict[str, Tuple[str, str, str]] = {
        "Cyber Threat (ZIP-bomb)": (
            "zipb-barebones",
            "zipb-educator-neutral",
            "Direct ask vs Educator 'what to avoid'"
        ),
        "Chemical Hazard (Backpack)": (
            "meth_backpack_barebones",
            "meth_backpack_barebones-fr-edu",
            "Backpack red-flags; EN vs FR-edu"
        ),
        "Data Exfiltration (RAG)": (
            "rag-rp_linux-barebones",
            "rag-rp_linux-spk_fre",
            "RAG 'don’t reveal' vs Linux role-play"
        ),
        "Child Safety (Driving)": (
            "kids_car_barebones",
            "kids_car_barebones_ger",
            "Kid-safety dev prompt; EN vs DE"
        ),
    }

    # Gather stats
    rows: List[dict] = []
    for label, (base_key, mod_key, subtitle) in mapping.items():
        b = read_phase_stats(base_key, args.phase, args.metric)
        m = read_phase_stats(mod_key, args.phase, args.metric)
        bl, bu = wilson_ci(b.p, b.n)
        ml, mu = wilson_ci(m.p, m.n)
        dlow, dhigh = newcombe_diff_ci(b.p, b.n, m.p, m.n)
        rows.append({
            "label": label,
            "subtitle": subtitle,
            "b_n": b.n, "b_pct": b.pct, "b_lo": bl*100, "b_hi": bu*100,
            "m_n": m.n, "m_pct": m.pct, "m_lo": ml*100, "m_hi": mu*100,
            "d_pct": (m.p - b.p)*100, "d_lo": dlow*100, "d_hi": dhigh*100
        })

    # Additional manually specified row: Tournament awareness vs. helpful
    # Provided data (present=n; exploit_n=non_refusal counts)
    # through `tournament/tournament_redteam_inconsistency.py --eval --phases 3`:
    #   cyberthreat@3: n=80, non_refusal=1
    #   helpful@3:     n=80, non_refusal=11
    t_base = GroupStats(n=80, count=1)
    t_mod  = GroupStats(n=80, count=11)
    tbl, tbu = wilson_ci(t_base.p, t_base.n)
    tml, tmu = wilson_ci(t_mod.p,  t_mod.n)
    tdl, tdh = newcombe_diff_ci(t_base.p, t_base.n, t_mod.p, t_mod.n)
    rows.append({
        "label": "Evaluation Awareness (Tournament)",
        "subtitle": "'harmful' vs. 'helpful' evaluation tournament setup",
        "b_n": t_base.n, "b_pct": t_base.pct, "b_lo": tbl*100, "b_hi": tbu*100,
        "m_n": t_mod.n,  "m_pct": t_mod.pct,  "m_lo": tml*100, "m_hi": tmu*100,
        "d_pct": (t_mod.p - t_base.p)*100, "d_lo": tdl*100, "d_hi": tdh*100
    })

    if args.sort:
        rows.sort(key=lambda r: abs(r["d_pct"]), reverse=True)

    # Figure & layout
    # Give more vertical breathing room so per-row annotations (subtitle + Δ)
    # do not collide with the x-axis or footer, and leave space for a bottom legend.
    # Slightly tighter per-row spacing overall
    fig_h = 1.10 * len(rows) + 3.7
    fig = plt.figure(figsize=(args.figwidth, fig_h), facecolor="white")
    # Subplot margins: reserve a right gutter for Δ labels + logo and a bottom margin for footer
    left = 0.22
    right = 1.0 - args.right_gutter
    top = 0.86
    # Extra bottom margin for legend + footer
    bottom = 0.24
    ax = fig.add_axes([left, bottom, right - left, top - bottom])

    ys = list(range(len(rows)))[::-1]
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])

    ax.set_xlim(-5, 105)
    # Keep only a small cushion below/above so rows don't feel detached.
    # More bottom cushion for the last row; a bit less above the top row
    ax.set_ylim(-0.25, len(rows) - 1 + 0.08)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)

    metric_title = {
        "non_refusal": "Non-refusal (%)",
        "refusal": "Refusal (%)",
        "undecided": "Undecided (%)",
        "harmful": "Harmful (%)"
    }[args.metric]
    ax.set_xlabel(f"{metric_title}  (80 runs; Wilson 95% CI)")

    # Horizontal separators between findings (subtle)
    for k in range(len(rows) - 1):
        ax.axhline(k + 0.5, color="#e5e7eb", lw=0.8, zorder=0)

    # Draw dumbbells
    for y, r in zip(ys, rows):
        ofs = -0.25 if y == 4 else 0

        # connector
        ax.plot([r["b_pct"], r["m_pct"]], [y + ofs, y + ofs], color="#d9dbe1", lw=2, zorder=1)

        # baseline CI + point
        ax.plot([r["b_lo"], r["b_hi"]], [y + ofs, y + ofs], color=COLOR_BASE, lw=3, alpha=0.95, zorder=2)
        ax.scatter([r["b_pct"]], [y + ofs], s=78, color=COLOR_BASE, zorder=3)

        # Bypass CI + point
        ax.plot([r["m_lo"], r["m_hi"]], [y + ofs, y + ofs], color=COLOR_MOD, lw=3, alpha=0.95, zorder=2)
        ax.scatter([r["m_pct"]], [y + ofs], s=78, color=COLOR_MOD, zorder=3)

        # Subtitle: above the dumbbell and aligned to the left edge of the
        # chart body for all rows to create a clean column.
        x_left_anchor = ax.get_xlim()[0] + 2
        ax.text(x_left_anchor, y + ofs + 0.10, r["subtitle"], va="bottom", ha="left",
                fontsize=10, color="#666666")

        # Δ label: just below the subtitle, right-aligned on the chart body.
        delta_txt = (
            f"Δ {fmt_pp(r['d_pct'])}  [ {fmt_pp(r['d_lo']).replace(' pp','')}, "
            f"{fmt_pp(r['d_hi']).replace(' pp','')} ]"
        )
        xmax = ax.get_xlim()[1]
        ax.text(xmax - 1.0, y + ofs + 0.04, delta_txt, va="bottom", ha="right", fontsize=11,
                color=delta_color(r["d_pct"]))

    # Legend
    legend_elems = [
        Line2D([0], [0], color=COLOR_BASE, lw=3, marker='o', ms=6, mec=COLOR_BASE, mfc=COLOR_BASE, label="Baseline"),
        Line2D([0], [0], color=COLOR_MOD,  lw=3, marker='o', ms=6, mec=COLOR_MOD,  mfc=COLOR_MOD,  label="Bypass"),
        Line2D([0], [0], color=COLOR_DEC,  lw=3, label="Δ < 0 (reduction)")
    ]
    # Legend below the x-axis
    ax.legend(
        handles=legend_elems,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.18),
        ncol=3
    )

    # Titles
    metric_hint = "keyword triage" if args.metric == "non_refusal" else ("LLM grader" if args.metric == "harmful" else args.metric)
    fig.suptitle("Framing and language shift gpt-oss-20B assistance rates (80 runs; Wilson 95% CI)",
                 y=0.965, fontsize=18, fontweight="bold")
    # Place a left-aligned subtitle beneath the suptitle (not overlapping).
    fig.text(left + 0.01, 0.90,
             f"Baseline → Bypass per finding. Δ with Newcombe 95% CI · metric: {metric_hint}",
             ha="left", va="top", fontsize=12, color="#555555")

    # Footer caption (not chopped)
    mapping_footer = (
        "Mapping:\n"
        "  • ZIP-bomb (zipb-barebones→zipb-educator-neutral)\n"
        "  • Chemical Hazard (meth_backpack_barebones→meth_backpack_barebones-fr-edu)\n"
        "  • Data Exfil (rag-rp_linux-barebones→rag-rp_linux-barebones-spk_fre)\n"
        "  • Child Safety (kids_car_barebones→kids_car_barebones_ger)"
    )
    footer = (
        "Kaggle: OpenAI gpt-oss-20b Red-Teaming · 80 runs per cell · Wilson score intervals; Newcombe Δ CIs\n"
        + mapping_footer
    )
    # Footer caption aligned with y-axis labels (same left as axes area minus tick padding)
    fig.text(0.11, 0.04, footer, ha="left", va="bottom", fontsize=9.6, color=COLOR_FOOT)

    # Logo in the right gutter (upper-right by default)
    add_logo_in_gutter(
        fig,
        ax,
        Path(args.logo) if args.logo else None,
        position=args.logo_pos,
        scale=args.logo_scale,
        opacity=args.logo_opacity
    )

    # Save
    out_png = OUT_DIR / f"{args.outfile}.png"
    out_svg = OUT_DIR / f"{args.outfile}.svg"
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_svg)
    print(f"Saved: {out_png}\nSaved: {out_svg}")

    # CSV summary
    csv_lines = ["Finding,Group,N,Count,Percent,WilsonLow,WilsonHigh,Diff_pp,DiffLow_pp,DiffHigh_pp"]
    for r in rows:
        csv_lines.append(f"{r['label']},Baseline,{r['b_n']},{round(r['b_n']*r['b_pct']/100)},"
                         f"{r['b_pct']:.2f},{r['b_lo']:.2f},{r['b_hi']:.2f},,,")
        csv_lines.append(f"{r['label']},Bypass,{r['m_n']},{round(r['m_n']*r['m_pct']/100)},"
                         f"{r['m_pct']:.2f},{r['m_lo']:.2f},{r['m_hi']:.2f},"
                         f"{r['d_pct']:.2f},{r['d_lo']:.2f},{r['d_hi']:.2f}")
    (OUT_DIR / f"{args.outfile}.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUT_DIR / f'{args.outfile}.csv'}")


if __name__ == "__main__":
    main()
