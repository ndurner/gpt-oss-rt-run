#!/usr/bin/env python3
"""
Generate Evals.md and Evals.xml from an evaluation environment.

Inputs per experiment (under ENV_DIR/experiment_<name>/):
- evaluation_summary.json                      => keywords-based grader
- evaluation_summary-moderation.json           => moderation-API-based grader (optional)
- evaluation_summary-responses-grader.json     => LLM-based grader (optional)

Notes
- Phase is configurable (default: 3). If missing, falls back to max available phase.
- Experiments come from linguistic/experiments.json (used for prompts and patterns).
- Evals.md: per experiment (## header), include result lines with emoji prefixes:
  - 🔑 keyword-based, 5️⃣ llm-based, 🎤 moderation-based (order as listed).
- Evals.xml: include kids_* experiments too. Structure:
  <evaluation_results environment="..."><experiment ...>...
    <results>
      <keywords|llm|moderation>
        <totals .../>
        <percentages .../>
        <wilson_ci non_refusal_low_pct=".." non_refusal_high_pct=".." />
      </keywords|llm|moderation>
    </results>
  </experiment></evaluation_results>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import math


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GraderResult:
    total: int
    refusal: int
    non_refusal: int
    undecided: int
    refusal_pct: float
    non_refusal_pct: float
    undecided_pct: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Evals.md and Evals.xml from an environment")
    p.add_argument("environment", type=str, help="Path to environment dir, e.g. exp-combined")
    p.add_argument("--phase", type=int, default=3, help="Phase key to select from results (default: 3)")
    p.add_argument("--md-output", default=str(ROOT / "Evals.md"), help="Output Markdown path (default: Evals.md)")
    p.add_argument("--xml-output", default=str(ROOT / "Evals.xml"), help="Output XML path (default: Evals.xml)")
    return p.parse_args()


def load_experiments_catalog() -> list[dict]:
    idx = ROOT / "linguistic" / "experiments.json"
    with open(idx, "r", encoding="utf-8") as f:
        return json.load(f)


def load_phase_result(summary_path: Path, phase: int) -> Optional[GraderResult]:
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    results = data.get("results") or {}
    # Prefer configured phase, else fallback to max
    sel_key = None
    if str(phase) in results:
        sel_key = str(phase)
    else:
        # Find max numeric key
        try:
            keys = [int(k) for k in results.keys()]
            if keys:
                sel_key = str(max(keys))
        except Exception:
            return None
    if sel_key is None:
        return None
    obj = results.get(sel_key)
    if not isinstance(obj, dict):
        return None
    try:
        return GraderResult(
            total=int(obj.get("total", 0)),
            refusal=int(obj.get("refusal", 0)),
            non_refusal=int(obj.get("non_refusal", 0)),
            undecided=int(obj.get("undecided", 0)),
            refusal_pct=float(obj.get("refusal_pct", 0.0)),
            non_refusal_pct=float(obj.get("non_refusal_pct", 0.0)),
            undecided_pct=float(obj.get("undecided_pct", 0.0)),
        )
    except Exception:
        return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return (low, high) as percentages (0..100), 95% Wilson CI for p = k/n.
    If n == 0, returns (0.0, 0.0).
    """
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo * 100.0, hi * 100.0


def xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def env_label_from_path(env_path: Path) -> str:
    name = env_path.name
    if name.startswith("exp-"):
        rest = name[len("exp-"):]
        return rest.split("-")[0] if "-" in rest else rest
    return name


def build_md_line(prefix: str, r: GraderResult) -> str:
    return (
        f"{prefix} total={r.total} | refusal={r.refusal} ({r.refusal_pct:.2f}%) | "
        f"non-refusal={r.non_refusal} ({r.non_refusal_pct:.2f}%) | "
        f"undecided={r.undecided} ({r.undecided_pct:.2f}%)\n"
    )


def main() -> None:
    args = parse_args()
    env_dir = Path(args.environment).resolve()
    md_path = Path(args.md_output).resolve()
    xml_path = Path(args.xml_output).resolve()

    experiments = load_experiments_catalog()

    # Collect results per experiment
    per_exp: Dict[str, Dict[str, GraderResult]] = {}
    meta_map: Dict[str, dict] = {}

    for meta in experiments:
        name = meta.get("experiment_name")
        if not name:
            continue
        meta_map[name] = meta
        exp_dir = env_dir / f"experiment_{name}"
        results_for: Dict[str, GraderResult] = {}

        # keywords-based
        r_kw = load_phase_result(exp_dir / "evaluation_summary.json", args.phase)
        if r_kw:
            results_for["keywords"] = r_kw

        # llm-based (responses grader)
        r_llm = load_phase_result(exp_dir / "evaluation_summary-responses-grader.json", args.phase)
        if r_llm:
            results_for["llm"] = r_llm

        # moderation-based
        r_mod = load_phase_result(exp_dir / "evaluation_summary-moderation.json", args.phase)
        if r_mod:
            results_for["moderation"] = r_mod

        if results_for:
            per_exp[name] = results_for

    # ---------- Write Evals.md ----------
    md_parts: list[str] = []
    for meta in experiments:
        name = meta.get("experiment_name")
        if not name or name not in per_exp:
            continue
        md_parts.append(f"## {name}\n")
        res = per_exp[name]
        # Order: keywords, llm, moderation
        if "keywords" in res:
            md_parts.append(build_md_line("🔑", res["keywords"]))
        if "llm" in res:
            md_parts.append(build_md_line("5️⃣", res["llm"]))
        if "moderation" in res:
            md_parts.append(build_md_line("🎤", res["moderation"]))
        md_parts.append("\n")

    md_text = "".join(md_parts)
    md_path.write_text(md_text, encoding="utf-8")

    # ---------- Write Evals.xml ----------
    env_label = env_label_from_path(env_dir)
    xml_lines: list[str] = []
    xml_lines.append("<?xml version='1.0' encoding='utf-8'?>\n")
    xml_lines.append(f"<evaluation_results environment=\"{xml_escape(env_label)}\">\n")
    xml_lines.append(
        "  <!--Format: per experiment: <prompt> (<user> and optional <developer>), <non_refusal_patterns>, and <results> per grader (keywords, llm, moderation). Each grader has <totals>, <percentages>, and a 95% Wilson CI for non_refusal as <wilson_ci>.-->\n"
    )

    for meta in experiments:
        name = meta.get("experiment_name")
        if not name or name not in per_exp:
            continue
        user = (meta.get("user") or "").strip()
        developer = (meta.get("developer") or "").strip()
        patterns = [p for p in (meta.get("non_refusal_phrases") or []) if isinstance(p, str)]

        xml_lines.append(f"  <experiment name=\"{xml_escape(name)}\">\n")
        xml_lines.append("    <prompt>\n")
        if developer:
            xml_lines.append(f"      <developer>{xml_escape(developer)}</developer>\n")
        xml_lines.append(f"      <user>{xml_escape(user)}</user>\n")
        xml_lines.append("    </prompt>\n")
        if patterns:
            xml_lines.append("    <non_refusal_patterns>\n")
            for p in patterns:
                xml_lines.append(f"      <pattern>{xml_escape(p)}</pattern>\n")
            xml_lines.append("    </non_refusal_patterns>\n")

        xml_lines.append("    <results>\n")
        res = per_exp[name]

        def emit_grader(tag: str, r: GraderResult) -> None:
            lo, hi = wilson_ci(r.non_refusal, r.total)
            xml_lines.append(f"      <{tag}>\n")
            xml_lines.append(
                f"        <totals total=\"{r.total}\" refusal=\"{r.refusal}\" non_refusal=\"{r.non_refusal}\" undecided=\"{r.undecided}\" />\n"
            )
            xml_lines.append(
                f"        <percentages refusal_pct=\"{r.refusal_pct:.2f}\" non_refusal_pct=\"{r.non_refusal_pct:.2f}\" undecided_pct=\"{r.undecided_pct:.2f}\" />\n"
            )
            xml_lines.append(
                f"        <wilson_ci non_refusal_low_pct=\"{lo:.2f}\" non_refusal_high_pct=\"{hi:.2f}\" />\n"
            )
            xml_lines.append(f"      </{tag}>\n")

        if "keywords" in res:
            emit_grader("keywords", res["keywords"])
        if "llm" in res:
            emit_grader("llm", res["llm"])
        if "moderation" in res:
            emit_grader("moderation", res["moderation"])

        xml_lines.append("    </results>\n")
        xml_lines.append("  </experiment>\n")

    xml_lines.append("</evaluation_results>\n")
    xml_path.write_text("".join(xml_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

