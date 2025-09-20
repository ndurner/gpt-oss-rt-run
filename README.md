# Kaggle Red‑Teaming Challenge - OpenAI gpt-oss-20b
![Logo](assets/harmony.jpg)

# Introduction
This repository contains primary assets submitted to the [Kaggle red-teaming challenge of OpenAI gpt-oss](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming), as well as supplementary assets. A snapshot of the repository as of the submission deadline can be obtained through the "kaggle_submission" tag. This repository continues to evolve as work continues.

# Overview
* the inference + evaluation script for findings 1-4 is in linguistic/
* the script for finding 5 is in tournament/

# Usage
See my [Kaggle Writeup](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/writeups/in-a-sweet-harmony-guardrail-bypasses-and-evaluati).

## Concrete invocations
### gpt-5-mini adjudication
In the folder that hosts generated scenario experiment outputs, for all `rag2` experiments:
```
python3 ../linguistic/findings_1-4_run_eval.py --api-evals-only --api-eval-methods responses-grader --experiment-prefix rag2 --experiments-file ../linguistic/experiments.json
```

## Update Evals.{md,xml}
With scenerio experiment outputs in `exp-combined`, for N=80 iterations ("phase 3"):
```
python3 scripts/update_evals.py --phase 3 exp-combined
```

# Supplemental materials
## Evals
Preliminary evals over all experiments are in [Evals.md](Evals.md) and [Evals.xml](Evals.xml). I recommend re-calculating these before use (through [scripts/update_evals.py](scripts/update_evals.py)). Find the condensed main findings in OpenAI Harmony format in [kaggle_submission](kaggle_submission)/findings-*.json. The raw models outputs are included in the Release as an attachment (experiments-raw.tar.xz). Early experiment runs with hosted inference providers are summarized in [Evals-hosted.md](Evals-hosted.md)

## Reporting
* update script to load Harmony walkthroughs into finding.json files: [update_harmony_walkthroughs.py](scripts/update_harmony_walkthroughs.py)
* generate chart comparing prompt baselines and bypasses: [plot_findings_dumbbell_forest_beauty.py](scripts/plot_findings_dumbbell_forest_beauty.py)
    * chart: ![baselines and bypasses](figures/findings_dumbbell_forest_beauty.png)