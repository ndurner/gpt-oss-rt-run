# Red‑Teaming Challenge - OpenAI gpt-oss-20b
![Logo](assets/harmony.jpg)

# Introduction
This repository contains primary assets submitted to the [Kaggle red-teaming challenge of OpenAI gpt-oss](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming), as well as supplementary assets. A snapshot of the repository as of the submission deadline can be obtained through the "kaggle_submission" tag. This repository continues to evolve as work continues.

# Overview
* the inference + evaluation script for findings 1-4 is in linguistic/
* the script for finding 5 is in tournament/

# Usage
See my [Kaggle Writeup](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/writeups/in-a-sweet-harmony-guardrail-bypasses-and-evaluati)

# Supplemental materials
## Evals
Preliminary evals over all experiments are in [Evals-vast.md](Evals-vast.md) and [Evals-vast.xml](Evals-vast.xml). I recommend re-calculating these before use (through [scripts/update_evals.py](scripts/update_evals.py)). Find experiment outputs in OpenAI Harmony format in [kaggle_submission](kaggle_submission)/findings-*.json. Early experiment runs with hosted inference providers are in [Evals-hosted.md](Evals-hosted.md)

## Reporting
* update script to load Harmony walkthroughs into finding.json files: [update_harmony_walkthroughs.py](scripts/update_harmony_walkthroughs.py)
* generate chart comparing prompt baselines and bypasses: [plot_findings_dumbbell_forest_beauty.py](scripts/plot_findings_dumbbell_forest_beauty.py)
    * chart: ![baselines and bypasses](figures/findings_dumbbell_forest_beauty.png)