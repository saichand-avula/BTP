#!/usr/bin/env python3
"""Run the complete reproducible analysis and write machine-readable outputs."""
from pathlib import Path

import aux_fairness
import data_preprocessing
import layer1
import layer1_shap
import layer2_causal
import layer3_counterfactual
import layer_baseline_comparison
import layer_semisynthetic


def run_all():
    data_preprocessing.run()
    layer1.train()
    layer1_shap.run()
    causal  = layer2_causal.run()
    fairness = aux_fairness.run()
    recourse = layer3_counterfactual.run()
    baseline = layer_baseline_comparison.run()
    validation = layer_semisynthetic.run()
    print("\nCompleted. Results are in codebase/models/.")
    return causal, fairness, recourse


if __name__ == "__main__":
    run_all()
