"""Parameter sensitivity experiment.

Sweeps the confidence threshold tau (0.5 to 0.95) and max reflection
iterations K (1 to 5) on ContractNLI to measure impact on weighted F1.
"""

import numpy as np
from tqdm import tqdm

from app.core.config import Settings
from app.utils.dataset_utils import load_contractnli
from app.utils.logger import logger
from app.utils.metrics import compute_weighted_f1
from experiments.base_experiment import BaseExperiment


class SensitivityExperiment(BaseExperiment):
    """Parameter sensitivity analysis for tau and K."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.samples = []

    def run(self) -> dict:
        self.log_run_info()

        dataset_path = self.settings.DATASET_PATH + "/contractnli"
        self.samples = load_contractnli(dataset_path, split="test", max_samples=100)
        logger.info(f"Running sensitivity on {len(self.samples)} samples")

        results = {}

        # Tau sweep (K=3 fixed)
        tau_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        tau_results = {}
        for tau in tau_values:
            logger.info(f"--- tau = {tau:.2f} ---")
            # Actually change the threshold for this sweep
            self.settings.HCD_TAU = tau
            self.settings.HCD_K = 3
            preds, golds = [], []
            for sample in tqdm(self.samples, desc=f"  tau={tau:.2f}"):
                result = self._run_selfaudit(
                    query=sample["hypothesis"],
                    document_text=sample["nda_text"],
                )
                preds.append(result.get("final_label", "Not Mentioned"))
                golds.append(sample["label"])
            tau_results[f"tau_{tau:.2f}"] = {
                "tau": tau,
                "F1[W]": compute_weighted_f1(preds, golds),
            }
            logger.info(f"  tau={tau:.2f}: F1[W]={tau_results[f'tau_{tau:.2f}']['F1[W]']:.3f}")
        results["tau_sweep"] = tau_results

        # K sweep (tau=0.75 fixed)
        k_values = [1, 2, 3, 4, 5]
        k_results = {}
        for k in k_values:
            logger.info(f"--- K = {k} ---")
            # Actually change K for this sweep
            self.settings.HCD_K = k
            self.settings.HCD_TAU = 0.75
            preds, golds = [], []
            for sample in tqdm(self.samples, desc=f"  K={k}"):
                result = self._run_selfaudit(
                    query=sample["hypothesis"],
                    document_text=sample["nda_text"],
                )
                preds.append(result.get("final_label", "Not Mentioned"))
                golds.append(sample["label"])
            k_results[f"K_{k}"] = {
                "K": k,
                "F1[W]": compute_weighted_f1(preds, golds),
            }
            logger.info(f"  K={k}: F1[W]={k_results[f'K_{k}']['F1[W]']:.3f}")
        results["k_sweep"] = k_results

        self.save_results(results, "sensitivity_results.json")
        return results