"""Cross-model robustness experiment: Table 5.

Tests SelfAudit with four different backbone LLMs:
GPT-4o, Qwen2.5-72B, Llama-3.3-70B, DeepSeek-V3.

The heterogeneous verifier stays fixed (Qwen2.5-7B-Instruct).
"""

import numpy as np
from tqdm import tqdm

from app.core.config import Settings
from app.services.llm.factory import CompletionServiceFactory
from app.utils.dataset_utils import load_contractnli
from app.utils.logger import logger
from app.utils.metrics import compute_accuracy, compute_weighted_f1, compute_cv
from experiments.base_experiment import BaseExperiment


class RobustnessExperiment(BaseExperiment):
    """Cross-model robustness experiment (Table 5)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.samples = []

    def run(self) -> dict:
        self.log_run_info()

        dataset_path = self.settings.DATASET_PATH + "/contractnli"
        self.samples = load_contractnli(dataset_path, split="test", max_samples=200)
        logger.info(f"Running robustness on {len(self.samples)} samples")

        backbones = self.settings.BACKBONE_LIST
        results = {}

        for backbone in backbones:
            logger.info(f"--- Backbone: {backbone} ---")

            # Zero-shot baseline
            zs_llm = CompletionServiceFactory.create_service(self.settings, model_name=backbone)
            zs_preds, golds = [], []
            for sample in tqdm(self.samples, desc=f"  ZS {backbone}"):
                from app.services.llm.prompts import ZERO_SHOT_PROMPT
                prompt = ZERO_SHOT_PROMPT.substitute(
                    nda_text=sample["nda_text"],
                    hypothesis=sample["hypothesis"],
                )
                response = zs_llm.generate_completion(prompt)
                zs_preds.append(self._parse_label(response))
                golds.append(sample["label"])

            zs_acc = compute_accuracy(zs_preds, golds)
            zs_f1 = compute_weighted_f1(zs_preds, golds)

            # SelfAudit with this backbone
            # For non-OpenAI backbones, use appropriate API endpoint
            orig_llm = self.llm_service
            if "qwen" in backbone.lower():
                from app.services.llm.llm_service import OpenAIService
                import copy
                qwen_settings = copy.deepcopy(self.settings)
                qwen_settings.LLM_MODEL = backbone
                qwen_settings.LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                qwen_settings.LLM_API_KEY = self.settings.HETEROGENEOUS_LLM_API_KEY
                self.llm_service = OpenAIService(qwen_settings)
            elif "llama" in backbone.lower():
                from app.services.llm.llm_service import OpenAIService
                import copy
                llama_settings = copy.deepcopy(self.settings)
                llama_settings.LLM_MODEL = backbone
                self.llm_service = OpenAIService(llama_settings)
            elif "deepseek" in backbone.lower():
                from app.services.llm.llm_service import OpenAIService
                import copy
                ds_settings = copy.deepcopy(self.settings)
                ds_settings.LLM_MODEL = backbone
                ds_settings.LLM_BASE_URL = "https://api.deepseek.com/v1"
                self.llm_service = OpenAIService(ds_settings)
            else:
                self.llm_service = zs_llm

            sa_preds = []
            for sample in tqdm(self.samples, desc=f"  SelfAudit {backbone}"):
                result = self._run_selfaudit(
                    query=sample["hypothesis"],
                    document_text=sample["nda_text"],
                )
                sa_preds.append(result.get("final_label", "Not Mentioned"))

            sa_acc = compute_accuracy(sa_preds, golds)
            sa_f1 = compute_weighted_f1(sa_preds, golds)

            self.llm_service = orig_llm

            results[backbone] = {
                "ZS_Acc": zs_acc,
                "ZS_F1W": zs_f1,
                "SelfAudit_Acc": sa_acc,
                "SelfAudit_F1W": sa_f1,
            }
            logger.info(f"  {backbone}: ZS Acc={zs_acc:.3f}, SA Acc={sa_acc:.3f}")

        # Compute averages and CV
        zs_accs = [results[b]["ZS_Acc"] for b in backbones]
        zs_f1s = [results[b]["ZS_F1W"] for b in backbones]
        sa_accs = [results[b]["SelfAudit_Acc"] for b in backbones]
        sa_f1s = [results[b]["SelfAudit_F1W"] for b in backbones]

        results["Average_ZS"] = {"Acc": float(np.mean(zs_accs)), "F1[W]": float(np.mean(zs_f1s))}
        results["Average_SelfAudit"] = {"Acc": float(np.mean(sa_accs)), "F1[W]": float(np.mean(sa_f1s))}
        results["CV_ZS"] = {"Acc": compute_cv(zs_accs), "F1[W]": compute_cv(zs_f1s)}
        results["CV_SelfAudit"] = {"Acc": compute_cv(sa_accs), "F1[W]": compute_cv(sa_f1s)}

        self.save_results(results, "robustness_results.json")
        return results

    @staticmethod
    def _parse_label(response: str | None) -> str:
        if response is None:
            return "Not Mentioned"
        r = response.strip().lower()
        if "entailment" in r:
            return "Entailment"
        if "contradiction" in r:
            return "Contradiction"
        return "Not Mentioned"