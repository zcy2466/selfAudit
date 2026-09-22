"""ContractNLI experiment: Table 1 (main results) + 13 baselines.

Evaluates SelfAudit on the ContractNLI test set (2,091 samples).
Each sample: NDA text + hypothesis -> Entailment/Contradiction/Not Mentioned.

Metrics: Acc, F1[W] (weighted), F1[E], F1[C], F1[N]
"""

from tqdm import tqdm

from app.core.config import Settings
from app.services.llm.prompts import (
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
    COT_PROMPT,
    RAG_PROMPT,
)
from app.utils.dataset_utils import load_contractnli
from app.utils.logger import logger
from app.utils.metrics import (
    compute_accuracy,
    compute_weighted_f1,
    compute_class_f1,
    compute_f1,
)
from experiments.base_experiment import BaseExperiment


class ContractNLIExperiment(BaseExperiment):
    """ContractNLI benchmark experiment (Table 1)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.samples = []

    def run(self) -> dict:
        """Run the full ContractNLI experiment with all baselines."""
        self.log_run_info()

        dataset_path = self.settings.DATASET_PATH + "/contractnli"
        self.samples = load_contractnli(dataset_path, split="test")
        logger.info(f"Running ContractNLI on {len(self.samples)} samples")

        methods = [
            ("ZS", self._run_zs),
            ("FS", self._run_fs),
            ("CoT", self._run_cot),
            ("Self-Consistency", self._run_self_consistency),
            ("ReAct", self._run_react),
            ("Standard RAG", self._run_standard_rag),
            ("Self-RAG", self._run_self_rag),
            ("Reflexion", self._run_reflexion),
            ("Self-Refine", self._run_self_refine),
            ("CRITIC", self._run_critic),
            ("AutoGen", self._run_autogen),
            ("MetaGPT", self._run_metagpt),
            ("PAKTON", self._run_pakton),
            ("SelfAudit", self._run_selfaudit_wrapper),
        ]

        results_table = {}
        for method_name, method_fn in methods:
            logger.info(f"--- Running {method_name} ---")
            predictions, gold_labels = self._evaluate_method(method_fn)
            results_table[method_name] = self._compute_metrics(predictions, gold_labels)
            logger.info(
                f"{method_name}: Acc={results_table[method_name]['Acc']:.3f}, "
                f"F1[W]={results_table[method_name]['F1[W]']:.3f}"
            )

        self.save_results(results_table, "contractnli_results.json")
        return results_table

    def _evaluate_method(self, method_fn) -> tuple[list[str], list[str]]:
        """Run a method on all samples and collect predictions."""
        predictions = []
        gold_labels = []
        for sample in tqdm(self.samples, desc="Evaluating"):
            pred = method_fn(sample)
            predictions.append(pred)
            gold_labels.append(sample["label"])
        return predictions, gold_labels

    def _compute_metrics(self, predictions: list[str], gold_labels: list[str]) -> dict:
        """Compute all ContractNLI metrics."""
        return {
            "Acc": compute_accuracy(predictions, gold_labels),
            "F1[W]": compute_weighted_f1(predictions, gold_labels),
            "F1[E]": compute_class_f1(predictions, gold_labels, "Entailment"),
            "F1[C]": compute_class_f1(predictions, gold_labels, "Contradiction"),
            "F1[N]": compute_class_f1(predictions, gold_labels, "Not Mentioned"),
        }

    # ================================================================
    # SelfAudit
    # ================================================================
    def _run_selfaudit_wrapper(self, sample: dict) -> str:
        result = self._run_selfaudit(
            query=sample["hypothesis"],
            document_text=sample["nda_text"],
        )
        return result.get("final_label", "Not Mentioned")

    # ================================================================
    # Prompting Baselines
    # ================================================================
    def _run_zs(self, sample: dict) -> str:
        prompt = ZERO_SHOT_PROMPT.substitute(
            nda_text=sample["nda_text"],
            hypothesis=sample["hypothesis"],
        )
        response = self.llm_service.generate_completion(prompt)
        return self._parse_label(response)

    def _run_fs(self, sample: dict) -> str:
        prompt = FEW_SHOT_PROMPT.substitute(
            nda_text=sample["nda_text"],
            hypothesis=sample["hypothesis"],
        )
        response = self.llm_service.generate_completion(prompt)
        return self._parse_label(response)

    # ================================================================
    # Reasoning Baselines
    # ================================================================
    def _run_cot(self, sample: dict) -> str:
        prompt = COT_PROMPT.substitute(
            nda_text=sample["nda_text"],
            hypothesis=sample["hypothesis"],
        )
        response = self.llm_service.generate_completion(prompt)
        return self._parse_label(response)

    def _run_self_consistency(self, sample: dict, n: int = 5) -> str:
        """Self-Consistency: majority vote over n reasoning paths."""
        votes = []
        for _ in range(n):
            pred = self._run_cot(sample)
            votes.append(pred)
        return max(set(votes), key=votes.count)

    def _run_react(self, sample: dict) -> str:
        """ReAct: interleaved reasoning, action, and observation.

        Thought → Action (search) → Observation → Thought → ... → Final Answer.
        """
        thought_prompt = (
            f"You are analyzing a legal NDA. Think about what evidence you need to "
            f"verify this hypothesis: '{sample['hypothesis']}'. "
            f"Write a specific search query to find the relevant clause in the NDA."
        )
        thought = self.llm_service.generate_completion(thought_prompt) or ""

        action = thought.strip() if thought else sample["hypothesis"]

        chunks = self._simple_retrieve(action, sample["nda_text"], k=3)
        observation = "\n".join(chunks)

        final_prompt = (
            f"NDA Excerpts:\n{observation}\n\n"
            f"Hypothesis: {sample['hypothesis']}\n\n"
            f"Based on the excerpts above, answer with exactly one word: "
            f"Entailment, Contradiction, or Not Mentioned."
        )
        response = self.llm_service.generate_completion(final_prompt)
        return self._parse_label(response)

    # ================================================================
    # Retrieval Baselines
    # ================================================================
    def _run_standard_rag(self, sample: dict) -> str:
        """Standard RAG: retrieve then answer."""
        chunks = self._simple_retrieve(sample["hypothesis"], sample["nda_text"])
        chunks_text = "\n\n".join(chunks[:5])
        prompt = RAG_PROMPT.substitute(
            retrieved_chunks=chunks_text,
            hypothesis=sample["hypothesis"],
        )
        response = self.llm_service.generate_completion(prompt)
        return self._parse_label(response)

    def _run_self_rag(self, sample: dict) -> str:
        """Self-RAG: retrieve, critique relevance, re-retrieve if needed."""
        # Initial retrieval
        chunks = self._simple_retrieve(sample["hypothesis"], sample["nda_text"], k=5)
        chunks_text = "\n\n".join(chunks)

        # Self-reflection on retrieval quality
        critique_prompt = (
            f"Hypothesis: {sample['hypothesis']}\n\n"
            f"Retrieved excerpts:\n{chunks_text}\n\n"
            f"Are these excerpts relevant and sufficient to determine whether the "
            f"hypothesis is Entailment, Contradiction, or Not Mentioned? "
            f"Answer YES or NO."
        )
        critique = self.llm_service.generate_completion(critique_prompt)
        is_relevant = critique and "yes" in critique.strip().lower()

        if not is_relevant:
            refined_query = (
                f"Find specific legal clauses related to: {sample['hypothesis']}"
            )
            chunks2 = self._simple_retrieve(refined_query, sample["nda_text"], k=5)
            chunks_text = "\n\n".join(chunks2)

        prompt = RAG_PROMPT.substitute(
            retrieved_chunks=chunks_text,
            hypothesis=sample["hypothesis"],
        )
        response = self.llm_service.generate_completion(prompt)
        return self._parse_label(response)

    # ================================================================
    # Reflection Baselines
    # ================================================================
    def _run_reflexion(self, sample: dict) -> str:
        """Reflexion: iterative self-reflection with episodic memory.

        A single model generates an answer, reflects on potential errors,
        and retries with self-feedback. Uses up to 2 reflection iterations.
        """
        pred = self._run_zs(sample)

        reflection_prompt = (
            f"Your previous answer for this hypothesis was: '{pred}'\n\n"
            f"Hypothesis: {sample['hypothesis']}\n"
            f"NDA (first 3000 chars): {sample['nda_text'][:3000]}\n\n"
            f"Reflect: Did you miss any relevant clause? Did you misinterpret "
            f"the hypothesis? What is the correct answer? "
            f"Output only one word: Entailment, Contradiction, or Not Mentioned."
        )
        refined = self.llm_service.generate_completion(reflection_prompt)
        if refined:
            refined_label = self._parse_label(refined)
            # Second reflection with accumulated feedback
            reflection2_prompt = (
                f"Initial answer: '{pred}'. After reflection: '{refined_label}'.\n"
                f"Hypothesis: {sample['hypothesis']}\n\n"
                f"Do a final review. Output only one word: "
                f"Entailment, Contradiction, or Not Mentioned."
            )
            final_response = self.llm_service.generate_completion(reflection2_prompt)
            return self._parse_label(final_response) if final_response else refined_label
        return pred

    def _run_self_refine(self, sample: dict) -> str:
        pred1 = self._run_zs(sample)
        prompt = (
            f"You previously answered '{pred1}' for this hypothesis:\n"
            f"{sample['hypothesis']}\n\n"
            f"Review the NDA text and refine your answer. Output only one word: "
            f"Entailment, Contradiction, or Not Mentioned.\n\n"
            f"NDA: {sample['nda_text'][:3000]}"
        )
        response = self.llm_service.generate_completion(prompt)
        return self._parse_label(response) if response else pred1

    def _run_critic(self, sample: dict) -> str:
        pred1 = self._run_zs(sample)
        prompt = (
            f"Critique this answer: The hypothesis '{sample['hypothesis']}' was "
            f"classified as '{pred1}'. Review for errors and provide the corrected "
            f"label as one word: Entailment, Contradiction, or Not Mentioned."
        )
        response = self.llm_service.generate_completion(prompt)
        return self._parse_label(response) if response else pred1

    # ================================================================
    # Multi-Agent Baselines (simplified)
    # ================================================================
    def _run_autogen(self, sample: dict) -> str:
        """AutoGen: two-agent conversation (planner + executor).

        Planner agent decomposes the task, executor agent carries it out,
        then planner reviews and produces the final answer.
        """
        planner_prompt = (
            f"As a legal audit planner, analyze this hypothesis and create a plan:\n"
            f"'{sample['hypothesis']}'\n\n"
            f"Output: 1) What clauses to look for 2) How to verify."
        )
        plan = self.llm_service.generate_completion(planner_prompt) or ""

        executor_prompt = (
            f"Plan: {plan}\n\n"
            f"NDA (first 4000 chars): {sample['nda_text'][:4000]}\n\n"
            f"Execute the plan. Output your finding as exactly one word: "
            f"Entailment, Contradiction, or Not Mentioned."
        )
        executor_result = self.llm_service.generate_completion(executor_prompt)
        executor_label = self._parse_label(executor_result) if executor_result else "Not Mentioned"

        reviewer_prompt = (
            f"Planner's plan:\n{plan}\n\n"
            f"Executor's finding: {executor_label}\n"
            f"Hypothesis: {sample['hypothesis']}\n\n"
            f"As the planner, review the executor's finding. Do you agree? "
            f"Output the final answer as exactly one word: "
            f"Entailment, Contradiction, or Not Mentioned."
        )
        final = self.llm_service.generate_completion(reviewer_prompt)
        return self._parse_label(final) if final else executor_label

    def _run_metagpt(self, sample: dict) -> str:
        """MetaGPT: structured multi-agent with product manager, architect, engineer.

        PM defines requirements → Architect designs the analysis → Engineer executes.
        """
        pm_prompt = (
            f"As a Product Manager, define the requirement for auditing this hypothesis:\n"
            f"'{sample['hypothesis']}'\n\n"
            f"Output: 1) What must be verified 2) Acceptance criteria."
        )
        requirement = self.llm_service.generate_completion(pm_prompt) or ""

        architect_prompt = (
            f"Requirement: {requirement}\n\n"
            f"Design the analysis approach for verifying this hypothesis against "
            f"an NDA. Specify which sections and legal concepts to examine."
        )
        design = self.llm_service.generate_completion(architect_prompt) or ""

        engineer_prompt = (
            f"Design: {design}\n\n"
            f"NDA: {sample['nda_text'][:4000]}\n\n"
            f"Hypothesis: {sample['hypothesis']}\n\n"
            f"As the engineer, execute the analysis. Output exactly one word: "
            f"Entailment, Contradiction, or Not Mentioned."
        )
        response = self.llm_service.generate_completion(engineer_prompt)
        return self._parse_label(response) if response else "Not Mentioned"

    def _run_pakton(self, sample: dict) -> str:
        """PAKTON: multi-agent orchestration with RAG for long-form legal text.

        Orchestrator + Retriever + Reasoner. The orchestrator coordinates
        retrieval and reasoning with explicit evidence grounding.
        """
        chunks = self._simple_retrieve(sample["hypothesis"], sample["nda_text"], k=5)
        evidence_text = "\n\n".join(chunks)

        orchestrator_prompt = (
            f"As an audit orchestrator, analyze this hypothesis:\n"
            f"'{sample['hypothesis']}'\n\n"
            f"Evidence from NDA:\n{evidence_text}\n\n"
            f"First, decompose the hypothesis into sub-claims.\n"
            f"Then, for each sub-claim, check if the evidence supports or contradicts it.\n"
            f"Finally, synthesize: Entailment, Contradiction, or Not Mentioned?\n"
            f"Output only the final label as exactly one word."
        )
        response = self.llm_service.generate_completion(orchestrator_prompt)
        return self._parse_label(response) if response else "Not Mentioned"

    # ================================================================
    # Helpers
    # ================================================================
    def _simple_retrieve(self, query: str, document: str, k: int = 5) -> list[str]:
        """Simple paragraph-based retrieval."""
        query_emb = self.embedding_service.get_single_embedding(query)
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
        if not paragraphs:
            return [document]

        import numpy as np
        para_embs = self.embedding_service.get_embeddings(paragraphs)
        query_vec = np.array(query_emb)
        scores = []
        for i, emb in enumerate(para_embs):
            vec = np.array(emb)
            sim = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8))
            scores.append((sim, paragraphs[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scores[:k]]

    @staticmethod
    def _parse_label(response: str | None) -> str:
        """Extract label from LLM response."""
        if response is None:
            return "Not Mentioned"
        response_lower = response.strip().lower()
        if "entailment" in response_lower:
            return "Entailment"
        if "contradiction" in response_lower:
            return "Contradiction"
        return "Not Mentioned"