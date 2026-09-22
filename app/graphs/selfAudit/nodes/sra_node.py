"""Self-Reflection Agent (SRA) node with HCD-EA.

The core contribution of SelfAudit. HCD-EA consists of:
1. Multi-Dimensional Confidence Decomposition:
   RFS = alpha * cos(Embed(q_k), Embed(E_k)) + (1-alpha) * LLM_judge(q_k, E_k)
   ESS = logical entailment from evidence to conclusion
   RCS = rule compliance check
   c_k = min(RFS, ESS, RCS)

2. Dependency-Aware Error Attribution:
   When c_k < tau, build G=(V,E), find topologically earliest failing node,
   re-execute only the affected sub-pipeline.

3. Heterogeneous Verification:
   Uses Qwen2.5-7B-Instruct for independent verification. When the
   heterogeneous verdict disagrees with the primary, triggers one
   additional reflection before finalizing.

Ablation modes (set via state["ablation_mode"]):
   "no_multidim" — single scalar confidence (1 LLM prompt) + blind re-execution
   "no_heterov"  — skip heterogeneous verification (loses disagreement safeguard)
"""

import numpy as np

from app.graphs.selfAudit.state.audit_state import AuditState
from app.schemas.audit_schema import Confidence3D, Verdict
from app.services.llm.prompts import (
    SRA_RFS_PROMPT,
    SRA_ESS_PROMPT,
    SRA_RCS_PROMPT,
    SRA_SINGLE_SCALAR_PROMPT,
    SRA_FINAL_VERDICT_PROMPT,
)
from app.utils.logger import logger
from app.utils.value_parser import ValueParser


def sra_node(state: AuditState) -> AuditState:
    """
    Run HCD-EA: confidence decomposition, error attribution, and
    heterogeneous verification.

    If c_k < tau and iteration < K: set reexecute_from to trigger
    selective pipeline re-execution.
    If c_k >= tau or iteration >= K: produce final verdict.
    """
    iteration = state.get("iteration", 0)
    settings = state.get("settings")
    tau = settings.HCD_TAU if settings else 0.75
    alpha = settings.HCD_ALPHA if settings else 0.5
    k_max = settings.HCD_K if settings else 3
    ablation_mode = state.get("ablation_mode", "")

    llm_service = state.get("llm_service")
    embedding_service = state.get("embedding_service")
    dependency_graph = state.get("dependency_graph")
    heterogeneous_verifier = state.get("heterogeneous_verifier")

    verification_results = state.get("verification_results", {})
    evidence_map = state.get("evidence_map", {})
    audit_points = state.get("audit_points", [])

    # ================================================================
    # BRANCH A: w/o Multi-Dim — single scalar confidence + blind re-execution
    # ================================================================
    if ablation_mode == "no_multidim":
        single_conf, tentative_label = _compute_single_scalar_confidence(
            state=state,
            verification_results=verification_results,
            llm_service=llm_service,
        )
        state["confidence_3d"] = Confidence3D(
            rfs=single_conf, ess=single_conf, rcs=single_conf
        )
        logger.info(
            f"SRA (no_multidim) iteration {iteration + 1}: "
            f"scalar_confidence={single_conf:.3f}"
        )

        if single_conf < tau and iteration + 1 < k_max:
            # Blind re-execution: no dimension → no selective attribution.
            # Re-run the full pipeline from TPA.
            state["reexecute_from"] = "tpa"
            state["iteration"] = iteration + 1
            logger.info(
                f"SRA (no_multidim) triggers blind re-execution from TPA "
                f"(iteration {iteration + 1}/{k_max})"
            )
        else:
            verdict = Verdict(
                final_label=tentative_label,
                confidence=single_conf,
                reasoning=f"Single scalar confidence: {single_conf:.3f}",
                iteration=iteration,
            )
            state["verdict"] = verdict
            state["reexecute_from"] = ""
            state["iteration"] = iteration + 1
        return state

    # ================================================================
    # BRANCH B: Full HCD-EA — 3D confidence + selective re-execution
    # ================================================================
    confidence_3d = _compute_confidence_3d(
        state=state,
        verification_results=verification_results,
        llm_service=llm_service,
        embedding_service=embedding_service,
        alpha=alpha,
    )
    state["confidence_3d"] = confidence_3d
    logger.info(
        f"SRA iteration {iteration + 1}: "
        f"RFS={confidence_3d.rfs:.3f}, ESS={confidence_3d.ess:.3f}, "
        f"RCS={confidence_3d.rcs:.3f}, combined={confidence_3d.combined:.3f}"
    )

    if confidence_3d.combined < tau and iteration + 1 < k_max:
        node_confidences = {
            "tpa": confidence_3d.rcs,
            "era": confidence_3d.rfs,
            "ia": confidence_3d.ess,
            "verdict": confidence_3d.combined,
        }
        attribution = dependency_graph.attribute_error(node_confidences, tau)
        state["error_attribution"] = attribution
        state["reexecute_from"] = attribution.source_node
        state["iteration"] = iteration + 1
        logger.info(
            f"SRA triggers selective re-execution from '{attribution.source_node}' "
            f"(iteration {iteration + 1}/{k_max})"
        )
    else:
        verdict = _produce_final_verdict(
            state=state,
            verification_results=verification_results,
            confidence_3d=confidence_3d,
            llm_service=llm_service,
        )
        state["verdict"] = verdict
        state["reexecute_from"] = ""

        # Heterogeneous verification: when enabled, an independent model
        # (Qwen2.5-7B) re-evaluates the verdict. If it disagrees, trigger
        # one additional IA re-execution for the primary model to reconsider.
        # The ablation (no_heterov) skips this safeguard entirely.
        if heterogeneous_verifier is not None and ablation_mode != "no_heterov":
            evidence_items = []
            audit_rules = []
            for key, results_list in verification_results.items():
                evidence = evidence_map.get(key, [])
                evidence_items.extend(evidence[:3])
                for ap in audit_points:
                    if f"{ap.objective}||{ap.audit_point}" == key:
                        audit_rules.append(ap.rule)
                        break

            rule_text = "; ".join(audit_rules[:3]) if audit_rules else "General compliance"
            heterogeneous_verdict = heterogeneous_verifier.verify(
                primary_verdict=verdict,
                evidence=evidence_items[:5],
                audit_rule=rule_text,
            )
            state["heterogeneous_verdict"] = heterogeneous_verdict
            agreement = 1.0 if heterogeneous_verdict.final_label == verdict.final_label else 0.0
            state["heterogeneous_agreement"] = agreement

            # Heterogeneous disagreement safeguard: when the independent
            # verifier disagrees (different label, non-trivial confidence),
            # give the primary model one more chance to re-inspect.
            if (
                agreement == 0.0
                and heterogeneous_verdict.confidence > 0.5
                and iteration + 1 < k_max
            ):
                logger.info(
                    f"SRA heterogeneous disagreement detected "
                    f"(primary={verdict.final_label}, hetero={heterogeneous_verdict.final_label})"
                    f" — triggering extra IA re-execution"
                )
                state["reexecute_from"] = "ia"
                state["iteration"] = iteration + 1
                return state

        state["iteration"] = iteration + 1

    return state


def _compute_single_scalar_confidence(
    state: AuditState,
    verification_results: dict,
    llm_service,
) -> tuple[float, str]:
    """Compute a single aggregate confidence score (ablation: w/o Multi-Dim).

    Uses ONE LLM prompt instead of decomposing into RFS/ESS/RCS.
    Returns (confidence, tentative_label).
    """
    audit_results_text = ""
    for ap in state.get("audit_points", []):
        key = f"{ap.objective}||{ap.audit_point}"
        results = verification_results.get(key, [])
        dim_results = "; ".join(
            f"{r.dimension}: {'Pass' if r.passed else 'Fail'} (conf={r.confidence:.2f})"
            for r in results
        )
        audit_results_text += (
            f"- Objective: {ap.objective}\n"
            f"  Rule: {ap.rule}\n"
            f"  Results: {dim_results}\n\n"
        )

    if llm_service:
        try:
            prompt = SRA_SINGLE_SCALAR_PROMPT.substitute(
                audit_results=audit_results_text[:3000],
            )
            response = llm_service.generate_completion(prompt)
            if response:
                data = ValueParser.extract_json_from_output(response)
                if data:
                    return (
                        float(data.get("confidence", 0.5)),
                        data.get("final_label", "Not Mentioned"),
                    )
        except Exception:
            pass

    # Fallback heuristic
    all_results = []
    for results in verification_results.values():
        all_results.extend(results)
    passed_count = sum(1 for r in all_results if r.passed)
    total = len(all_results) or 1
    heuristic_conf = passed_count / total
    label = "Entailment" if passed_count == total else (
        "Contradiction" if passed_count == 0 else "Not Mentioned"
    )
    return heuristic_conf, label


def _compute_confidence_3d(
    state: AuditState,
    verification_results: dict,
    llm_service,
    embedding_service,
    alpha: float,
) -> Confidence3D:
    """Compute the three-dimensional confidence vector (RFS, ESS, RCS)."""
    rfs_scores = []
    ess_scores = []
    rcs_scores = []

    audit_points = state.get("audit_points", [])
    evidence_map = state.get("evidence_map", {})

    for ap in audit_points:
        key = f"{ap.objective}||{ap.audit_point}"
        evidence = evidence_map.get(key, [])
        results = verification_results.get(key, [])

        # RFS: weighted combination of embedding cosine similarity + LLM judge
        rfs = _compute_rfs(ap, evidence, llm_service, embedding_service, alpha)
        rfs_scores.append(rfs)

        # ESS: logical entailment from evidence to conclusion
        ess = _compute_ess(ap, evidence, results, llm_service)
        ess_scores.append(ess)

        # RCS: rule compliance check
        rcs = _compute_rcs(ap, results, llm_service)
        rcs_scores.append(rcs)

    avg_rfs = float(np.mean(rfs_scores)) if rfs_scores else 0.0
    avg_ess = float(np.mean(ess_scores)) if ess_scores else 0.0
    avg_rcs = float(np.mean(rcs_scores)) if rcs_scores else 0.0

    return Confidence3D(rfs=avg_rfs, ess=avg_ess, rcs=avg_rcs)


def _compute_rfs(
    ap, evidence, llm_service, embedding_service, alpha: float
) -> float:
    """
    RFS = alpha * cos(Embed(q_k), Embed(E_k)) + (1-alpha) * LLM_judge(q_k, E_k)
    """
    if not evidence:
        return 0.0

    # Cosine similarity
    query_text = f"{ap.objective} {ap.audit_point}"
    evidence_text = " ".join(e.text for e in evidence[:3])

    cos_sim = 0.5
    if embedding_service:
        try:
            q_emb = np.array(embedding_service.get_single_embedding(query_text))
            e_emb = np.array(embedding_service.get_single_embedding(evidence_text))
            cos_sim = float(
                np.dot(q_emb, e_emb)
                / (np.linalg.norm(q_emb) * np.linalg.norm(e_emb) + 1e-8)
            )
        except Exception:
            pass

    # LLM judge
    llm_score = 0.5
    if llm_service:
        try:
            prompt = SRA_RFS_PROMPT.substitute(
                objective=ap.objective,
                evidence=evidence_text[:2000],
            )
            response = llm_service.generate_completion(prompt)
            if response:
                data = ValueParser.extract_json_from_output(response)
                if data:
                    llm_score = float(data.get("relevance_score", 0.5))
        except Exception:
            pass

    return alpha * cos_sim + (1 - alpha) * llm_score


def _compute_ess(ap, evidence, results, llm_service) -> float:
    """ESS: Logical entailment from evidence to conclusion."""
    if not results or not llm_service:
        return 0.5

    try:
        conclusion = "; ".join(
            f"{r.dimension}: {'Pass' if r.passed else 'Fail'}" for r in results
        )
        evidence_text = " ".join(e.text for e in evidence[:3]) if evidence else "None"

        prompt = SRA_ESS_PROMPT.substitute(
            conclusion=conclusion,
            evidence=evidence_text[:2000],
        )
        response = llm_service.generate_completion(prompt)
        if response:
            data = ValueParser.extract_json_from_output(response)
            if data:
                return float(data.get("support_score", 0.5))
    except Exception:
        pass

    return 0.5


def _compute_rcs(ap, results, llm_service) -> float:
    """RCS: Rule compliance check."""
    if not results or not llm_service:
        return 0.5

    try:
        conclusion = "; ".join(
            f"{r.dimension}: {'Pass' if r.passed else 'Fail'}" for r in results
        )

        prompt = SRA_RCS_PROMPT.substitute(
            rule=ap.rule,
            conclusion=conclusion,
        )
        response = llm_service.generate_completion(prompt)
        if response:
            data = ValueParser.extract_json_from_output(response)
            if data:
                return float(data.get("compliance_score", 0.5))
    except Exception:
        pass

    return 0.5


def _produce_final_verdict(
    state: AuditState,
    verification_results: dict,
    confidence_3d: Confidence3D,
    llm_service,
) -> Verdict:
    """Synthesize all findings into a final audit verdict."""
    audit_results_text = ""
    for ap in state.get("audit_points", []):
        key = f"{ap.objective}||{ap.audit_point}"
        results = verification_results.get(key, [])
        dim_results = "; ".join(
            f"{r.dimension}: {'Pass' if r.passed else 'Fail'} (conf={r.confidence:.2f})"
            for r in results
        )
        audit_results_text += (
            f"- Objective: {ap.objective}\n"
            f"  Rule: {ap.rule}\n"
            f"  Results: {dim_results}\n\n"
        )

    if llm_service:
        try:
            prompt = SRA_FINAL_VERDICT_PROMPT.substitute(
                audit_results=audit_results_text[:3000],
                rfs=f"{confidence_3d.rfs:.3f}",
                ess=f"{confidence_3d.ess:.3f}",
                rcs=f"{confidence_3d.rcs:.3f}",
                combined=f"{confidence_3d.combined:.3f}",
            )
            response = llm_service.generate_completion(prompt)
            if response:
                data = ValueParser.extract_json_from_output(response)
                if data:
                    return Verdict(
                        final_label=data.get("final_label", "Not Mentioned"),
                        confidence=float(data.get("confidence", confidence_3d.combined)),
                        reasoning=data.get("reasoning", ""),
                        iteration=state.get("iteration", 0),
                    )
        except Exception:
            pass

    # Fallback: heuristic verdict
    all_results = []
    for results in verification_results.values():
        all_results.extend(results)
    passed_count = sum(1 for r in all_results if r.passed)
    total = len(all_results) or 1

    if passed_count == total:
        label = "Entailment"
    elif passed_count == 0:
        label = "Contradiction"
    else:
        label = "Not Mentioned"

    return Verdict(
        final_label=label,
        confidence=confidence_3d.combined,
        reasoning=f"Heuristic: {passed_count}/{total} dimensions passed, "
        f"RFS={confidence_3d.rfs:.2f}, ESS={confidence_3d.ess:.2f}, RCS={confidence_3d.rcs:.2f}",
        iteration=state.get("iteration", 0),
    )