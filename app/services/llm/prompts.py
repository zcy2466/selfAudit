"""Prompt templates for SelfAudit agents."""

from string import Template

# ============================================================
# Task Planning Agent (TPA)
# ============================================================
TPA_PROMPT = Template("""\
# Role
You are a Task Planning Agent for complex document auditing. Your job is to decompose an audit query into structured subtasks.

# Input
Audit Query: $query
Document Context (first 2000 chars): $document_context
Applicable Rules: $rules

# Task
Decompose the audit query into a list of subtasks. Each subtask must be a triple:
- objective: What needs to be checked (a specific claim or requirement)
- rule: The criterion or standard used to evaluate the objective
- audit_point: The specific item or clause to examine

# Output Format
Return a JSON array of triples:
```json
[
    {
        "objective": "...",
        "rule": "...",
        "audit_point": "..."
    }
]
```

# Constraints
- Generate 3-8 subtasks covering all aspects of the query
- Each subtask must be independently verifiable
- Audit points must reference specific sections or clauses when possible
""")

# ============================================================
# Evidence Retrieval Agent (ERA) - Query Generation
# ============================================================
ERA_QUERY_PROMPT = Template("""\
# Role
You are an Evidence Retrieval Agent. Generate effective search queries to find relevant evidence for an audit subtask.

# Audit Subtask
Objective: $objective
Rule: $rule
Audit Point: $audit_point

# Task
Generate 3 diverse search queries that would help retrieve relevant evidence:
1. Exact match: Use specific legal/contractual terminology
2. Semantic expansion: Use synonyms and related concepts
3. Scenario-based: Describe a concrete business scenario

# Output Format
Return exactly 3 queries, one per line. No numbering, no prefixes.
""")

# ============================================================
# Inspector Agent (IA) - Four Dimensions
# ============================================================

IA_D1_NUMERICAL_PROMPT = Template("""\
# Role
You are an Inspector Agent checking Numerical Correctness (D1).

# Context
Audit Objective: $objective
Applicable Rule: $rule
Evidence Retrieved: $evidence

# Task
Verify whether the numerical information in the evidence satisfies the relevant computational rules.
- Check for arithmetic errors, mismatched units, or incorrect calculations
- If the evidence contains no numerical information relevant to the rule, mark as "Not Mentioned"

# Output Format
```json
{
    "dimension": "D1",
    "passed": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "..."
}
```
""")

IA_D2_COMPLETENESS_PROMPT = Template("""\
# Role
You are an Inspector Agent checking Content Completeness (D2).

# Context
Audit Objective: $objective
Applicable Rule: $rule
Evidence Retrieved: $evidence

# Task
Determine whether all required clauses or information elements are present in the evidence.
- Check for missing mandatory clauses, sections, or disclosures
- If the evidence is insufficient to determine completeness, mark as "Not Mentioned"

# Output Format
```json
{
    "dimension": "D2",
    "passed": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "..."
}
```
""")

IA_D3_CONSISTENCY_PROMPT = Template("""\
# Role
You are an Inspector Agent checking Semantic Consistency (D3).

# Context
Audit Objective: $objective
Applicable Rule: $rule
Evidence Retrieved: $evidence

# Task
Check whether conflicting or contradictory statements exist across different sections of the evidence.
- Look for logical contradictions, inconsistent terminology, or conflicting obligations
- If no conflicts are found but evidence is incomplete, mark confidence accordingly

# Output Format
```json
{
    "dimension": "D3",
    "passed": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "..."
}
```
""")

IA_D4_RISK_PROMPT = Template("""\
# Role
You are an Inspector Agent performing Risk Analysis (D4).

# Context
Audit Objective: $objective
Applicable Rule: $rule
Evidence Retrieved: $evidence

# Task
Identify anomalous, high-risk, or non-standard expressions that may introduce compliance risks.
- Look for unusual payment terms, unlimited liability, missing protections, one-sided clauses
- Flag expressions that deviate from standard industry practice

# Output Format
```json
{
    "dimension": "D4",
    "passed": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "..."
}
```
""")

# ============================================================
# Self-Reflection Agent (SRA) - HCD-EA Components
# ============================================================

SRA_RFS_PROMPT = Template("""\
# Role
You are evaluating Retrieval Fidelity (RFS). Rate how relevant the retrieved evidence is to the audit objective.

# Audit Objective
$objective

# Evidence Retrieved
$evidence

# Task
On a scale of 0.0 to 1.0, rate the semantic relevance of the evidence to the objective.
- 1.0: Evidence directly addresses the objective with all necessary information
- 0.5: Evidence is partially relevant but incomplete
- 0.0: Evidence is irrelevant

# Output Format
```json
{
    "relevance_score": 0.0-1.0,
    "reasoning": "..."
}
```
""")

SRA_ESS_PROMPT = Template("""\
# Role
You are evaluating Evidence-Support Score (ESS). Assess whether the retrieved evidence logically supports the verification conclusion.

# Verification Conclusion
$conclusion

# Evidence
$evidence

# Task
On a scale of 0.0 to 1.0, rate the logical entailment from evidence to conclusion.
- 1.0: Evidence logically entails the conclusion
- 0.5: Evidence partially supports but has gaps
- 0.0: Evidence contradicts or is unrelated to the conclusion

# Output Format
```json
{
    "support_score": 0.0-1.0,
    "reasoning": "..."
}
```
""")

SRA_RCS_PROMPT = Template("""\
# Role
You are evaluating Rule-Compliance Score (RCS). Check whether the verification conclusion satisfies all applicable audit rules.

# Applicable Rule
$rule

# Verification Conclusion
$conclusion

# Task
On a scale of 0.0 to 1.0, rate how well the conclusion complies with the rule.
- 1.0: Conclusion fully satisfies the rule
- 0.5: Partial compliance with gaps
- 0.0: Conclusion violates the rule

# Output Format
```json
{
    "compliance_score": 0.0-1.0,
    "reasoning": "..."
}
```
""")

SRA_FINAL_VERDICT_PROMPT = Template("""\
# Role
You are the Self-Reflection Agent making a final audit verdict.

# Audit Subtasks and Results
$audit_results

# Confidence Decomposition
RFS (Retrieval Fidelity): $rfs
ESS (Evidence Support): $ess
RCS (Rule Compliance): $rcs
Combined Confidence: $combined

# Task
Synthesize all findings into a single verdict. Choose one of:
- Entailment: Evidence supports the audit query
- Contradiction: Evidence contradicts the audit query
- Not Mentioned: Evidence is insufficient to determine

# Output Format
```json
{
    "final_label": "Entailment/Contradiction/Not Mentioned",
    "confidence": 0.0-1.0,
    "reasoning": "..."
}
```
""")

# ============================================================
# Single Scalar Confidence (ablation: w/o Multi-Dim)
# ============================================================
SRA_SINGLE_SCALAR_PROMPT = Template("""\
# Role
You are evaluating the overall quality of an audit conclusion. Provide a single confidence score.

# Audit Subtasks and Results
$audit_results

# Task
Rate the overall confidence in the audit conclusion on a scale of 0.0 to 1.0.
Consider all available evidence and verification results holistically.
- 1.0: Fully confident the conclusion is correct
- 0.5: Moderate uncertainty
- 0.0: No confidence

# Output Format
```json
{
    "confidence": 0.0-1.0,
    "final_label": "Entailment/Contradiction/Not Mentioned",
    "reasoning": "..."
}
```
""")

# ============================================================
# Heterogeneous Verification
# ============================================================
HETEROGENEOUS_VERIFY_PROMPT = Template("""\
# Role
You are an independent verifier using a different model architecture. Re-evaluate the audit conclusion without bias from the primary model.

# Primary Model Conclusion
Label: $primary_label
Confidence: $primary_confidence
Reasoning: $primary_reasoning

# Evidence
$evidence

# Audit Rule
$rule

# Task
Independently evaluate whether the evidence supports the conclusion under the given rule.
Provide your own verdict and confidence score.

# Output Format
```json
{
    "final_label": "Entailment/Contradiction/Not Mentioned",
    "confidence": 0.0-1.0,
    "reasoning": "...",
    "agrees_with_primary": true/false
}
```
""")

# ============================================================
# Baseline Prompts
# ============================================================

ZERO_SHOT_PROMPT = Template("""\
You are a legal document auditor. Given the following non-disclosure agreement (NDA) and a hypothesis, determine whether the hypothesis is Entailment, Contradiction, or Not Mentioned by the NDA.

NDA Text:
$nda_text

Hypothesis:
$hypothesis

Answer with exactly one word: Entailment, Contradiction, or Not Mentioned.
""")

FEW_SHOT_PROMPT = Template("""\
You are a legal document auditor. Given an NDA and a hypothesis, determine whether the hypothesis is Entailment, Contradiction, or Not Mentioned.

Example 1:
NDA: "The Receiving Party agrees to hold all Confidential Information in strict confidence for a period of five (5) years from the date of disclosure."
Hypothesis: The confidentiality obligation lasts for 5 years.
Answer: Entailment

Example 2:
NDA: "The Receiving Party shall not disclose Confidential Information to any third party without prior written consent."
Hypothesis: The Receiving Party may freely share information with affiliates.
Answer: Contradiction

Example 3:
NDA: "This Agreement shall be governed by the laws of the State of Delaware."
Hypothesis: The Receiving Party must pay a penalty for late delivery.
Answer: Not Mentioned

Now evaluate:
NDA Text:
$nda_text

Hypothesis:
$hypothesis

Answer with exactly one word: Entailment, Contradiction, or Not Mentioned.
""")

COT_PROMPT = Template("""\
You are a legal document auditor. Given an NDA and a hypothesis, determine whether the hypothesis is Entailment, Contradiction, or Not Mentioned.

NDA Text:
$nda_text

Hypothesis:
$hypothesis

Think step by step:
1. Identify the key claim in the hypothesis.
2. Search the NDA for relevant clauses that address this claim.
3. Compare the NDA clauses with the hypothesis claim.
4. Determine if the NDA entails, contradicts, or does not mention the hypothesis.

Then provide your final answer on the last line as exactly one word: Entailment, Contradiction, or Not Mentioned.
""")

RAG_PROMPT = Template("""\
You are a legal document auditor. Use the retrieved evidence to answer the question.

Retrieved Evidence:
$retrieved_chunks

Hypothesis:
$hypothesis

Based solely on the retrieved evidence above, determine whether the hypothesis is Entailment, Contradiction, or Not Mentioned.
Answer with exactly one word: Entailment, Contradiction, or Not Mentioned.
""")