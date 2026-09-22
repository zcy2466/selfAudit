"""Tests for SelfAudit agents."""

import pytest


class TestTPANode:
    def test_audit_point_creation(self):
        from app.schemas.audit_schema import AuditPoint
        ap = AuditPoint(
            objective="Check confidentiality period",
            rule="Confidentiality must not exceed 5 years",
            audit_point="Section 3.1",
        )
        assert ap.objective == "Check confidentiality period"
        assert ap.rule == "Confidentiality must not exceed 5 years"
        assert ap.audit_point == "Section 3.1"

    def test_empty_audit_point(self):
        from app.schemas.audit_schema import AuditPoint
        ap = AuditPoint(objective="", rule="", audit_point="")
        assert ap.objective == ""


class TestConfidence3D:
    def test_combined_confidence(self):
        from app.schemas.audit_schema import Confidence3D
        c = Confidence3D(rfs=0.9, ess=0.8, rcs=0.7)
        assert c.combined == 0.7

    def test_perfect_confidence(self):
        from app.schemas.audit_schema import Confidence3D
        c = Confidence3D(rfs=1.0, ess=1.0, rcs=1.0)
        assert c.combined == 1.0

    def test_to_dict(self):
        from app.schemas.audit_schema import Confidence3D
        c = Confidence3D(rfs=0.9, ess=0.8, rcs=0.7)
        d = c.to_dict()
        assert d["combined"] == 0.7


class TestVerdict:
    def test_verdict_creation(self):
        from app.schemas.audit_schema import Verdict
        v = Verdict(
            final_label="Entailment",
            confidence=0.85,
            reasoning="Evidence supports the hypothesis",
            iteration=2,
        )
        assert v.final_label == "Entailment"
        assert v.confidence == 0.85
        assert v.iteration == 2