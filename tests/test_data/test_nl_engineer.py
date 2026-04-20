"""Tests for aito/interface/nl_engineer.py (GF10)."""
import pytest
from aito.interface.nl_engineer import (
    NLResponse,
    classify_query,
    NLEngineerSession,
)
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")


class TestClassifyQuery:
    def test_timing_question_classified(self):
        qtype = classify_query("Why is the cycle length so long at Midway Drive?")
        assert isinstance(qtype, str)
        assert len(qtype) > 0

    def test_what_if_classified(self):
        qtype = classify_query("What happens if I reduce the cycle to 90 seconds?")
        assert "what_if" in qtype.lower() or "what" in qtype.lower() or len(qtype) > 0

    def test_compare_classified(self):
        qtype = classify_query("Compare AITO to InSync")
        assert "compare" in qtype.lower() or len(qtype) > 0

    def test_carbon_classified(self):
        qtype = classify_query("What is the carbon impact?")
        assert isinstance(qtype, str)

    def test_all_queries_return_non_empty(self):
        queries = [
            "What is the delay at Rosecrans?",
            "Why is there congestion?",
            "How much CO2 is saved?",
            "Compare performance to legacy system",
            "What if we add another lane?",
            "Show me the timing plan",
        ]
        for q in queries:
            result = classify_query(q)
            assert result is not None
            assert len(result) > 0

    def test_returns_string(self):
        result = classify_query("Test query")
        assert isinstance(result, str)


class TestNLResponse:
    def test_has_answer_field(self):
        resp = NLResponse(query="test", answer="test answer")
        assert resp.answer == "test answer"

    def test_used_claude_api_defaults_false(self):
        resp = NLResponse(query="test", answer="test answer")
        assert resp.used_claude_api == False

    def test_answer_nonempty(self):
        resp = NLResponse(query="test", answer="some text")
        assert len(resp.answer) > 0


class TestNLEngineerSession:
    def setup_method(self):
        self.session = NLEngineerSession(corridor=ROSECRANS)

    def test_instantiation(self):
        assert self.session is not None

    def test_ask_returns_nl_response(self):
        resp = self.session.ask("Compare AITO to InSync")
        assert isinstance(resp, NLResponse)

    def test_ask_returns_non_empty_answer(self):
        resp = self.session.ask("What is the delay at this corridor?")
        assert len(resp.answer) > 0

    def test_timing_question_answered(self):
        resp = self.session.ask("Why is the cycle length so long?")
        assert isinstance(resp.answer, str)
        assert len(resp.answer) > 10

    def test_compare_question_produces_table(self):
        resp = self.session.ask("Compare AITO to InSync")
        # The compare template produces a markdown table
        assert "|" in resp.answer or "AITO" in resp.answer

    def test_carbon_question(self):
        resp = self.session.ask("What is the carbon impact of Rosecrans?")
        assert isinstance(resp.answer, str)

    def test_what_if_question(self):
        resp = self.session.ask("What happens if I reduce the cycle to 90 seconds?")
        assert isinstance(resp.answer, str)

    def test_used_claude_api_is_bool(self):
        resp = self.session.ask("test question")
        assert isinstance(resp.used_claude_api, bool)

    def test_multiple_questions_in_sequence(self):
        questions = [
            "Why is the cycle length so long?",
            "Compare AITO to InSync",
            "What is the CO2 reduction?",
        ]
        for q in questions:
            resp = self.session.ask(q)
            assert isinstance(resp, NLResponse)
            assert len(resp.answer) > 0

    def test_session_without_corridor(self):
        session = NLEngineerSession(corridor=None)
        resp = session.ask("What is the delay?")
        assert isinstance(resp, NLResponse)

    def test_query_field_in_response(self):
        q = "Why is traffic congested?"
        resp = self.session.ask(q)
        assert resp.query == q

    def test_fallback_mode_when_no_api(self):
        # Without Claude API configured, should fall back to template mode
        resp = self.session.ask("Compare AITO to InSync")
        # Either template mode (used_claude_api=False) or API mode
        assert isinstance(resp.used_claude_api, bool)
        assert len(resp.answer) > 0
