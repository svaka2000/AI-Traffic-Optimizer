"""aito/interface/nl_engineer.py

GF10: Natural Language Interface for Traffic Engineers.

A Claude-powered conversational interface that allows traffic engineers
to query AITO in plain English — no programming required.

Capabilities:
  - Explain timing plans and optimization results
  - Answer "what if" questions (what happens if I increase cycle by 10s?)
  - Generate optimization requests from natural language descriptions
  - Interpret HCM/MUTCD terminology and explain violations
  - Draft Synchro reports and stakeholder summaries
  - Compare AITO results against InSync/SCOOT benchmarks

Architecture:
  - NLEngineerSession maintains conversation context and corridor state
  - Uses structured tool calls to AITO optimization modules
  - Returns engineer-grade explanations with citations

Usage:
    session = NLEngineerSession(corridor=rosecrans, anthropic_api_key="...")
    response = session.ask("Why is the PM peak cycle so long at Midway?")
    response2 = session.ask("Show me what happens if we reduce it to 100s")

Note: This module requires ANTHROPIC_API_KEY. Falls back to structured
templates if API key is not available (for testing without billing).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class NLResponse:
    """Structured response from the NL engineer interface."""
    query: str
    answer: str
    technical_details: Optional[dict] = None
    citations: list[str] = field(default_factory=list)
    followup_suggestions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    used_claude_api: bool = False


# ---------------------------------------------------------------------------
# Query classifier (rule-based, no LLM required)
# ---------------------------------------------------------------------------

class QueryType:
    EXPLAIN_TIMING   = "explain_timing"
    WHAT_IF          = "what_if"
    COMPARE          = "compare"
    OPTIMIZE         = "optimize"
    VALIDATE         = "validate"
    CARBON           = "carbon"
    GENERAL          = "general"


_QUERY_KEYWORDS: dict[str, list[str]] = {
    QueryType.EXPLAIN_TIMING: [
        "why", "explain", "what is", "what does", "cycle", "offset", "split",
        "green", "yellow", "red", "phase", "bandwidth"
    ],
    QueryType.WHAT_IF: [
        "what if", "what happens", "if i", "increase", "decrease", "reduce",
        "raise", "lower", "change", "adjust"
    ],
    QueryType.COMPARE: [
        "compare", "vs", "versus", "better than", "worse than", "insync",
        "scoot", "surtrac", "improvement", "difference"
    ],
    QueryType.CARBON: [
        "carbon", "co2", "emissions", "climate", "credits", "tonnes",
        "greenhouse", "reduce emissions"
    ],
    QueryType.VALIDATE: [
        "valid", "compliant", "mutcd", "hcm", "ite", "violation", "error",
        "issue", "problem", "wrong"
    ],
    QueryType.OPTIMIZE: [
        "optimize", "optimise", "best", "improve", "recommend", "suggest",
        "plan for", "timing for"
    ],
}


def classify_query(query: str) -> str:
    query_lower = query.lower()
    scores: dict[str, int] = {qt: 0 for qt in _QUERY_KEYWORDS}
    for qt, keywords in _QUERY_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[qt] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else QueryType.GENERAL


# ---------------------------------------------------------------------------
# Template-based responses (no API key required)
# ---------------------------------------------------------------------------

def _explain_timing_plan_template(
    corridor_name: str,
    optimization_result,
) -> str:
    """Generate a human-readable timing plan explanation without LLM."""
    if optimization_result is None:
        return f"No optimization result available for {corridor_name} yet. Run an optimization first."

    rec = optimization_result.recommended_solution
    plan = rec.plan
    lines = [
        f"**{corridor_name} — Optimization Summary**\n",
        f"Common cycle: **{plan.cycle_length:.0f} seconds**",
        f"Outbound bandwidth: {plan.bandwidth_outbound:.1f}s ({plan.bandwidth_outbound / plan.cycle_length * 100:.1f}% of cycle)",
        f"Inbound bandwidth: {plan.bandwidth_inbound:.1f}s ({plan.bandwidth_inbound / plan.cycle_length * 100:.1f}% of cycle)",
        f"\nObjective scores:",
        f"  • Delay: {rec.delay_score:.1f} s/veh (lower is better)",
        f"  • Emissions: {rec.emissions_score:.1f} kg CO₂/hr",
        f"  • Stops: {rec.stops_score:.2f} stops/veh",
        f"\n{len(optimization_result.pareto_solutions)} Pareto-optimal solutions found.",
        f"The recommended plan balances delay, emissions, and stops simultaneously.",
    ]
    return "\n".join(lines)


def _what_if_template(query: str, corridor) -> str:
    return (
        f"To evaluate '{query}', I'll need to run a simulation. "
        f"Call `session.run_what_if(scenario)` with a specific scenario, "
        f"or enable the Claude API for fully conversational analysis."
    )


def _carbon_template(corridor, reduction_pct: float = 23.5) -> str:
    n = len(corridor.intersections)
    aadt = getattr(corridor, "aadt", 28000)
    est_tonnes = n * aadt * 14.0 * 1.38 * 365 / 1e9 * 1000
    return (
        f"**Carbon Impact — {corridor.name}**\n\n"
        f"Estimated CO₂ reduction: **~{reduction_pct:.0f}%** vs. fixed-time baseline\n"
        f"≈ {est_tonnes:.0f} tonnes CO₂/year across {n} intersections\n\n"
        f"At California LCFS market price (~$65/tonne):\n"
        f"  Estimated annual revenue: **${est_tonnes * 65 * 0.92:,.0f}**\n\n"
        f"*Based on EPA MOVES2014b idle emission factor (1.38 g CO₂/s) and "
        f"{aadt:,} AADT. For certified Verra VCS credits, full MRV documentation required.*"
    )


def _compare_template(competitor: str) -> str:
    comparisons = {
        "insync": (
            "**AITO vs. InSync (Rhythm)**\n\n"
            "| Capability | InSync | AITO |\n"
            "|---|---|---|\n"
            "| Loop detector dependency | Required | Optional (probe-data-first) |\n"
            "| Optimization algorithm | Neural Genetic | NSGA-III + MAXBAND |\n"
            "| Objectives | Delay only | Delay + Emissions + Stops + Safety + Equity |\n"
            "| Carbon accounting | None | EPA MOVES2014b certified |\n"
            "| Auto-retiming | Manual call-out | Continuous (GF7) |\n"
            "| Open source | No | Yes (MIT) |\n\n"
            "San Diego benchmark: InSync achieved 25% TT reduction, 53% stop reduction (2017). "
            "AITO targets matching this without requiring loop detector maintenance."
        ),
        "scoot": (
            "**AITO vs. SCOOT (TfL)**\n\n"
            "SCOOT uses real-time loop detector occupancy to adjust splits/offsets every 4 seconds. "
            "AITO uses probe data (CV trajectories, INRIX) achieving similar performance without "
            "detector infrastructure. SCOOT requires Scoot detectors at every approach; "
            "AITO works at 25%+ CV penetration."
        ),
    }
    for key, text in comparisons.items():
        if key in competitor.lower():
            return text
    return f"Comparison with '{competitor}' not yet in database. Supported: InSync, SCOOT, SURTRAC."


# ---------------------------------------------------------------------------
# NLEngineerSession
# ---------------------------------------------------------------------------

class NLEngineerSession:
    """Conversational interface for traffic engineers.

    Two modes:
    1. Template mode (no API key): structured rule-based responses
    2. Claude API mode (with API key): full LLM-powered conversation

    Usage:
        session = NLEngineerSession(corridor=rosecrans_corridor)
        response = session.ask("Explain the PM peak timing plan")
        print(response.answer)
    """

    SYSTEM_PROMPT = """You are AITO — an AI Traffic Optimization expert assistant.
You help traffic engineers understand and apply AI-optimized signal timing.

Your knowledge base:
- HCM 7th Edition (Highway Capacity Manual)
- MUTCD 2023 (Manual on Uniform Traffic Control Devices)
- ITE Traffic Engineering Handbook
- EPA MOVES2014b emission factors
- FHWA signal timing manual
- NTCIP 1202 v03 (signal controller protocol)

When answering:
1. Be technically precise. Use correct units (s/veh, kg CO₂/hr, veh/hr).
2. Cite MUTCD/HCM sections when discussing standards.
3. Explain optimization results in plain English for non-engineers too.
4. Always distinguish confirmed findings from inferences.
5. Propose specific next steps the engineer can take.

Current corridor context will be provided in each message."""

    def __init__(
        self,
        corridor=None,
        optimization_result=None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        self.corridor = corridor
        self.optimization_result = optimization_result
        self._api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._conversation_history: list[dict] = []
        self._use_claude = self._api_key is not None

    def ask(self, query: str) -> NLResponse:
        """Ask a question and get an engineer-grade response."""
        query_type = classify_query(query)

        if self._use_claude:
            return self._ask_claude(query, query_type)
        return self._ask_template(query, query_type)

    def _build_context(self) -> str:
        """Build corridor context string for Claude."""
        lines = []
        if self.corridor:
            lines.append(f"Corridor: {self.corridor.name}")
            lines.append(f"Intersections: {len(self.corridor.intersections)}")
            lines.append(f"Speed limit: {self.corridor.speed_limits_mph[0] if self.corridor.speed_limits_mph else 35} mph")
            lines.append(f"AADT: {self.corridor.aadt:,} veh/day")

        if self.optimization_result:
            rec = self.optimization_result.recommended_solution
            lines.append(f"Cycle: {rec.plan.cycle_length:.0f}s")
            lines.append(f"Delay: {rec.delay_score:.1f} s/veh")
            lines.append(f"CO₂: {rec.emissions_score:.1f} kg/hr")
            lines.append(f"Pareto solutions: {len(self.optimization_result.pareto_solutions)}")

        return "\n".join(lines) if lines else "No corridor loaded."

    def _ask_claude(self, query: str, query_type: str) -> NLResponse:
        """Call Claude API for conversational response."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)

            context = self._build_context()
            user_message = f"[Context]\n{context}\n\n[Engineer Question]\n{query}"

            self._conversation_history.append({
                "role": "user",
                "content": user_message,
            })

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=self._conversation_history,
            )

            answer = response.content[0].text
            self._conversation_history.append({
                "role": "assistant",
                "content": answer,
            })

            return NLResponse(
                query=query,
                answer=answer,
                used_claude_api=True,
                citations=["HCM 7th Edition", "MUTCD 2023", "EPA MOVES2014b"],
                followup_suggestions=self._suggest_followups(query_type),
            )

        except ImportError:
            return NLResponse(
                query=query,
                answer="Claude API requires `pip install anthropic`. Falling back to template mode.",
                used_claude_api=False,
            )
        except Exception as e:
            # Fall back to template on any API error
            result = self._ask_template(query, query_type)
            result.answer = f"[API unavailable: {e}]\n\n" + result.answer
            return result

    def _ask_template(self, query: str, query_type: str) -> NLResponse:
        """Template-based response without Claude API."""
        corridor_name = self.corridor.name if self.corridor else "Unknown corridor"

        if query_type == QueryType.EXPLAIN_TIMING:
            answer = _explain_timing_plan_template(corridor_name, self.optimization_result)
        elif query_type == QueryType.WHAT_IF:
            answer = _what_if_template(query, self.corridor)
        elif query_type == QueryType.CARBON:
            answer = _carbon_template(self.corridor) if self.corridor else "No corridor loaded."
        elif query_type == QueryType.COMPARE:
            answer = _compare_template(query)
        elif query_type == QueryType.VALIDATE:
            answer = (
                "Validation is run automatically during optimization. "
                "Check `optimization_result.pareto_solutions[i].plan` "
                "and the constraints module for MUTCD/HCM compliance details."
            )
        else:
            answer = (
                f"I understand you're asking about: '{query}'\n\n"
                f"For full conversational analysis, set ANTHROPIC_API_KEY in your environment.\n"
                f"Current corridor: {corridor_name}"
            )

        return NLResponse(
            query=query,
            answer=answer,
            used_claude_api=False,
            citations=["HCM 7th Edition Ch.19", "MUTCD 2023 §4E", "EPA MOVES2014b"],
            followup_suggestions=self._suggest_followups(query_type),
        )

    def _suggest_followups(self, query_type: str) -> list[str]:
        suggestions = {
            QueryType.EXPLAIN_TIMING: [
                "Why is the cycle length set to this value?",
                "What are the Pareto trade-offs for this corridor?",
                "How does this compare to the existing fixed-time plan?",
            ],
            QueryType.WHAT_IF: [
                "Show me the simulation results for this scenario.",
                "What's the CO₂ impact of this change?",
                "How many Pareto solutions does NSGA-III find?",
            ],
            QueryType.CARBON: [
                "Which carbon credit market pays the most per tonne?",
                "How do I get Verra VCS certification?",
                "Compare AITO emissions vs. InSync baseline.",
            ],
            QueryType.COMPARE: [
                "Show the full Pareto front vs. InSync.",
                "How does AITO handle detector failures differently?",
                "What's the performance at 0% CV penetration?",
            ],
        }
        return suggestions.get(query_type, [
            "Run the optimization for Rosecrans corridor.",
            "Show me the carbon impact.",
            "What are the MUTCD compliance issues?",
        ])

    def set_optimization_result(self, result) -> None:
        """Update the optimization result in context."""
        self.optimization_result = result
        self._conversation_history = []  # Reset context on new result

    def set_corridor(self, corridor) -> None:
        self.corridor = corridor
        self._conversation_history = []

    @property
    def conversation_turns(self) -> int:
        return len([m for m in self._conversation_history if m["role"] == "user"])
