"""
evals/eval_pipeline.py — Regression gate 

Runs fixed test cases through the pipeline and scores output on two layers:
  - Heuristic: structural checks (no API key needed, runs in CI)
  - LLM Judge: Groq rates on vibe, completeness, practicality, narrative

Usage:
  python -m evals.eval_pipeline --heuristic-only   # safe for CI
  python -m evals.eval_pipeline                     # full eval (needs GROQ_API_KEY)
  python -m evals.eval_pipeline --case paris_romantic
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# ANSI colours — gracefully degrade on Windows without colour support
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _c(colour: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"{colour}{text}{RESET}"
    return text


@dataclass
class EvalCase:
    name: str
    destination: str
    duration_days: int
    vibe: str
    places_to_avoid: List[str] = field(default_factory=list)
    expected_day_count: int = 0          # defaults to duration_days if 0
    must_mention: List[str] = field(default_factory=list)

TEST_CASES: List[EvalCase] = [
    EvalCase(
        name="paris_romantic",
        destination="Paris, France",
        duration_days=2,
        vibe="romantic, art, fine dining, hidden cafes",
        must_mention=["Paris", "Day 1"],
    ),
    EvalCase(
        name="london_historic",
        destination="London, UK",
        duration_days=3,
        vibe="historic, literary, cozy pubs",
        places_to_avoid=["Buckingham Palace"],
        must_mention=["London", "Day 1", "Day 2"],
    ),
    EvalCase(
        name="tokyo_modern",
        destination="Tokyo, Japan",
        duration_days=2,
        vibe="futuristic, anime, street food, neon",
        must_mention=["Tokyo", "Day 1"],
    ),
]

HEURISTIC_PASS_THRESHOLD = 0.70   # fraction of checks that must pass
LLM_JUDGE_PASS_THRESHOLD  = 6.5   # out of 10


@dataclass
class HeuristicResult:
    checks: dict[str, bool]

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(self.checks.values()) / len(self.checks)

    @property
    def passed(self) -> bool:
        return self.score >= HEURISTIC_PASS_THRESHOLD


def heuristic_score(case: EvalCase, itinerary: str) -> HeuristicResult:
    """Run structural checks against the itinerary text."""
    day_count = case.expected_day_count or case.duration_days
    text = itinerary or ""
    checks: dict[str, bool] = {}

    checks["min_length"]      = len(text) >= day_count * 200

    for d in range(1, day_count + 1):
        checks[f"day_{d}_present"] = bool(
            re.search(rf"\bday\s*{d}\b", text, re.IGNORECASE)
        )

    checks["has_morning"]     = bool(re.search(r"\bmorning\b",   text, re.IGNORECASE))
    checks["has_afternoon"]   = bool(re.search(r"\bafternoon\b", text, re.IGNORECASE))
    checks["has_evening"]     = bool(re.search(r"\bevening\b",   text, re.IGNORECASE))

    dest_keyword = case.destination.split(",")[0].strip()
    checks["destination_mentioned"] = dest_keyword.lower() in text.lower()

    # Catch raw coordinates leaking into output (e.g. "51.5194, -0.127")
    checks["no_raw_coords"]   = not bool(
        re.search(r"-?\d{1,3}\.\d{3,},\s*-?\d{1,3}\.\d{3,}", text)
    )

    for avoid in case.places_to_avoid:
        checks[f"avoids_{avoid.replace(' ', '_')[:20]}"] = (
            avoid.lower() not in text.lower()
        )

    for phrase in case.must_mention:
        checks[f"mentions_{phrase.replace(' ', '_')[:20]}"] = (
            phrase.lower() in text.lower()
        )

    checks["non_empty"]       = len(text.strip()) > 50

    return HeuristicResult(checks=checks)


@dataclass
class LLMJudgeResult:
    vibe_alignment:    float
    completeness:      float
    practicality:      float
    narrative_quality: float
    reasoning:         str

    @property
    def average(self) -> float:
        return (
            self.vibe_alignment
            + self.completeness
            + self.practicality
            + self.narrative_quality
        ) / 4

    @property
    def passed(self) -> bool:
        return self.average >= LLM_JUDGE_PASS_THRESHOLD


def llm_judge_score(case: EvalCase, itinerary: str) -> Optional[LLMJudgeResult]:
    """Rate the itinerary on 4 dimensions (1-10 each) using an LLM judge.
    Returns None if GROQ_API_KEY is not set or the call fails.
    """
    if not os.environ.get("GROQ_API_KEY"):
        return None

    try:
        from langchain_groq import ChatGroq
        judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    except Exception as e:
        print(f"  [judge] Could not initialise judge model: {e}")
        return None

    prompt = textwrap.dedent(f"""
        You are an expert travel planner evaluating an AI-generated itinerary.

        TRIP REQUEST:
          Destination  : {case.destination}
          Duration     : {case.duration_days} days
          Vibe/Interests: {case.vibe}
          Avoid        : {case.places_to_avoid or 'nothing specified'}

        GENERATED ITINERARY:
        ---
        {itinerary[:3000]}
        ---

        Rate the itinerary on each dimension from 1 (terrible) to 10 (excellent).
        Reply ONLY with valid JSON — no markdown, no explanation wrapper:
        {{
          "vibe_alignment":    <1-10>,
          "completeness":      <1-10>,
          "practicality":      <1-10>,
          "narrative_quality": <1-10>,
          "reasoning":         "<one sentence>"
        }}
    """).strip()

    try:
        response = judge.invoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        data = json.loads(raw)
        return LLMJudgeResult(
            vibe_alignment    = float(data["vibe_alignment"]),
            completeness      = float(data["completeness"]),
            practicality      = float(data["practicality"]),
            narrative_quality = float(data["narrative_quality"]),
            reasoning         = str(data.get("reasoning", "")),
        )
    except Exception as e:
        print(f"  [judge] Scoring failed: {e}")
        return None


def run_pipeline(case: EvalCase) -> Optional[str]:
    """Run the full LangGraph pipeline on a test case and return the itinerary."""
    try:
        from main import app  # noqa: PLC0415
    except Exception as e:
        print(_c(RED, f"  Failed to import main: {e}"))
        return None

    initial_input = {
        "destination":     case.destination,
        "duration_days":   case.duration_days,
        "vibe":            case.vibe,
        "user_feedback":   "Start",
        "places_to_avoid": case.places_to_avoid,
        "keywords":        [],
        "search_results":  [],
        "itinerary_draft": "",
    }

    final_state = initial_input.copy()
    try:
        for event in app.stream(initial_input):
            for _, node_output in event.items():
                if isinstance(node_output, dict):
                    final_state.update(node_output)
    except Exception as e:
        print(_c(RED, f"  Pipeline error: {e}"))
        return None

    return final_state.get("itinerary_draft", "")


def print_heuristic_report(result: HeuristicResult) -> None:
    for check, passed in result.checks.items():
        icon = _c(GREEN, "✓") if passed else _c(RED, "✗")
        print(f"    {icon}  {check}")
    bar = "█" * int(result.score * 20) + "░" * (20 - int(result.score * 20))
    colour = GREEN if result.passed else RED
    print(f"\n    Score: {_c(colour, bar)}  {result.score:.0%}  "
          f"({'PASS' if result.passed else 'FAIL'})")


def print_judge_report(result: Optional[LLMJudgeResult]) -> None:
    if result is None:
        print(_c(YELLOW, "    LLM judge skipped (no GROQ_API_KEY or error)."))
        return
    dims = [
        ("Vibe Alignment",    result.vibe_alignment),
        ("Completeness",      result.completeness),
        ("Practicality",      result.practicality),
        ("Narrative Quality", result.narrative_quality),
    ]
    for name, score in dims:
        bar = "█" * int(score) + "░" * (10 - int(score))
        colour = GREEN if score >= LLM_JUDGE_PASS_THRESHOLD else YELLOW
        print(f"    {name:<22} {_c(colour, bar)}  {score:.1f}/10")
    avg_colour = GREEN if result.passed else RED
    print(f"\n    Average: {_c(avg_colour, f'{result.average:.1f}/10')}  "
          f"({'PASS' if result.passed else 'FAIL'})")
    print(f"    Reasoning: {result.reasoning}")


def main() -> int:
    parser = argparse.ArgumentParser(description="WanderLust-AI eval pipeline")
    parser.add_argument(
        "--heuristic-only", action="store_true",
        help="Skip LLM judge (use for CI without API keys)."
    )
    parser.add_argument(
        "--case", metavar="NAME",
        help="Run only the named test case (e.g. paris_romantic)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score a placeholder itinerary instead of running the pipeline (tests the scorer itself)."
    )
    args = parser.parse_args()

    cases = TEST_CASES
    if args.case:
        cases = [c for c in TEST_CASES if c.name == args.case]
        if not cases:
            names = [c.name for c in TEST_CASES]
            print(_c(RED, f"Unknown case '{args.case}'. Available: {names}"))
            return 1

    overall_pass = True
    results_summary: list[dict] = []

    for case in cases:
        print()
        print(_c(BOLD, f"{'='*60}"))
        print(_c(BOLD, f" EVAL: {case.name}"))
        print(_c(BOLD, f" {case.destination} · {case.duration_days}d · \"{case.vibe}\""))
        print(_c(BOLD, f"{'='*60}"))

        if args.dry_run:
            itinerary = (
                f"**Day 1:**\n\n**Morning:** Explore central {case.destination.split(',')[0]}.\n"
                f"**Afternoon:** Visit local museums.\n**Evening:** Dinner at a local bistro.\n\n"
                f"**Day 2:**\n\n**Morning:** Stroll the river walk.\n"
                f"**Afternoon:** Art galleries.\n**Evening:** Jazz club (General Suggestion).\n"
            ) * case.duration_days
        else:
            print(f"\n  ⟳  Running pipeline (this may take 30-90s)…")
            t0 = time.time()
            itinerary = run_pipeline(case)
            elapsed = time.time() - t0
            if itinerary is None:
                print(_c(RED, "  Pipeline returned no itinerary — skipping scoring."))
                overall_pass = False
                continue
            print(f"  ✓  Pipeline completed in {elapsed:.1f}s  "
                  f"({len(itinerary)} chars output)\n")

        print(_c(BOLD, "\n  ▸ Layer 1: Heuristic Checks"))
        h_result = heuristic_score(case, itinerary)
        print_heuristic_report(h_result)

        j_result: Optional[LLMJudgeResult] = None
        if not args.heuristic_only:
            print(_c(BOLD, "\n  ▸ Layer 2: LLM Judge"))
            j_result = llm_judge_score(case, itinerary)
            print_judge_report(j_result)

        case_pass = h_result.passed
        if j_result is not None:
            case_pass = case_pass and j_result.passed

        gate_colour = GREEN if case_pass else RED
        gate_label  = "PASS ✓" if case_pass else "FAIL ✗"
        print(_c(BOLD, f"\n  ▸ Regression Gate: {_c(gate_colour, gate_label)}"))

        if not case_pass:
            overall_pass = False

        results_summary.append({
            "case":            case.name,
            "heuristic_score": round(h_result.score, 2),
            "heuristic_pass":  h_result.passed,
            "llm_judge_avg":   round(j_result.average, 1) if j_result else None,
            "llm_judge_pass":  j_result.passed if j_result else None,
            "overall_pass":    case_pass,
        })

    print()
    print(_c(BOLD, f"{'='*60}"))
    print(_c(BOLD, " SUMMARY"))
    print(_c(BOLD, f"{'='*60}"))
    for r in results_summary:
        icon = _c(GREEN, "✓ PASS") if r["overall_pass"] else _c(RED, "✗ FAIL")
        llm_str = (
            f"  LLM={r['llm_judge_avg']}/10"
            if r["llm_judge_avg"] is not None
            else "  LLM=skipped"
        )
        print(f"  {icon}  {r['case']:<25} H={r['heuristic_score']:.0%}{llm_str}")

    print()
    if overall_pass:
        print(_c(GREEN, _c(BOLD, "  ALL EVALS PASSED — regression gate is GREEN ✓")))
    else:
        print(_c(RED, _c(BOLD, "  SOME EVALS FAILED — regression gate is RED ✗")))
    print()

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
