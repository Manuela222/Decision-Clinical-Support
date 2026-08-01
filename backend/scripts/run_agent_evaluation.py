"""Runs the agent (real OpenAI API) across the real test-set admissions,
alongside baseline and trained model, and produces the data behind
FINAL_REPORT.md's agent evaluation tests (Section 4.3):

  Test 1 -- smoke test on a small admission subset (fail fast before
            spending the full run's API budget).
  Test 2 -- full three-way comparison (baseline/model/agent) on every
            test-set admission -- the agent's row in the Section 4.1 table.
  Test 3 -- agent safety/consistency audit over the same run: tool-call
            budget usage, hypertension-compatibility-check coverage,
            vocabulary-hallucination rejections.
  Test 4 -- confidence calibration per method: mean self-reported
            confidence split by whether the recommendation was actually
            correct (matched vs. extra against ground truth).

Requires OPENAI_API_KEY (backend/.env or shell env) -- makes real, billed
API calls. Not covered by pytest (same status as run_integration.py):
a reporting script, not library code.

Usage:
  python scripts/run_agent_evaluation.py --smoke-only   # Test 1 only
  python scripts/run_agent_evaluation.py                # full run (all 4 tests)
  python scripts/run_agent_evaluation.py --limit 30      # cap test-set admissions
"""
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cds.agent import AgentError, OpenAIProvider
from cds.evaluation import evaluate_all_methods
from cds_api import services
from cds_api.real_data import build_real_app_state

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "integration_results" / "agent_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_agent_over(state, admissions, label):
    """Run the agent over a list of CohortEntry, returning
    (results_by_hadm, errors) -- errors keyed by hadm_id, not raised, so one
    bad admission doesn't abort the whole batch."""
    results, errors = {}, {}
    t0 = time.time()
    for i, entry in enumerate(admissions, 1):
        try:
            results[entry.hadm_id] = services.predict_agent(state, entry.subject_id, entry.hadm_id)
        except AgentError as e:
            errors[entry.hadm_id] = str(e)
        except Exception as e:  # noqa: BLE001 -- record and keep going
            errors[entry.hadm_id] = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        print(f"  [{label}] {i}/{len(admissions)} (hadm_id={entry.hadm_id}) -- {elapsed:.1f}s elapsed", flush=True)
    return results, errors


def audit_agent_run(results_by_hadm):
    """Test 3: safety/consistency audit over a completed agent run."""
    tool_call_counts, rejected_events, htn_check_counts = [], [], []
    for hadm_id, result in results_by_hadm.items():
        n_tool_calls = sum(1 for s in result.reasoning_trace if s.step_type in ("tool_call", "retrieval"))
        tool_call_counts.append(n_tool_calls)
        n_htn_checks = sum(1 for s in result.reasoning_trace if s.step_type == "hypertension_compatibility_check")
        htn_check_counts.append(n_htn_checks)
        n_recs = len(result.recommended_medications)
        if n_htn_checks < n_recs:
            rejected_events.append({"hadm_id": hadm_id, "issue": "fewer compatibility checks than recommendations"})
        for s in result.reasoning_trace:
            if s.step_type == "reasoning" and "Rejected candidate" in (s.description or ""):
                rejected_events.append({"hadm_id": hadm_id, "issue": s.description})

    n = len(tool_call_counts) or 1
    return {
        "n_admissions": len(results_by_hadm),
        "tool_calls_per_admission": {
            "min": min(tool_call_counts, default=0), "max": max(tool_call_counts, default=0),
            "mean": sum(tool_call_counts) / n,
        },
        "hypertension_checks_per_admission": {
            "min": min(htn_check_counts, default=0), "max": max(htn_check_counts, default=0),
            "mean": sum(htn_check_counts) / n,
        },
        "every_recommendation_had_a_forced_hypertension_check": len(
            [e for e in rejected_events if "fewer compatibility checks" in e["issue"]]
        ) == 0,
        "vocabulary_hallucination_rejections": [e for e in rejected_events if "not part of the fixed" in e["issue"]],
        "other_flagged_events": [e for e in rejected_events if "not part of the fixed" not in e["issue"] and "fewer compatibility checks" not in e["issue"]],
    }


def confidence_calibration(method_name, results_by_hadm, ground_truth_by_hadm):
    """Test 4: mean self-reported confidence, split by matched vs. extra."""
    matched_conf, extra_conf = [], []
    for hadm_id, result in results_by_hadm.items():
        truth = set(ground_truth_by_hadm.get(hadm_id, []))
        for rec in result.recommended_medications:
            (matched_conf if rec.medication_class in truth else extra_conf).append(rec.confidence)
    return {
        "method": method_name,
        "n_matched_recommendations": len(matched_conf),
        "mean_confidence_when_matched": sum(matched_conf) / len(matched_conf) if matched_conf else None,
        "n_extra_recommendations": len(extra_conf),
        "mean_confidence_when_extra": sum(extra_conf) / len(extra_conf) if extra_conf else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true", help="Run only the small smoke test (Test 1) and exit.")
    parser.add_argument("--smoke-n", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of test-set admissions for the full run.")
    args = parser.parse_args()

    print("Building AppState from real MIMIC_III_10k data...")
    state = build_real_app_state(llm_provider_factory=OpenAIProvider)
    test_cohort = state.test_cohort()
    print(f"  {len(test_cohort)} test-set admissions available.")

    print(f"\n=== Test 1: smoke test ({args.smoke_n} admissions) ===")
    smoke_admissions = test_cohort[: args.smoke_n]
    smoke_results, smoke_errors = run_agent_over(state, smoke_admissions, "smoke")
    print(f"  {len(smoke_results)} succeeded, {len(smoke_errors)} failed.")
    (OUT_DIR / "test1_smoke_test.json").write_text(json.dumps({
        "n_admissions": len(smoke_admissions),
        "n_succeeded": len(smoke_results),
        "n_failed": len(smoke_errors),
        "errors": smoke_errors,
    }, indent=2))
    if smoke_errors:
        print(f"  ERRORS: {smoke_errors}")
    if args.smoke_only:
        print("\n--smoke-only set, stopping here.")
        return

    admissions = test_cohort if args.limit is None else test_cohort[: args.limit]
    print(f"\n=== Test 2+3+4: full run ({len(admissions)} admissions) ===")

    ground_truth_by_hadm = {c.hadm_id: state.get_ground_truth(c.subject_id, c.hadm_id) for c in admissions}

    baseline_results, model_results = {}, {}
    for entry in admissions:
        baseline_results[entry.hadm_id] = services.predict_baseline(state, entry.subject_id, entry.hadm_id)
        model_results[entry.hadm_id] = services.predict_model(state, entry.subject_id, entry.hadm_id)

    agent_results, agent_errors = run_agent_over(state, admissions, "agent")
    print(f"\nAgent: {len(agent_results)} succeeded, {len(agent_errors)} failed out of {len(admissions)}.")

    # Test 2: three-way comparison, only over admissions where the agent succeeded
    # (a fair comparison needs the identical admission set across all three methods).
    ok_hadm_ids = [c.hadm_id for c in admissions if c.hadm_id in agent_results]
    ok_admissions = [c for c in admissions if c.hadm_id in agent_results]
    ground_truth_list = [ground_truth_by_hadm[h] for h in ok_hadm_ids]
    baseline_list = [baseline_results[h] for h in ok_hadm_ids]
    model_list = [model_results[h] for h in ok_hadm_ids]
    agent_list = [agent_results[h] for h in ok_hadm_ids]

    comparison = evaluate_all_methods(ground_truth_list, baseline_list, model_list, agent_list)
    comparison.table.to_csv(OUT_DIR / "test2_full_comparison.csv", index=False)
    print("\n=== Test 2 result ===")
    print(comparison.table.to_string(index=False))

    # Test 3: safety/consistency audit
    audit = audit_agent_run(agent_results)
    audit["n_agent_errors"] = len(agent_errors)
    audit["agent_errors"] = agent_errors
    (OUT_DIR / "test3_safety_audit.json").write_text(json.dumps(audit, indent=2))
    print("\n=== Test 3 result ===")
    print(json.dumps(audit, indent=2))

    # Test 4: confidence calibration, all three methods, over the same admission set
    calibration = [
        confidence_calibration("baseline", {h: baseline_results[h] for h in ok_hadm_ids}, ground_truth_by_hadm),
        confidence_calibration("trained_model", {h: model_results[h] for h in ok_hadm_ids}, ground_truth_by_hadm),
        confidence_calibration("agent", agent_results, ground_truth_by_hadm),
    ]
    (OUT_DIR / "test4_confidence_calibration.json").write_text(json.dumps(calibration, indent=2))
    print("\n=== Test 4 result ===")
    print(json.dumps(calibration, indent=2))

    (OUT_DIR / "run_summary.json").write_text(json.dumps({
        "n_test_admissions_total": len(test_cohort),
        "n_admissions_in_this_run": len(admissions),
        "n_agent_succeeded": len(agent_results),
        "n_agent_failed": len(agent_errors),
    }, indent=2))
    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
