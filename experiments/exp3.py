"""
Experiment 3: Counterfactual Validity Test
"""

import json
import csv
import argparse
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams



PREDICT_PROMPT = """\
You are an expert chemist. Carefully analyse the following proposed reaction.

Reaction SMILES  : {rxn}
Reactant         : {reactant}
Proposed product : {proposed_product}

Task:
1. Decide whether this reaction can proceed under standard laboratory conditions.
2. State your verdict clearly as one of:
     PROCEEDS      — the reaction occurs and the product is chemically valid
     IMPOSSIBLE    — the reaction cannot occur
3. Give a mechanistic explanation (2–4 sentences) identifying any violated
   chemical principles (e.g. Bredt's rule, Woodward–Hoffmann, impossible
   valence, charge imbalance, reagent mismatch, aromaticity destruction).

Answer:"""



# ─────────────────────────────────────────────────────────────────────────────
# Response parsers
# ─────────────────────────────────────────────────────────────────────────────

# FIX 9: unified vocabulary — both "no_reaction" and "impossible" → "impossible"
_IMPOSSIBLE_SIGNALS = [
    "IMPOSSIBLE", "NO_REACTION", "NO REACTION", "CANNOT PROCEED",
    "DOES NOT PROCEED", "WILL NOT PROCEED", "NOT FEASIBLE",
    "VIOLATES", "FORBIDDEN", "CANNOT OCCUR", "DOES NOT OCCUR",
    "NOT POSSIBLE", "NOT CHEMICALLY", "WOULD NOT", "CANNOT FORM",
]
_PROCEEDS_SIGNALS = [
    "PROCEEDS", "WILL PROCEED", "CAN PROCEED", "REACTION OCCURS",
    "DOES PROCEED", "PRODUCT IS", "PRODUCT SMILES", "GOES FORWARD",
    "TAKES PLACE",
]

# FIX 9: normalised gold label vocabulary
_GOLD_NORMALISE = {
    "no_reaction": "impossible",
    "impossible":  "impossible",
    "proceeds":    "proceeds",
}


def _normalise_gold(label: str) -> str:
    """Normalise gold label to 'impossible' | 'proceeds' | 'unknown'."""
    return _GOLD_NORMALISE.get(label.strip().lower(), "unknown")


def parse_verdict(text: str) -> str:
    """
    Return 'impossible', 'proceeds', or 'unknown'.
    FIX 9: output vocabulary unified — 'no_reaction' renamed to 'impossible'
    to match paper's two-class framing.
    """
    upper = text.upper()
    for sig in _IMPOSSIBLE_SIGNALS:
        if sig in upper:
            return "impossible"
    for sig in _PROCEEDS_SIGNALS:
        if sig in upper:
            return "proceeds"
    return "unknown"

def extract_smiles(text: str) -> str:
    """Try to pull a product SMILES from a model response that said PROCEEDS."""
    m = re.search(
        r"(?:product|SMILES|structure)\s*[:\-=]?\s*"
        r"([A-Za-z0-9@\[\]()\-=#%+.\\\/]{6,})",
        text, re.IGNORECASE
    )
    return m.group(1).strip() if m else ""


def run_experiment(
    model_path: str,
    dataset_path: str,
    out_dir: str,
    judge_model_path: str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,       # FIX 7: was 10000
    judge_max_tokens: int = 128,     # FIX 7: was 10000
    temperature: float = 0.1,
    max_records: int | None = None,
) -> None:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Zero-shot Experiment 3: Counterfactual Validity Test")
    # FIX 8: self-judge warning
    effective_judge = judge_model_path or model_path
    if effective_judge == model_path:
        print(
            "\n⚠  WARNING: judge_model is the same as model_path.\n"
            "   Self-evaluation inflates explanation quality scores.\n"
            "   Use --judge_model with a separate stronger model.\n"
        )

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(dataset_path) as f:
        dataset = json.load(f)
    if isinstance(dataset, dict) and "records" in dataset:
        dataset = dataset["records"]

    # FIX 3: filter before slice to guarantee max_records usable records
    dataset = [r for r in dataset if r.get("reactant_smiles", "").strip()]
    if max_records:
        dataset = dataset[:max_records]
    print(f"Loaded {len(dataset)} counterfactual records")

    # ── Load subject model ────────────────────────────────────────────────────
    print(f"\nLoading subject model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=16384, 
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        # Do NOT add "\n\n" or similar — MiMo emits blank lines between
        # <think> and the answer block; stopping on whitespace kills the answer.
        stop=["###", "<|endoftext|>", "</s>", "<|im_end|>"],
    )

    print("\n── Pass 1: Verdict inference ──")
    results = []

    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Inferring"):
        batch   = dataset[batch_start: batch_start + batch_size]
        prompts = [
            PREDICT_PROMPT.format(
                rxn=rec.get("impossible_rxn_smiles", ""),
                reactant=rec.get("reactant_smiles", ""),
                # FIX 1: proposed_product_smiles (was fake_product_smiles)
                proposed_product=rec.get(
                    "proposed_product_smiles",
                    rec.get("fake_product_smiles", "")   # backwards compat
                ),
            )
            for rec in batch
        ]

        outputs = llm.generate(prompts, sampling_params)

        for rec, out in zip(batch, outputs):
            response = out.outputs[0].text.strip()
            verdict  = parse_verdict(response)

            gold = _normalise_gold(rec.get("gold_label", "impossible"))

            correct = (verdict == gold)

            results.append({
                # ── Identity ──────────────────────────────────────────────────
                "record_id":          rec.get("record_id", ""),
                "source":             rec.get("source", ""),
                "violation_category": rec.get("violation_category", ""),
                "difficulty":         rec.get("difficulty", ""),

                # ── Chemistry ────────────────────────────────────────────────
                "impossible_rxn_smiles":  rec.get("impossible_rxn_smiles", ""),
                "reactant_smiles":        rec.get("reactant_smiles", ""),
                # FIX 1: renamed field stored in results
                "proposed_product_smiles": rec.get(
                    "proposed_product_smiles",
                    rec.get("fake_product_smiles", "")
                ),

                # ── Gold ──────────────────────────────────────────────────────
                "gold_label":            gold,
                "violation_explanation": rec.get("violation_explanation", ""),
                "surface_plausibility":  rec.get("surface_plausibility", ""),

                # ── Model output (Pass 1) ─────────────────────────────────────
                "model_verdict":          verdict,
                "model_raw_response":     response,
                "model_extracted_smiles": (
                    extract_smiles(response) if verdict == "proceeds" else ""
                ),

                # ── Pass 1 metrics ────────────────────────────────────────────
                # FIX 2: True/False (not "" or None)
                "failure_detection_correct": correct,
                "false_positive":            verdict == "proceeds" and gold == "impossible",
                "verdict_unknown":           verdict == "unknown",

            })


    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total   = len(results)
    correct = sum(1 for r in results if r["failure_detection_correct"])
    fp      = sum(1 for r in results if r["false_positive"])
    unk     = sum(1 for r in results if r["verdict_unknown"])


    fdr = correct / total if total else 0.0
    fpr = fp      / total if total else 0.0

    # Per-category
    categories = sorted(set(r["violation_category"] for r in results))
    per_cat = {}
    for cat in categories:
        cat_recs   = [r for r in results if r["violation_category"] == cat]
        cat_correct= sum(1 for r in cat_recs if r["failure_detection_correct"])

        per_cat[cat] = {
            "total":           len(cat_recs),
            "correct":         cat_correct,
            "fdr":             round(cat_correct / len(cat_recs), 4) if cat_recs else 0.0,
        }

    # Per-difficulty
    per_diff = {}
    for diff in ["easy", "medium", "hard"]:
        diff_recs    = [r for r in results if r["difficulty"] == diff]
        if not diff_recs:
            continue
        diff_correct = sum(1 for r in diff_recs if r["failure_detection_correct"])
        label = diff if diff else "unspecified"
        per_diff[label] = {
            "total":   len(diff_recs),
            "correct": diff_correct,
            "fdr":     round(diff_correct / len(diff_recs), 4),
        }

    # Per-source (curated_template vs thermodynamic_reversal)
    sources = sorted(set(r["source"] for r in results))
    per_source = {}
    for src in sources:
        src_recs    = [r for r in results if r["source"] == src]
        src_correct = sum(1 for r in src_recs if r["failure_detection_correct"])
        per_source[src] = {
            "total":   len(src_recs),
            "correct": src_correct,
            "fdr":     round(src_correct / len(src_recs), 4) if src_recs else 0.0,
        }

    summary = {
        "run_timestamp":           run_ts,
        "model_path":              model_path,
        "judge_model":             effective_judge,
        "self_judge":              effective_judge == model_path,
        "dataset_path":            dataset_path,
        "total_records":           total,
        "failure_detection_rate":  round(fdr, 4),
        "false_positive_rate":     round(fpr, 4),
        "unknown_rate":            round(unk / total, 4) if total else 0.0,
        "per_category":            per_cat,
        "per_difficulty":          per_diff,
        "per_source":              per_source,   # curated vs thermodynamic
    }

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 65,
        "  EXPERIMENT 3 — COUNTERFACTUAL VALIDITY TEST",
        f"  Model    : {model_path}",
        f"  Dataset  : {dataset_path}",
        f"  Run time : {run_ts}",
        "=" * 65,
        f"  Total records evaluated     : {total}",
        f"  Failure Detection Rate (FDR): {fdr:.3f}  ({correct}/{total})",
        f"  False Positive Rate   (FPR) : {fpr:.3f}  ({fp}/{total})",
        f"  Unknown / no verdict        : {unk}",
        "",
        "  Per-source FDR:",
    ]
    for src, m in per_source.items():
        report_lines.append(
            f"    {src:<35} {m['correct']:>3}/{m['total']:<3}  FDR={m['fdr']:.3f}"
        )

    report_lines += ["", "  Per-difficulty FDR:"]
    for diff, m in per_diff.items():
        report_lines.append(
            f"    {diff:<10} {m['correct']:>3}/{m['total']:<3}  FDR={m['fdr']:.3f}"
        )
    report_lines.append("=" * 65)
    report = "\n".join(report_lines)
    print("\n" + report)

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = out_dir / f"results_exp3_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": results}, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    csv_path = out_dir / f"results_exp3_{run_ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved CSV  -> {csv_path}")

    txt_path = out_dir / f"summary_exp3_{run_ts}.txt"
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved TXT  -> {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 3: Counterfactual Validity"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path or HuggingFace ID of model to evaluate"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to counterfactual_dataset.json (from exp3_counterfactual.py)"
    )
    parser.add_argument(
        "--out_dir", default="./results",
        help="Directory for output files (default: ./results)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--max_tokens", type=int, default=10000, 
        help="Max new tokens for verdict pass (default: 512)"
    )
    parser.add_argument(
        "--judge_max_tokens", type=int, default=128,
        help="Max new tokens for judge pass (default: 128)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0
    )
    parser.add_argument(
        "--max_records", type=int, default=None,
        help="Limit to N records for quick testing (default: all)"
    )
    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        judge_model_path=args.judge_model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        temperature=args.temperature,
        max_records=args.max_records,
    )
