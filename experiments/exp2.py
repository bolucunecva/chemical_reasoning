"""
Experiment 2: Chain-of-Chemistry
"""
import json
import csv
import argparse
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from vllm import LLM, SamplingParams

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

COC_PROMPT = """\
You are an expert organic chemist. Reason through the following reaction \
step by step.

Reactant SMILES : {reactant}
Reagents        : {reagents}

Answer each step clearly:

Step A — Reactive center:
What atom or bond in the reactant will be directly involved in the reaction?
Give the atom symbol and its index in the SMILES string (e.g. O@3, N@7).

Step B — Key intermediate:
What is the most important intermediate that forms during this reaction?
Name the intermediate type (e.g. carbocation, enolate, Meisenheimer complex,
organopalladium complex) and provide its SMILES if possible.

Step C — Final product:
What is the SMILES of the major product?

Reply in this exact format:
STEP_A: <symbol>@<index>
STEP_B_TYPE: <intermediate type>
STEP_B_SMILES: <SMILES or 'unknown'>
STEP_C: <product SMILES>
REASONING: <2-4 sentences explaining the mechanism>

Answer:"""

JUDGE_PROMPT = """\
You are an expert chemistry evaluator assessing mechanistic reasoning quality.

Reaction: {reactant} → (reagents: {reagents})
Gold intermediate type: {gold_intermediate_type}
Gold intermediate hint: {gold_intermediate_hint}

Model's Step B response:
  Type  : {model_b_type}
  SMILES: {model_b_smiles}

Scoring rubric:
  3 — Correct intermediate type AND chemically valid SMILES (or correct type \
with good mechanistic justification if SMILES unknown)
  2 — Correct intermediate type but SMILES invalid/missing or reasoning vague
  1 — Wrong intermediate type but identifies a real intermediate in the pathway
  0 — Completely wrong, nonsensical, or no intermediate given

Reply in this exact format:
SCORE: <integer 0-3>
REASON: <one sentence>"""

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (MiMo-7B-RL / reasoning models)."""
    return re.sub(r"<think>.*?</think>", "", text,
                  flags=re.DOTALL | re.IGNORECASE).strip()


def canonicalise(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def tanimoto(smiles_a: str, smiles_b: str) -> float | None:
    ma = Chem.MolFromSmiles(smiles_a)
    mb = Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

    fa = morgan_gen.GetFingerprint(ma)
    fb = morgan_gen.GetFingerprint(mb)
    return round(TanimotoSimilarity(fa, fb), 4)


def parse_coc_response(text: str) -> dict:
    """
    Extract Step A, B, C fields from model CoC response.
    All fields default to empty/sentinel if not found.
    """
    clean = _strip_thinking(text)

    step_a  = re.search(r"STEP_A\s*[:\-]\s*([A-Za-z]+)@(\d+)",
                        clean, re.IGNORECASE)
    step_bt = re.search(r"STEP_B_TYPE\s*[:\-]\s*(.+?)(?=\nSTEP|$)",
                        clean, re.IGNORECASE | re.DOTALL)
    step_bs = re.search(r"STEP_B_SMILES\s*[:\-]\s*([^\n]+)",
                        clean, re.IGNORECASE)
    step_c = re.search(
                        r"STEP_C\s*[:\-]\s*([^\n\r]+)",
                        clean,
                        re.IGNORECASE,
)
    step_c_smiles = ""
    if step_c:
        step_c_smiles = step_c.group(1).strip()

        # Remove accidental trailing section headers
        step_c_smiles = re.split(
            r"\b(REASONING|STEP_[A-Z_]+)\b",
            step_c_smiles,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

    reason  = re.search(r"REASONING\s*[:\-]\s*(.+?)(?=\n[A-Z_]+:|$)",
                        clean, re.IGNORECASE | re.DOTALL)

    return {
        "step_a_label":  step_a.group(1).strip()       if step_a  else "",
        "step_a_index":  int(step_a.group(2))           if step_a  else -1,
        "step_b_type":   step_bt.group(1).strip()[:120] if step_bt else "",
        "step_b_smiles": step_bs.group(1).strip()       if step_bs else "",
        "step_c_smiles": step_c_smiles,
        "reasoning":     reason.group(1).strip()[:500]  if reason  else clean[:500],
    }

def _parse_gold_indices(gold_indices_json: str) -> list[int]:
    """
      - JSON int list:  "[1, 3, 10]"
      - symbol@index string: "O@1, O@3, B@10"
    Returns list of ints (may be empty).
    """
    raw = (gold_indices_json or "").strip()
    if not raw:
        return []

    # Try JSON int list first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try symbol@index format
    indices = [int(m) for m in re.findall(r"@(\d+)", raw)]
    return indices

def evaluate_step_a(
    predicted_label: str,
    predicted_index: int,
    gold_label: str,
    gold_indices_json: str,
) -> dict:
    """
    Returns dict with:
      index_correct : bool — predicted index in gold indices list
      label_correct : bool — predicted symbol matches a gold symbol
      both_correct  : bool — index_correct AND label_correct  (primary metric)
    """
    gold_indices = _parse_gold_indices(gold_indices_json)

    # Index check
    index_correct = (
        predicted_index >= 0
        and predicted_index in gold_indices
    )

    # Label / symbol check against gold_label like "O@1, O@3, B@10"
    gold_symbols = [s.upper() for s in re.findall(r"\b([A-Z][a-z]?)@\d+", gold_label)]
    label_correct = (
        bool(gold_symbols)
        and bool(predicted_label)
        and predicted_label.upper() in gold_symbols
    )

    return {
        "index_correct": index_correct,
        "label_correct": label_correct,
        "both_correct":  index_correct and label_correct,  # FIX 4: AND not OR
    }


def parse_judge_score(text: str) -> tuple[int, str]:
    clean   = _strip_thinking(text)
    score_m = re.search(r"SCORE\s*[:\-]\s*([0-3])", clean, re.IGNORECASE)
    reas_m  = re.search(r"REASON\s*[:\-]\s*(.+)",   clean, re.IGNORECASE | re.DOTALL)
    score   = int(score_m.group(1))           if score_m else -1
    reason  = reas_m.group(1).strip()[:300]   if reas_m  else clean[:300]
    return score, reason


def run_inference(args):
    dataset_path_obj = Path(args.dataset_path)
    if dataset_path_obj.suffix.lower() == ".csv":
        import csv as _csv
        with open(args.dataset_path, newline="") as f:
            dataset = list(_csv.DictReader(f))
    else:
        with open(args.dataset_path) as f:
            raw = json.load(f)
        dataset = raw if isinstance(raw, list) else raw.get("records", raw)

    dataset = [r for r in dataset if r.get("step_c_gold_smiles", "").strip()]
    if args.max_records:
        dataset = dataset[:args.max_records]

    print(f"Loaded {len(dataset)} CoC records with gold products")
    if not dataset:
        print("ERROR: No records have step_c_gold_smiles filled in.")
        return

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # stop=["###", "<|endoftext|>", "</s>", "<|im_end|>"],
    )

    results = []
    if args.max_records:
        dataset = dataset[:args.max_records]

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i:i+args.batch_size]

        prompts = [
            COC_PROMPT.format(
                reactant=rec.get("reactant_smiles", ""),
                reagents=rec.get("reagent_smiles", "not specified") or "not specified",
            )
            for rec in batch
        ]

        outputs = llm.generate(prompts, params)

        for rec, out in zip(batch, outputs):
            raw = out.outputs[0].text
            parsed = parse_coc_response(raw)

            results.append({
                **rec,
                **parsed,
                "model_raw": raw
            })

    out = Path(args.out_dir) / f"inference_{args.model.split('/')[-1]}.json"
    json.dump(results, open(out, "w"), indent=2)

    print("Saved →", out)


def run_eval(args):
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.inference_file:
        inference_file = Path(args.inference_file)
    else:
        inference_file = Path(args.out_dir) / f"inference_cot_{args.model.split('/')[-1]}.json"
    results = json.load(open(inference_file))

    for r in results:
        a_eval = evaluate_step_a(
            r["step_a_label"],
            r["step_a_index"],
            r.get("step_a_gold_label", ""),
            r.get("step_a_gold_atom_indices", "[]"),
        )

        r["step_a_index_correct"] = a_eval["index_correct"]
        r["step_a_label_correct"] = a_eval["label_correct"]
        r["step_a_correct"] = a_eval["both_correct"]

        gold = r.get("step_c_gold_smiles", "")
        pred = r.get("step_c_smiles", "")
        tani_c     = tanimoto(pred, gold)
        canon_pred = canonicalise(pred)
        canon_gold = canonicalise(gold)
        exact_c    = (
                canon_pred is not None
                and canon_gold is not None
                and canon_pred == canon_gold
            )
        r["step_c_tanimoto_predicted_vs_gold"] = tani_c
        r["step_c_exact_match"] = exact_c
        r['step_c_valid_smiles'] = canon_pred is not None

    # Judge
    if args.judge_model:
        judge = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

        params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # stop=["###", "<|endoftext|>", "</s>", "<|im_end|>"],
    )


        for i in tqdm(range(0, len(results), args.batch_size)):
            batch = results[i:i+args.batch_size]

            prompts = [
                JUDGE_PROMPT.format(
                    reactant=r["reactant_smiles"],
                    reagents=r.get("reagent_smiles", ""),
                    gold_intermediate_type=r.get("step_b_gold_intermediate_type", ""),
                    gold_intermediate_hint=r.get("step_b_gold_intermediate_hint", ""),
                    model_b_type=r.get("step_b_type", ""),
                    model_b_smiles=r.get("step_b_smiles", ""),
                )
                for r in batch
            ]

            outs = judge.generate(prompts, params)

            for r, o in zip(batch, outs):
                s, reason = parse_judge_score(o.outputs[0].text)
                r["step_b_judge_score"] = s
                r["step_b_judge_reason"] = reason


    for r in results:
        product_correct = (
            r["step_c_exact_match"]
            or (
                r["step_c_tanimoto_predicted_vs_gold"] is not None
                and r["step_c_tanimoto_predicted_vs_gold"] >= 0.8
            )
        )
        mechanism_wrong = (
            not r["step_a_correct"]
            and r["step_b_judge_score"] <= 1
        )
        r["diagnostic_correct_product_wrong_mechanism"] = (
            product_correct and mechanism_wrong
        )

    # ── Summary ────────────────────────────────────────
    total = len(results)

    # Step A — use both_correct (AND) as primary
    n_a_correct       = sum(1 for r in results if r["step_a_correct"])
    n_a_index_only    = sum(1 for r in results if r["step_a_index_correct"])
    n_a_label_only    = sum(1 for r in results if r["step_a_label_correct"])

    # Step B
    b_judged = [r for r in results if r["step_b_judge_score"] >= 0]
    avg_b    = round(
        sum(r["step_b_judge_score"] for r in b_judged) / len(b_judged), 4
    ) if b_judged else None
    b_pass   = sum(1 for r in b_judged if r["step_b_judge_score"] >= 2)

    # Step C
    n_exact_c  = sum(1 for r in results if r["step_c_exact_match"])
    tanis_c    = [
        r["step_c_tanimoto_predicted_vs_gold"]
        for r in results
        if r["step_c_tanimoto_predicted_vs_gold"] is not None
    ]
    avg_tani_c = round(sum(tanis_c) / len(tanis_c), 4) if tanis_c else None

    # Diagnostic
    n_diag = sum(
        1 for r in results
        if r["diagnostic_correct_product_wrong_mechanism"] is True
    )

    summary = {
        "run_timestamp":   run_ts,
        "model_path":      args.model,
        "judge_model":     args.judge_model,
        "self_judge":      args.judge_model == args.model,
        "dataset_path": args.dataset_path,
        "total_records":   total,
        "step_a": {
            "accuracy_both":        round(n_a_correct    / total, 4) if total else 0,
            "accuracy_index_only":  round(n_a_index_only / total, 4) if total else 0,
            "accuracy_label_only":  round(n_a_label_only / total, 4) if total else 0,
            "correct":              n_a_correct,
        },
        "step_b": {
            "avg_judge_score": avg_b,
            "pass_rate":       round(b_pass / len(b_judged), 4) if b_judged else None,
            "judged":          len(b_judged),
        },
        "step_c": {
            "exact_match_rate": round(n_exact_c / total, 4) if total else 0,
            "avg_tanimoto":     avg_tani_c,
            "exact_correct":    n_exact_c,
        },
        "diagnostic": {
            "correct_product_wrong_mechanism": n_diag,
            "rate": round(n_diag / total, 4) if total else 0,
        },
    }

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 65,
        "  EXPERIMENT 2 — CHAIN-OF-CHEMISTRY (CoC)",
        f"  Model   : {args.model}",
        f"  Judge   : {args.judge_model}"
        + (" ⚠ (self-judge)" if summary['self_judge'] else ""),
        f"  Dataset : {args.dataset_path}",
        f"  Run     : {run_ts}",
        "=" * 65,
        f"  Total records : {total}",
        "",
        "  Step A — Reactive Center Accuracy",
        f"    Both correct (symbol + index)  : "
        f"{n_a_correct}/{total}  ({summary['step_a']['accuracy_both']:.3f})",
        f"    Index correct only             : "
        f"{n_a_index_only}/{total}  ({summary['step_a']['accuracy_index_only']:.3f})",
        f"    Label correct only             : "
        f"{n_a_label_only}/{total}  ({summary['step_a']['accuracy_label_only']:.3f})",
        "",
        "  Step B — Intermediate Quality (LLM-as-judge, 0–3)",
        f"    Avg score      : {avg_b if avg_b is not None else 'n/a'}",
        f"    Pass rate (≥2) : "
        f"{summary['step_b']['pass_rate'] if summary['step_b']['pass_rate'] is not None else 'n/a'}"
        f"  ({b_pass}/{len(b_judged)})",
        "",
        "  Step C — Final Product Prediction",
        f"    Exact Match  : {n_exact_c}/{total}  "
        f"({summary['step_c']['exact_match_rate']:.3f})",
        f"    Avg Tanimoto : {avg_tani_c if avg_tani_c is not None else 'n/a'}",
        "",
        "  Diagnostic — Correct Product, Wrong Mechanism",
        f"    Count : {n_diag}/{total}  ({summary['diagnostic']['rate']:.3f})",
        "    (Step C correct but Step A wrong AND Step B score <= 1)",
        "=" * 65,
    ]
    report = "\n".join(report_lines)
    print("\n" + report)

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = Path(args.out_dir) / f"results_exp2_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": results}, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    csv_path = Path(args.out_dir) / f"results_exp2_{run_ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved CSV  -> {csv_path}")

    txt_path = Path(args.out_dir) / f"summary_exp2_{run_ts}.txt"
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved TXT  -> {txt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["inference", "eval"], required=True)

    p.add_argument("--model")
    p.add_argument("--judge_model")

    p.add_argument("--dataset_path")
    p.add_argument("--inference_file")

    p.add_argument("--out_dir", default="./results")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_records",      type=int,   default=None)
    args = p.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    if args.mode == "inference":
        run_inference(args)
    else:
        run_eval(args)
