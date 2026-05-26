"""
Experiment 1: Regioselectivity Stress Testing
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
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


PREDICT_PROMPT = """\
You are an expert organic chemist. A reactant has been chemically perturbed \
({perturbation_type} perturbation). Predict the major reaction product and \
identify the site of reaction.

Perturbed reactant SMILES: {reactant}
Perturbation type        : {perturbation_type}

Instructions:
1. Predict the major product SMILES after the most likely reaction.
2. Identify the reactive center: give the atom symbol and its index in the \
reactant (e.g. O@3, N@7). If multiple sites, list the most important one first.

Reply in this exact format:
PRODUCT: <SMILES>
SITE: <symbol>@<index>
REASONING: <1-3 sentences>

Answer:"""



def _strip_thinking(text: str) -> str:
    ANSWER_KEYWORDS = re.compile(
        r"(PRODUCT\s*[:\-]|SITE\s*[:\-]|REASONING\s*[:\-])",
        re.IGNORECASE
    )

    stripped = re.sub(r"<think>.*?</think>", "", text,
                      flags=re.DOTALL | re.IGNORECASE).strip()
    if stripped and ANSWER_KEYWORDS.search(stripped):
        return stripped

    last_kw = None
    for m in ANSWER_KEYWORDS.finditer(text):
        last_kw = m
    if last_kw:
        return text[last_kw.start():].strip()

    # Fallback: no answer keywords anywhere — return stripped or original
    return stripped or text


def canonicalise(smiles: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def tanimoto(smiles_a: str, smiles_b: str) -> float | None:
    ma = Chem.MolFromSmiles(smiles_a)
    mb = Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fa = gen.GetFingerprint(ma)
    fb = gen.GetFingerprint(mb)
    return round(TanimotoSimilarity(fa, fb), 4)

def parse_response(text: str) -> dict:
    """
    Extract PRODUCT SMILES, SITE label, and REASONING from model output.
    Returns dict with keys: product_smiles, site_label, site_index, reasoning.
    """
    clean = _strip_thinking(text)

    # ── Bug 1: PRODUCT regex ─────────────────────────────────────────────────
    # Original stopped at first whitespace after PRODUCT:, so missed patterns
    # like "PRODUCT:\nCCO" (newline before SMILES) and "PRODUCT: CCO " (space).
    # Also rejected single-atom SMILES (N, O) with {2,} minimum length.
    # Fix: consume optional newline+whitespace after colon; drop {2,} to {1,}.
    product_match = re.search(
        r"PRODUCT\s*[:\-]\s*\n?\s*([A-Za-z0-9@\[\]()\-=#%+.\\\/\*~]{1,})",
        clean, re.IGNORECASE
    )

    site_match = re.search(
        r"SITE\s*[:\-]\s*\n?\s*([A-Za-z][a-z]?)\s*@\s*(\d+)",
        clean, re.IGNORECASE
    )

    reasoning_match = re.search(
        r"REASONING\s*[:\-]\s*\n?\s*(.+?)(?=\n\s*[A-Z_]+\s*[:\-]|\Z)",
        clean, re.IGNORECASE | re.DOTALL
    )

    product_smiles = product_match.group(1).strip() if product_match else ""
    site_label     = site_match.group(1).strip()    if site_match    else ""
    site_index     = int(site_match.group(2))        if site_match    else -1
    reasoning      = reasoning_match.group(1).strip()[:400] if reasoning_match else clean[:400]

    return {
        "product_smiles": product_smiles,
        "site_label":     site_label,
        "site_index":     site_index,
        "reasoning":      reasoning,
    }


def evaluate_site(predicted_index: int, gold_indices_json: str) -> bool:
    """
    Return True if predicted atom index appears in the gold site indices list.
    gold_indices_json is a JSON string like "[3, 7, 12]".
    """
    if predicted_index < 0:
        return False
    try:
        gold_indices = json.loads(gold_indices_json)
        return predicted_index in gold_indices
    except Exception:
        return False


def _apply_smarts_to_product(
    product_smiles: str,
    smarts_from: str,
    smarts_to: str,
) -> str | None:
    """
    Apply the same SMARTS substitution used on the reactant to the product SMILES.
    Returns canonical SMILES of the transformed product, or None if:
      - the molecule is invalid
      - the reaction SMARTS cannot be parsed
      - the substitution group is absent from the product (spectator assumption fails)
      - sanitisation of the transformed product fails
    """
    if not product_smiles or not smarts_from or not smarts_to:
        return None
    try:
        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return None

        rxn_smarts = f"{smarts_from}>>{smarts_to}"
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        if rxn is None:
            return None

        products = rxn.RunReactants((mol,))
        if not products:
            return None

        transformed = products[0][0]
        Chem.SanitizeMol(transformed)
        return Chem.MolToSmiles(transformed)

    except Exception:
        return None


def _resolve_gold_product(rec: dict) -> tuple[str, str]:
    """
    Resolve the gold product SMILES for a record and return (smiles, source).
    """
    explicit = rec.get("gold_product_smiles", "").strip()
    if explicit:
        return explicit, "explicit"

    original_product = rec.get("original_product_smiles", "").strip()
    if not original_product:
        return "", "missing"

    smarts_from = rec.get("smarts_applied_from", "").strip()
    smarts_to   = rec.get("smarts_applied_to",   "").strip()

    if smarts_from and smarts_to:
        derived = _apply_smarts_to_product(original_product, smarts_from, smarts_to)
        if derived:
            return derived, "derived"

    return original_product, "fallback"


def _resolve_gold_sites(rec: dict) -> str:
    """
    Resolve gold reactive site indices as a JSON list string.
    """
    raw = rec.get("gold_reaction_site_atom", "").strip()

    if raw:
        # Format 1: atom-label style — "B@4" or "B@4,N@7"
        indices = []
        for token in raw.split(","):
            m = re.search(r"@(\d+)", token.strip())
            if m:
                indices.append(int(m.group(1)))
        if indices:
            return json.dumps(indices)

        # Format 2: raw JSON int list — "[4, 7]"
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return json.dumps([int(x) for x in parsed])
        except (json.JSONDecodeError, ValueError):
            pass
    return rec.get("original_reaction_sites", "[]").strip() or "[]"


def run_experiment(
    model_path: str,
    dataset_path: str,
    out_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    max_records: int | None = None,
) -> None:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Zero-shot Experiment 1: Regioselectivity Stress Testing")
    # ── Load dataset  (supports both JSON and CSV) ───────────
    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.suffix.lower() == ".csv":
        import csv as _csv
        with open(dataset_path, newline="") as f:
            dataset = list(_csv.DictReader(f))
    else:
        with open(dataset_path) as f:
            raw = json.load(f)
        dataset = raw if isinstance(raw, list) else raw.get("records", raw)

    dataset = [
        r for r in dataset
        if r.get("gold_product_smiles", "").strip()
        or r.get("original_product_smiles", "").strip()
    ]

    if max_records:
        dataset = dataset[:max_records]

    print(f"Loaded {len(dataset)} records with resolvable gold products")

    if not dataset:
        print("ERROR: No records have original_product_smiles or gold_product_smiles.")
        return

    # Pre-resolve gold labels and report provenance counts before inference
    provenance_counts = {"explicit": 0, "derived": 0, "fallback": 0, "missing": 0}
    for rec in dataset:
        _, src = _resolve_gold_product(rec)
        provenance_counts[src] += 1

    print(
        f"  Gold label provenance preview:\n"
        f"    explicit : {provenance_counts['explicit']}\n"
        f"    derived  : {provenance_counts['derived']}\n"
        f"    fallback : {provenance_counts['fallback']} "
        f"(EM unreliable for these records)\n"
        f"    missing  : {provenance_counts['missing']}"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_path}")
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

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\n── Inference ──")
    results = []

    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
        batch   = dataset[batch_start: batch_start + batch_size]
        prompts = [
            PREDICT_PROMPT.format(
                reactant=rec.get("perturbed_reactant_smiles",
                                  rec.get("original_reactant_smiles", "")),
                perturbation_type=rec.get("perturbation_type", "unknown"),
            )
            for rec in batch
        ]

        outputs = llm.generate(prompts, sampling_params)

        for rec, out in zip(batch, outputs):
            raw_response = out.outputs[0].text.strip()
            parsed       = parse_response(raw_response)

            # ── Gold resolution ───────────────────────────────────────────────
            gold_product, gold_source = _resolve_gold_product(rec)
            gold_sites                = _resolve_gold_sites(rec)

            # ── Metrics ───────────────────────────────────────────────────────
            pred_canon = canonicalise(parsed["product_smiles"])
            gold_canon = canonicalise(gold_product)

            exact_match = (
                pred_canon is not None
                and gold_canon is not None
                and pred_canon == gold_canon
            )

            em_reliable  = gold_source in ("explicit", "derived")
            tani         = tanimoto(parsed["product_smiles"], gold_product) if parsed["product_smiles"] else None
            site_correct = evaluate_site(parsed["site_index"], gold_sites)

            results.append({
                # Identity
                "record_id":             rec.get("reaction_id", "")[:60],
                "reaction_class":        rec.get("reaction_class", ""),
                "perturbation_type":     rec.get("perturbation_type", ""),

                # Input
                "original_reactant":     rec.get("original_reactant_smiles", ""),
                "perturbed_reactant":    rec.get("perturbed_reactant_smiles", ""),
                "tanimoto_perturbation": rec.get("tanimoto_reactant_similarity", ""),

                # Gold — with provenance
                "gold_product_smiles":   gold_product,
                "gold_product_source":   gold_source,
                "gold_site_indices":     gold_sites,

                # Model output
                "model_product_smiles":  parsed["product_smiles"],
                "model_product_canon":   pred_canon or "",
                "model_site_label":      parsed["site_label"],
                "model_site_index":      parsed["site_index"],
                "model_reasoning":       parsed["reasoning"],
                "model_raw_response":    raw_response[:4000],

                # Metrics
                "exact_match":           exact_match,
                "em_reliable":           em_reliable,
                "tanimoto_similarity":   tani,
                "site_prediction_correct": site_correct,
                "product_valid_smiles":  pred_canon is not None,
            })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total     = len(results)
    n_valid   = sum(1 for r in results if r["product_valid_smiles"])
    n_site_ok = sum(1 for r in results if r["site_prediction_correct"])
    tanis     = [r["tanimoto_similarity"] for r in results if r["tanimoto_similarity"] is not None]
    avg_tani  = round(sum(tanis) / len(tanis), 4) if tanis else None

    reliable_recs    = [r for r in results if r["em_reliable"]]
    fallback_recs    = [r for r in results if not r["em_reliable"]]
    n_exact_reliable = sum(1 for r in reliable_recs if r["exact_match"])
    n_exact_fallback = sum(1 for r in fallback_recs if r["exact_match"])

    n_explicit = sum(1 for r in results if r["gold_product_source"] == "explicit")
    n_derived  = sum(1 for r in results if r["gold_product_source"] == "derived")
    n_fallback = sum(1 for r in results if r["gold_product_source"] == "fallback")

    ptypes    = sorted(set(r["perturbation_type"] for r in results))
    per_ptype = {}
    for pt in ptypes:
        pt_recs      = [r for r in results if r["perturbation_type"] == pt]
        pt_reliable  = [r for r in pt_recs if r["em_reliable"]]
        pt_tanis     = [r["tanimoto_similarity"] for r in pt_recs if r["tanimoto_similarity"] is not None]
        per_ptype[pt] = {
            "total":            len(pt_recs),
            "n_reliable_gold":  len(pt_reliable),
            "exact_match":      sum(1 for r in pt_reliable if r["exact_match"]),
            "site_correct":     sum(1 for r in pt_recs if r["site_prediction_correct"]),
            "avg_tanimoto":     round(sum(pt_tanis) / len(pt_tanis), 4) if pt_tanis else None,
            "valid_smiles":     sum(1 for r in pt_recs if r["product_valid_smiles"]),
        }

    summary = {
        "run_timestamp":      run_ts,
        "model_path":         model_path,
        "dataset_path":       dataset_path,
        "total_records":      total,
        "gold_provenance": {
            "explicit": n_explicit,
            "derived":  n_derived,
            "fallback": n_fallback,
        },
        "exact_match_rate_reliable": (
            round(n_exact_reliable / len(reliable_recs), 4)
            if reliable_recs else 0
        ),
        "exact_match_rate_fallback": (
            round(n_exact_fallback / len(fallback_recs), 4)
            if fallback_recs else 0
        ),
        "site_accuracy":      round(n_site_ok / total, 4) if total else 0,
        "valid_smiles_rate":  round(n_valid   / total, 4) if total else 0,
        "avg_tanimoto":       avg_tani,
        "per_perturbation_type": per_ptype,
    }

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 68,
        "  EXPERIMENT 1 — REGIOSELECTIVITY STRESS TEST",
        f"  Model   : {model_path}",
        f"  Dataset : {dataset_path}",
        f"  Run     : {run_ts}",
        "=" * 68,
        f"  Total records                     : {total}",
        "",
        "  Gold label provenance:",
        f"    explicit (pre-annotated)         : {n_explicit}",
        f"    derived  (SMARTS propagated)     : {n_derived}",
        f"    fallback (original product)      : {n_fallback}  <- EM unreliable",
        "",
        "  Exact Match — reliable (explicit + derived) :",
        f"    {n_exact_reliable}/{len(reliable_recs)}  "
        f"({summary['exact_match_rate_reliable']:.3f})",
        "  Exact Match — fallback records (informational only) :",
        f"    {n_exact_fallback}/{len(fallback_recs)}  "
        f"({summary['exact_match_rate_fallback']:.3f})",
        "",
        f"  Site Prediction Accuracy          : "
        f"{summary['site_accuracy']:.3f}  ({n_site_ok}/{total})",
        f"  Avg Tanimoto Similarity           : "
        f"{avg_tani if avg_tani is not None else 'n/a'}",
        f"  Valid SMILES Rate                 : "
        f"{summary['valid_smiles_rate']:.3f}  ({n_valid}/{total})",
        "",
        "  Per-perturbation-type breakdown:",
        f"    {'type':<15}  {'n':>5}  {'n_rel':>5}  "
        f"{'EM(rel)':>8}  {'site':>6}  {'tani':>6}  {'valid%':>6}",
        "    " + "-" * 60,
    ]
    for pt, m in per_ptype.items():
        em_rate   = (
            round(m["exact_match"] / m["n_reliable_gold"], 3)
            if m["n_reliable_gold"] else float("nan")
        )
        sit_rate  = round(m["site_correct"] / m["total"], 3) if m["total"] else 0
        tani_str  = f"{m['avg_tanimoto']:.3f}" if m["avg_tanimoto"] is not None else "n/a"
        valid_pct = round(m["valid_smiles"] / m["total"], 3) if m["total"] else 0
        report_lines.append(
            f"    {pt:<15}  {m['total']:>5}  {m['n_reliable_gold']:>5}  "
            f"{em_rate:>8.3f}  {sit_rate:>6.3f}  {tani_str:>6}  {valid_pct:>6.3f}"
        )
    report_lines.append("=" * 68)
    report = "\n".join(report_lines)
    print("\n" + report)

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = out_dir / f"results_exp1_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": results}, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    csv_path = out_dir / f"results_exp1_{run_ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved CSV  -> {csv_path}")

    txt_path = out_dir / f"summary_exp1_{run_ts}.txt"
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved TXT  -> {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 1: Regioselectivity Stress Testing"
    )
    parser.add_argument("--model",       required=True)
    parser.add_argument("--dataset",     required=True,
                        help="JSON or CSV from exp1_regioselectivity.py")
    parser.add_argument("--out_dir",     default="./results")
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--max_tokens",  type=int,   default=10000,
                        help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_records", type=int,   default=None)
    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        max_records=args.max_records,
    )
