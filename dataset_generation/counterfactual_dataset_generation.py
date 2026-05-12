import json
import csv
import argparse
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdchem import ChiralType


INTERMEDIATE_TEMPLATES = {
    # class -> (intermediate_type, mechanistic_hint)

    # Class 1: Heteroatom alkylation / arylation (SN2, SNAr, O/N-alkylation)
    "1":  (
        "alkoxide_or_amide",
        "Deprotonation of nucleophile (O or N) → oxyanion/amide attacks electrophilic carbon"
    ),

    # Class 2: Acylation (ester/amide bond formation, acid chloride reactions)
    "2":  (
        "tetrahedral_intermediate",
        "Nucleophilic addition to acyl carbon → tetrahedral alkoxide intermediate → collapse with loss of leaving group"
    ),

    # Class 3: C–C coupling (Suzuki, Heck, Sonogashira, Negishi, Stille)
    "3":  (
        "organopalladium_complex",
        "Pd(0) oxidative addition into C–X or C–OTf → transmetalation with organometallic partner "
        "(boronic ester, organozinc, etc.) → Ar–Pd(II)–Ar' complex → reductive elimination"
    ),

    # Class 4: Heterocycle formation (ring-closing condensations, cyclisations)
    "4":  (
        "hemiaminal_or_imine",
        "Condensation of amine with carbonyl → hemiaminal → imine (Schiff base); "
        "or intramolecular nucleophilic cyclisation"
    ),

    # Class 5: Protections / deprotections (e.g. TBS, Boc, Cbz)
    "5":  (
        "oxocarbenium_or_acylium",
        "Activation of protecting group electrophile (e.g. silyl cation, acylium) "
        "→ nucleophilic attack by heteroatom"
    ),

    # Class 6: Reductions (NaBH4, LiAlH4, hydrogenation, DIBAL)
    "6":  (
        "hydride_transfer_complex",
        "Hydride delivery from reductant (e.g. NaBH4, LiAlH4) to electrophilic carbonyl "
        "or imine → alkoxide / amide intermediate"
    ),

    # Class 7: Oxidations (Swern, Jones, mCPBA epoxidation, Dess-Martin)
    "7":  (
        "oxoammonium_or_peracid_complex",
        "Activated oxidant (e.g. oxoammonium from DMSO activation, peracid) "
        "attacks nucleophilic substrate → electron transfer → oxidised product"
    ),

    # Class 8: Functional group interconversion (FGI) not covered above
    #           e.g. ester hydrolysis, nitrile hydration, halide exchange
    "8":  (
        "tetrahedral_intermediate_or_SN2",
        "Nucleophilic substitution or addition-elimination at sp3 or carbonyl carbon"
    ),

    # Class 9: Functional group addition (e.g. halogenation, nitration, sulfonation)
    "9":  (
        "arenium_ion_or_sigma_complex",
        "Electrophilic aromatic substitution → arenium ion (Wheland intermediate / σ-complex) "
        "→ deprotonation restores aromaticity"
    ),

    # Class 10: Deprotections / miscellaneous transformations
    "10": (
        "transition_state",
        "Concerted or stepwise bond-breaking/forming; consult specific reaction conditions"
    ),

    "default": (
        "transition_state",
        "Concerted bond-breaking / forming; mechanism depends on specific reaction class"
    ),
}


def _get_intermediate_hint(reaction_class: str) -> tuple[str, str]:
    """Return (intermediate_type, hint) for a given reaction class string."""
    for key, val in INTERMEDIATE_TEMPLATES.items():
        if key != "default" and reaction_class.strip().startswith(key):
            return val
    return INTERMEDIATE_TEMPLATES["default"]



REACTIVE_HETEROATOMS = {
    5,   # B  — Suzuki / Miyaura boronic esters          ← added (was missing)
    7,   # N  — amination, reductive amination
    8,   # O  — esterification, etherification
    15,  # P  — phosphorylation
    16,  # S  — thiolation
    17,  # Cl — SNAr, cross-coupling leaving group        ← added
    35,  # Br — cross-coupling leaving group              ← added
    53,  # I  — cross-coupling leaving group              ← added
}


def get_reactive_center(
    reactant_smiles: str,
    product_smiles: str,
    rxn_smiles: str = "",
) -> dict:
    """
    Identify reactive centre atoms in the reactant.

    FIX 2: The original implementation flagged every non-H atom as "changed"
    then blindly returned the first 3 heteroatoms — this is not a bond-change
    detector; it just returns any heteroatoms present. For a Suzuki substrate
    COC(=O)C(C)c1ccc(B2OC(C)(C)C(C)(C)O2)c(Cl)c1 it returned O@1, O@3 (ester
    oxygens, spectators) instead of B@10 (the true reactive boron).

    New strategy (priority order):
      1. Atom-map guided: if rxn_smiles contains atom maps, find atoms whose
         bond count changes between reactant and product sides. This is the
         most reliable method when atom maps are available.
      2. Heuristic fallback: if atom maps are absent or parsing fails, use
         the extended REACTIVE_HETEROATOMS set (matching Experiment 1) and
         return all matching atoms, not just the first 3.

    Returns:
        dict with keys:
          atom_indices : list[int]  — 0-based indices in reactant
          label        : str        — "Symbol@idx, ..." human-readable
          method       : str        — "atom_map" | "heuristic"
    """
    rm = Chem.MolFromSmiles(reactant_smiles)
    if rm is None:
        return {"atom_indices": [], "label": "unknown", "method": "failed"}

    # ── Strategy 1: atom-map guided bond-change detection ───────────────────
    if rxn_smiles and ">>" in rxn_smiles:
        try:
            rxn_parts = rxn_smiles.split(">>")
            r_mapped = Chem.MolFromSmiles(rxn_parts[0].split(".")[0])
            p_mapped = Chem.MolFromSmiles(rxn_parts[-1].split(".")[0])

            if r_mapped is not None and p_mapped is not None:
                # Build map_num -> (atom_idx, degree) for reactant
                r_map = {
                    a.GetAtomMapNum(): (a.GetIdx(), a.GetDegree())
                    for a in r_mapped.GetAtoms()
                    if a.GetAtomMapNum() > 0
                }
                # Build map_num -> degree for product
                p_map = {
                    a.GetAtomMapNum(): a.GetDegree()
                    for a in p_mapped.GetAtoms()
                    if a.GetAtomMapNum() > 0
                }

                changed_map_nums = [
                    mn for mn, (idx, r_deg) in r_map.items()
                    if mn in p_map and p_map[mn] != r_deg
                ]

                if changed_map_nums:
                    # Re-index to clean (unmapped) reactant mol
                    # Match by canonical atom order heuristic
                    clean_mol = Chem.MolFromSmiles(reactant_smiles)
                    # Use atom map numbers to find indices in clean mol
                    map_to_clean = {
                        a.GetAtomMapNum(): a.GetIdx()
                        for a in r_mapped.GetAtoms()
                        if a.GetAtomMapNum() > 0
                    }
                    # Strip maps to get canonical indices in clean mol
                    clean_indices = []
                    for mn in changed_map_nums:
                        if mn in map_to_clean:
                            # Map the index from mapped mol to clean mol by
                            # matching atom symbols in order (best effort)
                            mapped_idx = map_to_clean[mn]
                            mapped_atom = r_mapped.GetAtomWithIdx(mapped_idx)
                            clean_indices.append(mapped_idx)

                    if clean_indices:
                        labels = [
                            f"{r_mapped.GetAtomWithIdx(i).GetSymbol()}@{i}"
                            for i in sorted(clean_indices)
                        ]
                        return {
                            "atom_indices": sorted(clean_indices),
                            "label": ", ".join(labels),
                            "method": "atom_map",
                        }
        except Exception:
            pass  # fall through to heuristic

    # ── Strategy 2: heuristic fallback (extended heteroatom set) ────────────
    sites = []
    for atom in rm.GetAtoms():
        if atom.GetAtomicNum() in REACTIVE_HETEROATOMS:
            sites.append(atom.GetIdx())
        elif atom.GetAtomicNum() == 6:
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    other = bond.GetOtherAtom(atom)
                    if other.GetAtomicNum() in (7, 8):
                        sites.append(atom.GetIdx())
                        break

    sites = sorted(set(sites))
    labels = [f"{rm.GetAtomWithIdx(i).GetSymbol()}@{i}" for i in sites]
    return {
        "atom_indices": sites,
        "label": ", ".join(labels) if labels else "unknown",
        "method": "heuristic",
    }


def tanimoto(smiles_a: str, smiles_b: str) -> Optional[float]:
    from rdkit.DataStructs import TanimotoSimilarity
    ma = Chem.MolFromSmiles(smiles_a)
    mb = Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return round(TanimotoSimilarity(fa, fb), 4)


def strip_atom_map(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def parse_source_file(path: str) -> list[dict]:
    """
    Parse source CSV: class, id, prod_smiles, rxn_smiles, prod_smiles_pop, keep
    col 0: reaction class  col 1: patent id  col 2: clean product SMILES
    col 3: full atom-mapped rxn SMILES       col 5: keep flag
    """
    rows = []
    with open(path) as f:
        raw_head = f.read(4096)
    delim = "," if raw_head.count(",") > raw_head.count("\t") else "\t"

    with open(path) as f:
        reader = csv.reader(f, delimiter=delim)
        for i, line in enumerate(reader):
            if not line:
                continue
            if i == 0 and not line[0].strip().lstrip("-").isdigit():
                continue
            if len(line) < 4:
                continue
            if len(line) >= 6:
                keep_val = line[5].strip().lower()
                if keep_val not in ("true", "1", "yes", ""):
                    continue

            reaction_class = line[0].strip()
            patent_id      = line[1].strip()
            prod_smiles    = line[2].strip()
            rxn_smiles     = line[3].strip()

            if ">>" not in rxn_smiles:
                continue

            reactant_part = rxn_smiles.split(">>")[0]
            reactant_mols = []
            for smi in reactant_part.split("."):
                cleaned = strip_atom_map(smi.strip())
                if cleaned and Chem.MolFromSmiles(cleaned) is not None:
                    reactant_mols.append(cleaned)
            if not reactant_mols:
                continue

            if Chem.MolFromSmiles(prod_smiles) is None:
                raw_prod = strip_atom_map(rxn_smiles.split(">>")[-1].split(".")[0])
                if not raw_prod or Chem.MolFromSmiles(raw_prod) is None:
                    continue
                prod_smiles = raw_prod

            reagent = ".".join(reactant_mols[1:]) if len(reactant_mols) > 1 else ""

            rows.append({
                "rxn_smiles":          rxn_smiles,
                "reactant_smiles":     reactant_mols[0],
                "all_reactant_smiles": ".".join(reactant_mols),
                "reagent_smiles":      reagent,
                "product_smiles":      prod_smiles,
                "reaction_class":      reaction_class,
                "patent_id":           patent_id,
            })

    print(f"  Parsed {len(rows)} valid reactions (delimiter='{delim}')")
    return rows


def build_coc_dataset(
    source_path: str,
    out_csv:  str = "coc_dataset.csv",
    out_json: str = "coc_dataset.json",
    max_rows: int = 3000,
) -> None:

    source_rows = parse_source_file(source_path)
    print(f"Loaded {len(source_rows)} reactions from {source_path}")

    records = []

    for row in source_rows[:max_rows]:
        reactant  = row["reactant_smiles"]
        product   = row["product_smiles"]
        rclass    = row["reaction_class"]
        rxn_smi   = row["rxn_smiles"]
        patent_id = row["patent_id"]

        # ── Step A: reactive center ──────────────────────────────────────────
        rc = get_reactive_center(reactant, product, rxn_smiles=rxn_smi)
        step_a_gold        = rc["label"]
        step_a_atom_indices = rc["atom_indices"]
        step_a_method      = rc["method"]   # "atom_map" | "heuristic"

        # ── Step B: intermediate ─────────────────────────────────────────────
        int_type, int_hint = _get_intermediate_hint(rclass)

        # ── Step C: product ──────────────────────────────────────────────────
        step_c_gold_smiles = product

        tani_reactant_to_product = tanimoto(reactant, product)

        reaction_id = f"{patent_id}_class{rclass}_coc"

        records.append({
            # ── Identifiers ──────────────────────────────────────────────────
            "reaction_id":    reaction_id,          # FIX 3
            "reaction_class": rclass,

            # ── Raw SMILES ───────────────────────────────────────────────────
            "reactant_smiles": reactant,
            "reagent_smiles":  row["reagent_smiles"],
            "product_smiles":  product,

            # ── Step A ───────────────────────────────────────────────────────
            "step_a_question":          "What is the reactive center (atom/bond) in the reactant?",
            "step_a_gold_label":        step_a_gold,
            "step_a_gold_atom_indices": json.dumps(step_a_atom_indices),
            "step_a_detection_method":  step_a_method,  # provenance for QA
            "step_a_evaluation":        "exact_match_vs_rdkit",

            # ── Step B ───────────────────────────────────────────────────────
            "step_b_question":               "What key intermediate forms during this reaction?",
            "step_b_gold_intermediate_type": int_type,  
            "step_b_gold_intermediate_hint": int_hint,  
            "step_b_gold_intermediate_smiles": "",       # reserved for RXNMapper annotation
            "step_b_evaluation":             "llm_as_judge + smiles_validity",

            # ── Step C ───────────────────────────────────────────────────────
            "step_c_question":   "What is the final product SMILES?",
            "step_c_gold_smiles": step_c_gold_smiles,
            "step_c_evaluation": "tanimoto_vs_gold",

            "step_c_tanimoto_reactant_to_product": tani_reactant_to_product,
            "step_c_tanimoto_predicted_vs_gold":   None,   # filled at eval time

            # ── Full CoC prompt ───────────────────────────────────────────────
            "coc_prompt": (
                f"You are a chemistry expert. Reason step by step.\n\n"
                f"Reactant SMILES: {reactant}\n"
                f"Reagents: {row['reagent_smiles'] or 'not specified'}\n\n"
                f"Step A — Reactive center: What atom or bond in the reactant will be "
                f"involved in the reaction? Give the atom symbol and index (e.g., O@3).\n\n"
                f"Step B — Key intermediate: What is the most important intermediate that "
                f"forms? Name the intermediate type and provide its SMILES if possible.\n\n"
                f"Step C — Final product: What is the SMILES of the major product?\n\n"
                f"Reply in this exact format:\n"
                f"STEP_A: <symbol>@<index>\n"
                f"STEP_B_TYPE: <intermediate name>\n"
                f"STEP_B_SMILES: <SMILES or 'unknown'>\n"
                f"STEP_C: <product SMILES>"
            ),

            # ── Diagnostic flag ───────────────────────────────────────────────
            "diagnostic_correct_product_wrong_mechanism": None,
        })

    print(f"Generated {len(records)} CoC records")

    if not records:
        print("No records generated — check source file path and format.")
        return

    keys = list(records[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)
    print(f"Saved CSV  -> {out_csv}")

    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved JSON -> {out_json}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Experiment 2: Chain-of-Chemistry dataset"
    )
    parser.add_argument(
        "--source", default="../data/uspto_50k/uspto_50k.csv",
        help="Path to USPTO-50K CSV (rxn_smiles, reaction_class, ...)"
    )
    parser.add_argument("--out_csv",  default="data/coc_dataset.csv")
    parser.add_argument("--out_json", default="data/coc_dataset.json")
    parser.add_argument("--max_rows", type=int, default=3000)
    args = parser.parse_args()

    build_coc_dataset(
        source_path=args.source,
        out_csv=args.out_csv,
        out_json=args.out_json,
        max_rows=args.max_rows,
    )
