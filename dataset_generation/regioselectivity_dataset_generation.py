"""
Experiment 1: Regioselectivity Stress Test Dataset Generator
Chemical Butterfly Effect — ReactionPerturbBench

Source: USPTO-50K reaction SMILES 
"""

import json
import csv
import argparse
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdchem import RWMol


# ---------------------------------------------------------------------------
# Perturbation rules  (SMARTS source -> SMARTS replacement)
# ---------------------------------------------------------------------------

PERTURBATIONS = {
    "electronic": [
        # EDG -> EWG
        ("[cH:1][OX2H0:2][CH3:3]", "[cH:1][N+:2](=O)[O-]"),           # -OCH3 -> -NO2
        ("[cH:1][NH2:2]",          "[cH:1][C:2](F)(F)F"),               # -NH2  -> -CF3
        ("[cH:1][OH:2]",           "[cH:1][C:2](=O)[OH]"),              # -OH   -> -COOH (aromatic)
        # EWG -> EDG
        ("[cH:1][N+:2](=O)[O-]",  "[cH:1][OX2H0:2][CH3:3]"),           # -NO2  -> -OCH3
        ("[cH:1][C:2](F)(F)F",    "[cH:1][NH2:2]"),                     # -CF3  -> -NH2
        ("[cH:1][C:2](=O)[OH]",   "[cH:1][OH:2]"),                      # -COOH -> -OH
    ],
    "isosteric": [
        ("[OH:1]",    "[SH:1]"),          # -OH  -> -SH
        ("[SH:1]",    "[OH:1]"),          # -SH  -> -OH
        ("[NH2:1]",   "[PH2:1]"),         # -NH2 -> -PH2
        ("[F:1]",     "[Cl:1]"),          # -F   -> -Cl
        ("[Cl:1]",    "[F:1]"),           # -Cl  -> -F
        ("[Br:1]",    "[I:1]"),           # -Br  -> -I
    ],
    "bioisosteric": [
        ("[C:1](=O)[OH]",   "[S:1](=O)(=O)[OH]"),    # -COOH  -> -SO3H
        ("[S:1](=O)(=O)[OH]","[C:1](=O)[OH]"),        # -SO3H  -> -COOH
        ("[C:1](=O)[NH2]",  "[S:1](=O)(=O)[NH2]"),   # -CONH2 -> -SO2NH2
    ],
}

# ---------------------------------------------------------------------------
# FIX 2: Reactive heteroatom set — now includes B and halogens
# ---------------------------------------------------------------------------

REACTIVE_HETEROATOMS = {
    5,   # B  — Suzuki / Miyaura boronic esters
    7,   # N  — amination, reductive amination
    8,   # O  — esterification, etherification
    15,  # P  — phosphorylation
    16,  # S  — thiolation
    17,  # Cl — SNAr, cross-coupling leaving group
    35,  # Br — cross-coupling leaving group
    53,  # I  — cross-coupling leaving group
}

# ---------------------------------------------------------------------------
# Shared prompt template (mirrors PREDICT_PROMPT in exp1_evaluate.py)
# ---------------------------------------------------------------------------

PREDICT_PROMPT_TEMPLATE = (
    "You are an expert organic chemist. A reactant has been chemically perturbed "
    "({perturbation_type} perturbation). Predict the major reaction product and "
    "identify the site of reaction.\n\n"
    "Perturbed reactant SMILES: {reactant}\n"
    "Perturbation type        : {perturbation_type}\n\n"
    "Instructions:\n"
    "1. Predict the major product SMILES after the most likely reaction.\n"
    "2. Identify the reactive center: give the atom symbol and its index in the "
    "reactant (e.g. O@3, N@7). If multiple sites, list the most important one first.\n\n"
    "Reply in this exact format:\n"
    "PRODUCT: <SMILES>\n"
    "SITE: <symbol>@<index>\n"
    "REASONING: <1-3 sentences>\n\n"
    "Answer:"
)



def apply_perturbation(smiles: str, smarts_from: str, smarts_to: str) -> str | None:
    """
    Apply a single SMARTS -> SMARTS substitution. Returns new SMILES or None.

    FIX 1: Previously called Chem.SanitizeMol() and used its integer bitmask
    return value in a boolean 'or' expression, meaning sanitisation failures
    were silently ignored. Now SanitizeMol is called for its side-effect only
    and raises SanitizationException on failure, which is caught below.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rxn = AllChem.ReactionFromSmarts(f"{smarts_from}>>{smarts_to}")
    if rxn is None:
        return None
    products = rxn.RunReactants((mol,))
    if not products:
        return None
    try:
        prod_mol = products[0][0]
        Chem.SanitizeMol(prod_mol)          # raises on failure — do NOT use return value
        prod = Chem.MolToSmiles(prod_mol)
        if Chem.MolFromSmiles(prod) is None:
            return None
        return prod
    except Exception:
        return None


def tanimoto(smiles_a: str, smiles_b: str) -> float | None:
    """Morgan fingerprint Tanimoto between two SMILES."""
    from rdkit.DataStructs import TanimotoSimilarity
    ma = Chem.MolFromSmiles(smiles_a)
    mb = Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return round(TanimotoSimilarity(fa, fb), 4)


def identify_reaction_sites(smiles: str) -> list[int]:
    """
    Return atom indices that are plausible reactive centers.

    FIX 2: Extended REACTIVE_HETEROATOMS to include boron (atomic num 5) and
    halogens Cl/Br/I (17/35/53). The original set {7,8,16,15} missed the boron
    atom in boronic esters — the reactive centre for Suzuki coupling (class 3).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    sites = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in REACTIVE_HETEROATOMS:
            sites.append(atom.GetIdx())
        elif atom.GetAtomicNum() == 6:
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    other = bond.GetOtherAtom(atom)
                    if other.GetAtomicNum() in (7, 8):
                        sites.append(atom.GetIdx())
                        break
    return sorted(set(sites))


def strip_atom_map(smiles: str) -> str:
    """Strip atom-map numbers from a plain molecule SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def parse_source_file(path: str) -> list[dict]:
    """
    Parse source CSV with columns:
        class, id, prod_smiles, rxn_smiles, prod_smiles_pop, keep

    col 0: reaction class (int)
    col 1: patent id
    col 2: prod_smiles       — clean product SMILES (no atom maps)
    col 3: rxn_smiles        — full atom-mapped reaction SMILES (reactants>>product)
    col 4: prod_smiles_pop   — (unused)
    col 5: keep              — filter flag (True/1 = include)

    Reactants are extracted from col 3 (before >>), atom maps stripped.
    Product is taken from col 2 directly (already clean).
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
            # Skip header row
            if i == 0 and not line[0].strip().lstrip("-").isdigit():
                continue
            if len(line) < 4:
                continue
            # Filter on keep column
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

            # Extract and clean all reactant molecules
            reactant_part = rxn_smiles.split(">>")[0]
            reactant_mols = []
            for smi in reactant_part.split("."):
                cleaned = strip_atom_map(smi.strip())
                if cleaned and Chem.MolFromSmiles(cleaned) is not None:
                    reactant_mols.append(cleaned)
            if not reactant_mols:
                continue

            # Validate product (col 2 is pre-cleaned; fall back to rxn product side)
            if Chem.MolFromSmiles(prod_smiles) is None:
                raw_prod = strip_atom_map(rxn_smiles.split(">>")[-1].split(".")[0])
                if not raw_prod or Chem.MolFromSmiles(raw_prod) is None:
                    continue
                prod_smiles = raw_prod

            rows.append({
                "rxn_smiles":          rxn_smiles,
                "reactant_smiles":     reactant_mols[0],
                "all_reactant_smiles": ".".join(reactant_mols),
                "product_smiles":      prod_smiles,
                "reaction_class":      reaction_class,
                "patent_id":           patent_id,
            })

    print(f"  Parsed {len(rows)} valid reactions from {path} (delimiter='{delim}')")
    return rows


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_regioselectivity_dataset(
    source_path: str,
    out_csv: str = "regioselectivity_dataset.csv",
    out_json: str = "regioselectivity_dataset.json",
    max_rows: int = 5000,
) -> None:

    source_rows = parse_source_file(source_path)
    print(f"Loaded {len(source_rows)} reactions from {source_path}")

    records = []
    skipped = 0

    for row in source_rows[:max_rows]:
        reactant = row.get("all_reactant_smiles", row["reactant_smiles"]).split(".")[0]
        product  = row["product_smiles"]

        for ptype, rules in PERTURBATIONS.items():
            for smarts_from, smarts_to in rules:
                perturbed = apply_perturbation(reactant, smarts_from, smarts_to)
                if perturbed is None or perturbed == reactant:
                    skipped += 1
                    continue

                tani_reactants = tanimoto(reactant, perturbed)
                reaction_sites = identify_reaction_sites(reactant)

                # Stable, human-readable reaction ID (avoids truncated SMILES keys)
                reaction_id = (
                    f"{row.get('patent_id', 'unk')}"
                    f"_{row['reaction_class']}"
                    f"_{ptype}"
                    f"_{smarts_from.replace('[','').replace(']','').replace(':1','')}"
                )

                records.append({
                    # --- Identifiers ---
                    "reaction_id":            reaction_id,
                    "reaction_class":         row["reaction_class"],

                    # --- Original ---
                    "original_reactant_smiles":  reactant,
                    "original_product_smiles":   product,
                    "original_reaction_sites":   json.dumps(reaction_sites),

                    # --- Perturbation ---
                    "perturbation_type":       ptype,
                    "smarts_applied_from":     smarts_from,
                    "smarts_applied_to":       smarts_to,
                    "perturbed_reactant_smiles": perturbed,

                    # --- Gold labels (model must predict) ---
                    # gold_product_smiles: resolved at eval time by _resolve_gold_product()
                    #   via SMARTS propagation (derived) or fallback to original product.
                    # gold_reaction_site_atom: populate post-hoc with RXNMapper annotation.
                    #   Format when filled: "B@4" or "B@4,N@7" (symbol@index, comma-separated).
                    "gold_product_smiles":      "",
                    "gold_reaction_site_atom":  "",

                    # --- Evaluation helpers ---
                    "tanimoto_reactant_similarity": tani_reactants,

                    # --- Prompts (FIX 3: aligned with evaluator's PREDICT_PROMPT) ---
                    "prompt_smiles": PREDICT_PROMPT_TEMPLATE.format(
                        reactant=perturbed,
                        perturbation_type=ptype,
                    ),
                    "prompt_iupac": "",   # populate with PubChem name lookup if needed
                })

    print(f"Generated {len(records)} perturbation pairs  ({skipped} skipped / no match)")

    # --- CSV ---
    if records:
        keys = list(records[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(records)
        print(f"Saved CSV  -> {out_csv}")

        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved JSON -> {out_json}")
    else:
        print("No records generated — check perturbation SMARTS against your reactant structures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Experiment 1: Regioselectivity dataset")
    parser.add_argument("--source", default="../data/uspto_50k/uspto_50k.csv",
                        help="Path to USPTO-50K TSV file (rxn_smiles<TAB>reaction_class)")
    parser.add_argument("--out_csv",  default="data/regioselectivity_dataset.csv")
    parser.add_argument("--out_json", default="data/regioselectivity_dataset.json")
    parser.add_argument("--max_rows", type=int, default=5000,
                        help="Max reactions to process (default 5000)")
    args = parser.parse_args()

    build_regioselectivity_dataset(
        source_path=args.source,
        out_csv=args.out_csv,
        out_json=args.out_json,
        max_rows=args.max_rows,
    )
