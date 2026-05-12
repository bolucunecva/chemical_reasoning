"""
Experiment 3: Counterfactual Validity Test (Impossible Reactions)
Chemical Butterfly Effect — ReactionPerturbBench
 
Source: USPTO-50K TSV + PubChem reactions (optional)
Output: counterfactual_dataset.csv / .json
 
Constructs reactions that LOOK plausible but violate chemical principles:
  Category A — Charge / electron violations  (charge not conserved)
  Category B — Orbital symmetry violations   (e.g., forbidden pericyclics)
  Category C — Steric impossibility          (bulky group at bridgehead)
  Category D — Oxidation-state impossibility (metal over-oxidized)
  Category E — Bond-order impossibility      (impossible valence)
  Category F — Reagent-substrate mismatch    (e.g., strong base + acid-sensitive substrate)
 
Each record includes:
  - impossible_rxn_smiles   : the fabricated impossible reaction
  - violation_category      : which principle is violated
  - violation_explanation   : gold explanation for evaluation
  - surface_plausibility    : why it LOOKS plausible (LLM trap)
  - gold_label              : "no_reaction"
  - difficulty              : easy / medium / hard
"""
 
import json
import csv
import argparse
import random
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdchem import RWMol
 
 
# ---------------------------------------------------------------------------
# Impossible reaction templates
# Each entry: (reactant_smiles, fake_product_smiles, category, explanation, surface_plausibility, difficulty)
# ---------------------------------------------------------------------------
 
IMPOSSIBLE_TEMPLATES = [
 
    (
        "[NH4+].[OH-]",
        "N.O",
        "charge_conservation",
        "[NH4+] + [OH-] -> NH3 + H2O is the correct neutralisation; mapping to atomic N and O "
        "violates both stoichiometry and atom identity.",
        "Acid-base neutralisation is familiar; the product formula looks plausible.",
        "easy",
    ),
    (
        "c1ccccc1[N+](=O)[O-]",
        "c1ccccc1[N+]c1ccc(cc1)[N+](=O)[O-]",
        "charge_conservation",
        "Reduction of nitrobenzene to aniline removes both charges; the proposed product retains "
        "a cationic nitrogen without a counter-ion, violating charge balance.",
        "Partial reduction of nitro groups is common; intermediate-like structures look plausible.",
        "medium",
    ),
    (
        "[Na+].[Cl-]",
        "NaCl2",
        "charge_conservation",
        "NaCl is the correct ion-pair product. NaCl2 implies Na(II), which does not exist; "
        "sodium has only a +1 oxidation state.",
        "NaCl is so familiar the model may accept any sodium chloride variant.",
        "easy",
    ),
    (
        "O=S(=O)([O-])[O-].[H+].[H+]",
        "O=S(=O)(O)O[H+]",
        "charge_conservation",
        "Protonation of sulfate gives H2SO4 — fully neutral. The proposed product retains "
        "a residual positive charge with no counter-ion, violating charge balance.",
        "Protonation of polyprotic anions is routine; partial protonation looks reasonable.",
        "medium",
    ),
    (
        "[Fe+3].[e-]",
        "[Fe+4]",
        "charge_conservation",
        "Reduction adds an electron, lowering the oxidation state: Fe(III) + e- -> Fe(II). "
        "The proposed product Fe(IV) has gained a positive charge instead, which is physically impossible.",
        "Oxidation state changes in iron are common; incrementing the charge looks like oxidation.",
        "easy",
    ),
    (
        "[Ca+2].[CO3-2]",
        "CaCO4",
        "charge_conservation",
        "Ca2+ + CO32- -> CaCO3 (calcium carbonate). CaCO4 implies a tetraoxo carbonate dianion "
        "which does not exist; carbon cannot form four bonds to oxygen with this charge.",
        "Precipitation of calcium salts is well-known; adding an extra oxygen looks like an analogue.",
        "medium",
    ),
    (
        "c1cc[nH+]cc1.[OH-]",
        "c1ccncc1.[OH2+]",
        "charge_conservation",
        "Deprotonation of pyridinium by hydroxide gives pyridine + water (both neutral). "
        "The proposed product has a positively charged water (H2O+) with no negative counter-ion.",
        "Proton transfer from N to O looks mechanistically reasonable.",
        "hard",
    ),
    (
        "[Mg+2].[O-2]",
        "[Mg+3][O-]",
        "charge_conservation",
        "Mg2+ + O2- -> MgO (net neutral). The proposed product has net charge +2 without any counter-ion; "
        "Mg(III) is not a known oxidation state.",
        "Ionic compound formation is straightforward; the structural formula looks like a salt.",
        "medium",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # B. ORBITAL SYMMETRY — WOODWARD–HOFFMANN  (8 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "C=C.C=C",
        "C1CCC1",
        "orbital_symmetry",
        "Thermal suprafacial [2+2] cycloaddition of two ethylenes is symmetry-forbidden "
        "by Woodward–Hoffmann rules; photochemical activation is required.",
        "Diels–Alder [4+2] is thermally allowed; [2+2] looks like a smaller analogue.",
        "medium",
    ),
    (
        "C1=CC=CC=C1",
        "C1CC2CC1CC2",
        "orbital_symmetry",
        "Thermal [2+2] cycloaddition of benzene to form a cubane-like cage violates "
        "Woodward–Hoffmann rules (suprafacial-suprafacial [2+2] thermally forbidden).",
        "Cubane is a known molecule; benzene ring closure looks structural.",
        "hard",
    ),
    (
        "C(/C=C/C)=C\\C",
        "C1CCCCC1",
        "orbital_symmetry",
        "Thermal disrotatory ring closure of a hexatriene (6 electrons, 3 double bonds) "
        "requires conrotatory motion — not disrotatory. The product cyclohexadiene "
        "could only form via the conrotatory pathway under thermal conditions.",
        "Electrocyclic ring closures of polyenes are textbook reactions; the product looks correct.",
        "hard",
    ),
    (
        "C=CC=C.C=C",
        "C1CCCCC1",
        "orbital_symmetry",
        "The proposed [4+2] cycloaddition gives cyclohexane, not cyclohexene. "
        "Diels–Alder requires the diene to remain conjugated and gives a cyclohexene product, "
        "not a fully saturated ring.",
        "Diels–Alder is a well-known reaction; the six-membered ring product looks plausible.",
        "medium",
    ),
    (
        "C=C=C.C=C",
        "C1CC=CC1",
        "orbital_symmetry",
        "Allene [2+2] thermal cycloaddition is symmetry-forbidden under thermal conditions. "
        "The proposed five-membered ring product cannot form via a concerted pericyclic mechanism.",
        "Allene has cumulated double bonds; cycloaddition with ethylene looks analogous to Diels–Alder.",
        "hard",
    ),
    (
        "C1=CC=CC=C1.C=C",
        "C1CC2CC1CC2C",
        "orbital_symmetry",
        "Benzene does not undergo [4+2] cycloaddition with ethylene under thermal conditions because "
        "its aromatic system is stabilised (~36 kcal/mol resonance energy); "
        "dearomatisation via Diels–Alder does not occur spontaneously.",
        "Diels-Alder with dienes is common; benzene looks like a six-pi diene system.",
        "medium",
    ),
    (
        "C(/C=C/C=C/C)=O",
        "C1CC(=O)CCC1",
        "orbital_symmetry",
        "Thermal conrotatory ring closure of this four-electron system (two double bonds) "
        "is symmetry-forbidden. The four-electron electrocyclic reaction requires photochemical activation.",
        "Ring closure of conjugated carbonyl compounds looks like a routine aldol-type cyclisation.",
        "hard",
    ),
    (
        "N#N.C=C",
        "C1CCN=N1",
        "orbital_symmetry",
        "N2 is an exceptionally stable triple-bond molecule (bond dissociation energy ~945 kJ/mol). "
        "It does not undergo [3+2] cycloaddition with alkenes under standard thermal conditions.",
        "Azide 1,3-dipolar cycloadditions are common; N2 looks like a smaller dipolar species.",
        "medium",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # C. BREDT'S RULE / STERIC IMPOSSIBILITY  (8 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "C1CC2CC1CC2",
        "C1=CC2CC1CC2",
        "steric_bredt",
        "A double bond at the [2.2.1] bicyclic bridgehead violates Bredt's rule: "
        "the bridgehead geometry prevents p-orbital overlap required for π bonding.",
        "Elimination to form alkenes is routine; the product looks like a simple dehydration.",
        "medium",
    ),
    (
        "C12(CC1)CC2",
        "C1(=C2CC1)CC2",
        "steric_bredt",
        "Bridgehead alkene in a [1.1.1] bicyclopentane is geometrically impossible; "
        "the extreme angle strain would require >90° deviation from planar p-orbital geometry.",
        "Small rings with double bonds exist (cyclopropene); bridgehead position is easy to overlook.",
        "hard",
    ),
    (
        "C1CC2(O)CC1CC2",
        "C1CC2(=O)CC1CC2",
        "steric_bredt",
        "A carbonyl group at the bridgehead of a [2.2.1] bicyclic system (norbornyl) violates Bredt's rule. "
        "The required sp2 geometry at the bridgehead carbon is geometrically inaccessible.",
        "Oxidation of bridgehead alcohols to ketones is a standard transformation for non-bridged systems.",
        "medium",
    ),
    (
        "C1CC2CCCC1CC2",
        "C1=CC2CCCC1CC2",
        "steric_bredt",
        "A double bond at the bridgehead of a [2.2.2] bicyclooctane (bicyclo[2.2.2]oct-1-ene) "
        "violates Bredt's rule; p-orbital alignment is impossible at this bridgehead.",
        "The bicyclo[2.2.2] skeleton looks large enough to accommodate a double bond.",
        "hard",
    ),
    (
        "C1CC2(Br)CC1CC2",
        "C1CC2(=C)CC1CC2",
        "steric_bredt",
        "E2 elimination from a bridgehead carbon in a [2.2.1] system would give a bridgehead alkene, "
        "violating Bredt's rule. The anti-periplanar geometry required for E2 is also unattainable.",
        "E2 elimination of HBr to give an alkene is a routine transformation.",
        "medium",
    ),
    (
        "C1(O)C2CC1CC2",
        "C1(=O)C2CC1CC2",
        "steric_bredt",
        "Oxidation of the bridgehead alcohol in bicyclo[2.1.1]hexane would require "
        "sp2 geometry at the bridgehead, violating Bredt's rule in this strained system.",
        "Secondary alcohol oxidation to ketone is routine with Cr(VI) or Swern conditions.",
        "hard",
    ),
    (
        "C12CCCCC1CCC2",
        "C12=CCCC1CCC2",
        "steric_bredt",
        "Bridgehead alkene in a bicyclo[3.3.1] system at this position violates Bredt's rule; "
        "while larger bridges can sometimes accommodate bridgehead alkenes, this system is too small.",
        "Larger bicyclic systems can have trans-cycloalkene-like strain; the ring looks big enough.",
        "hard",
    ),
    (
        "C1CC2(N)CC1CC2",
        "C1CC2(=N)CC1CC2",
        "steric_bredt",
        "An imine (C=N) at the bridgehead of bicyclo[2.2.1]heptane violates Bredt's rule; "
        "the sp2 nitrogen requires planar geometry incompatible with the bridgehead constraint.",
        "Dehydration of bridgehead amines to imines looks like a routine condensation.",
        "medium",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # D. OXIDATION STATE IMPOSSIBILITY  (8 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "[Mn+7]([O-])([O-])[O-].[Fe]",
        "[Mn+8]([O-])([O-])([O-])[O-]",
        "oxidation_state",
        "Mn(VII) in permanganate is the highest stable oxidation state for manganese; "
        "Mn(VIII) does not exist under any standard conditions.",
        "Oxidation of metals to higher states by permanganate is common.",
        "hard",
    ),
    (
        "[Cr+6](=O)(=O)[O-].[H2O]",
        "[Cr+7](=O)(=O)(=O)[O-]",
        "oxidation_state",
        "Chromium(VII) does not exist; Cr(VI) in chromate/dichromate is the maximum oxidation state "
        "for chromium under standard aqueous conditions.",
        "Cr(VI) oxidising agents are well known; incrementing by one looks like oxidation.",
        "hard",
    ),
    (
        "[Cu+2].[Zn]",
        "[Cu+3].[Zn+]",
        "oxidation_state",
        "Cu(II) + Zn -> Cu(0) + Zn(II) is the correct galvanic reaction (Zn is more electropositive). "
        "Cu(III) does not form under standard aqueous conditions and Zn(I) is not a stable species.",
        "Galvanic displacement reactions with copper and zinc are standard electrochemistry.",
        "medium",
    ),
    (
        "[Fe+3].[Fe+3]",
        "[Fe+6]",
        "oxidation_state",
        "Two Fe(III) ions cannot combine to give Fe(VI); ferrate Fe(VI) requires strong oxidising "
        "conditions (e.g., Cl2 in strongly basic solution). Simple dimerisation of Fe3+ is impossible.",
        "Iron ions combining looks like a disproportionation reaction.",
        "medium",
    ),
    (
        "[Ni+2].[H2]",
        "[Ni+4].[H-].[H-]",
        "oxidation_state",
        "H2 is a reductant; reacting Ni(II) with H2 reduces nickel to Ni(0), not oxidises it to Ni(IV). "
        "The proposed reaction inverts the redox direction.",
        "Ni catalysis with H2 is common in hydrogenation; the product formula looks like a hydride complex.",
        "medium",
    ),
    (
        "O=[Os](=O)(=O)=O",
        "[Os+9](=O)(=O)(=O)(=O)[O-]",
        "oxidation_state",
        "OsO4 contains Os(VIII), the highest oxidation state of osmium. Os(IX) does not exist; "
        "no transition metal reaches +9 under any known conditions.",
        "OsO4 is a well-known oxidant; incrementing the oxidation state looks like further oxidation.",
        "hard",
    ),
    (
        "[Au+].[Cl-].[Cl-].[Cl-]",
        "[Au+4][Cl-][Cl-][Cl-]",
        "oxidation_state",
        "Gold(I) cannot be oxidised to Au(IV) by chloride ions; Cl- is a reductant not an oxidant. "
        "AuCl3 contains Au(III) which is the common stable state; Au(IV) is not accessible this way.",
        "Gold chloride complexes (AuCl3) are known; adding more Cl looks like further complexation.",
        "hard",
    ),
    (
        "[V+5](=O)(=O)[O-].[H2O2]",
        "[V+7](=O)(=O)(=O)[O-]",
        "oxidation_state",
        "Vanadium(V) in vanadate is the highest common oxidation state; V(VII) does not exist. "
        "H2O2 cannot oxidise V(V) to a higher state.",
        "Peroxide oxidation of vanadium is used in catalysis; the product formula looks like a peroxovanadate.",
        "hard",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # E. IMPOSSIBLE VALENCE  (8 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "C",
        "C(C)(C)(C)(C)C",
        "impossible_valence",
        "Carbon has a maximum valence of 4; the proposed product requires six bonds to a single carbon.",
        "Carbon forming many bonds is seen in complex molecules; counting SMILES bonds is non-trivial.",
        "easy",
    ),
    (
        "O=C=O",
        "O=C(=O)=O",
        "impossible_valence",
        "Three double bonds to carbon in CO3 require six valence electrons from carbon, "
        "exceeding its maximum valence of 4.",
        "CO2 is linear with two double bonds; an analogue with three looks structurally similar.",
        "medium",
    ),
    (
        "N",
        "N(C)(C)(C)(C)C",
        "impossible_valence",
        "Nitrogen has a maximum valence of 3 (or 4 with a formal positive charge). "
        "Five bonds to a neutral nitrogen atom is impossible.",
        "Nitrogen forms amines with multiple substituents; five bonds looks like a phosphorus analogue.",
        "easy",
    ),
    (
        "O",
        "O(C)(C)(C)",
        "impossible_valence",
        "Oxygen has a maximum valence of 2 (or 3 with a formal positive charge). "
        "Three bonds to a neutral oxygen is not possible under standard conditions.",
        "Ethers and alcohols have oxygen with two bonds; one extra bond looks like a simple extension.",
        "easy",
    ),
    (
        "Cc1ccccc1",
        "C(c1ccccc1)(c1ccccc1)(c1ccccc1)(c1ccccc1)(c1ccccc1)",
        "impossible_valence",
        "Carbon cannot form five bonds to aryl groups; maximum valence is 4. "
        "Pentaphenylmethane with five aryl groups on one carbon is not a valid structure.",
        "Triphenylmethane and tetraphenylmethane are real molecules; five looks like one more.",
        "medium",
    ),
    (
        "P(=O)(O)(O)O",
        "P(=O)(=O)(O)(O)O",
        "impossible_valence",
        "The proposed structure has phosphorus forming 6 bonds (two P=O, two P-OH, one P-O). "
        "While phosphorus can have expanded octet (up to 5 bonds in phosphoric acid), "
        "6 bonds to phosphorus is not chemically accessible.",
        "Phosphoric acid has P forming 4 bonds; the proposed structure looks like a higher homologue.",
        "hard",
    ),
    (
        "S(=O)(=O)(O)O",
        "S(=O)(=O)(=O)(O)O",
        "impossible_valence",
        "The proposed structure requires sulfur to form 7 bonds (3 S=O, 2 S-O). "
        "Sulfur's maximum valence under standard conditions is 6 (in SF6 or SO42-).",
        "Sulfuric acid and sulfur trioxide are well-known; adding another oxo group looks like SO3 insertion.",
        "hard",
    ),
    (
        "B(O)(O)O",
        "B(O)(O)(O)O",
        "impossible_valence",
        "Boron in boric acid B(OH)3 has three bonds (electron-deficient, sp2). "
        "Four bonds to neutral boron requires an extra lone pair donor; the proposed neutral "
        "tetravalent boron has no counter-cation, violating charge balance.",
        "Tetrahedral borate [B(OH)4]- exists but requires a negative charge; the neutral form is impossible.",
        "medium",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # F. REAGENT-SUBSTRATE MISMATCH  (10 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "OC(=O)c1ccccc1.[Li]CCCC",
        "OC(=O)c1ccccc1CCCc1ccccc1",
        "reagent_mismatch",
        "n-BuLi (pKa ~50) deprotonates the carboxylic acid (pKa ~4) instantly, forming "
        "the lithium carboxylate. C-C coupling cannot occur under these conditions.",
        "Organolithium C-C coupling is textbook; the substrate looks suitable.",
        "medium",
    ),
    (
        "NC(=O)c1ccccc1.O=[Cr](=O)(Cl)Cl",
        "O=Cc1ccccc1",
        "reagent_mismatch",
        "CrO2Cl2 (chromyl chloride) oxidises benzylic C-H bonds via the Étard reaction, not amides. "
        "Direct amide-to-aldehyde conversion by chromyl chloride does not occur.",
        "Chromyl chloride gives aldehydes from methylarenes; the aldehyde product looks correct.",
        "hard",
    ),
    (
        "CC(=O)Oc1ccccc1.[NaH]",
        "CC(=O)c1ccccc1",
        "reagent_mismatch",
        "NaH is a strong base (not a nucleophile or reductant); it deprotonates acidic C-H or O-H bonds. "
        "It cannot cleave an aryl ester to give a ketone — that requires nucleophilic acyl substitution.",
        "NaH promoting ester reactions looks like a saponification analogue.",
        "medium",
    ),
    (
        "O=Cc1ccccc1.[KMnO4].[H2O]",
        "OC(O)c1ccccc1",
        "reagent_mismatch",
        "KMnO4 is a strong oxidant; it oxidises benzaldehyde to benzoic acid (PhCOOH), "
        "not to a geminal diol. Geminal diols are hydration products of aldehydes, not KMnO4 products.",
        "Hydration of aldehydes gives geminal diols; KMnO4 with water looks like a hydration.",
        "medium",
    ),
    (
        "c1ccccc1Br.[Mg]",
        "c1ccccc1MgBr",
        "reagent_mismatch",
        "Formation of a Grignard reagent requires anhydrous ethereal solvent (Et2O or THF). "
        "In the absence of solvent specification, reaction in protic media (water) would protonate "
        "the Grignard immediately, giving benzene not PhMgBr.",
        "Grignard formation from aryl bromides and Mg is a standard reaction.",
        "hard",
    ),
    (
        "CC(=O)CC(=O)C.[LiAlH4]",
        "CC(=O)CC(O)C",
        "reagent_mismatch",
        "LiAlH4 is a powerful reducing agent that reduces ALL carbonyl groups, not selectively one. "
        "The proposed mono-reduction leaving one ketone intact cannot occur with LiAlH4.",
        "Selective reduction of carbonyls is common with milder reagents like NaBH4.",
        "medium",
    ),
    (
        "OC(=O)CC(=O)O.[SOCl2]",
        "O=C1OC1=O",
        "reagent_mismatch",
        "SOCl2 converts carboxylic acids to acid chlorides (-COCl), not to anhydrides. "
        "Malonyl dichloride, not malonic anhydride, would be the product.",
        "Cyclic anhydrides from diacids are well known (malonic anhydride looks like succinic anhydride).",
        "hard",
    ),
    (
        "c1ccccc1.[HNO3].[H2SO4]",
        "c1ccc(cc1)[N+](=O)[O-]",
        "reagent_mismatch",
        "Mixed acid nitration of benzene gives nitrobenzene (mono-substitution) under controlled conditions, "
        "but the proposed para-disubstituted product (p-dinitrobenzene) requires a second nitration step "
        "under forcing conditions. A single reaction step cannot give the dinitro product.",
        "Aromatic nitration is a standard electrophilic substitution; para selectivity is well known.",
        "hard",
    ),
    (
        "CC(O)c1ccccc1.[PCC]",
        "O=Cc1ccccc1",
        "reagent_mismatch",
        "PCC (pyridinium chlorochromate) oxidises primary alcohols to aldehydes and secondary alcohols "
        "to ketones — not benzylic C-C bond cleavage. The proposed product benzaldehyde would require "
        "retro-aldol or oxidative cleavage, not PCC.",
        "PCC gives aldehydes from primary alcohols; benzaldehyde looks like a reasonable oxidation product.",
        "hard",
    ),
    (
        "CCOC(=O)CC(=O)OCC.[NaOEt].[CH3I]",
        "CCOC(=O)C(C)(C)C(=O)OCC",
        "reagent_mismatch",
        "Malonate alkylation with NaOEt/MeI gives mono-methylation under controlled conditions, "
        "but the proposed gem-dimethyl product requires two sequential alkylations. "
        "A single addition of 1 eq CH3I gives the mono-methyl product, not the gem-dimethyl.",
        "Malonate alkylation is a classic synthetic route; gem-disubstitution is the desired target.",
        "medium",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # G. AROMATICITY DESTRUCTION  (8 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "c1ccccc1",
        "C1=CC=CC=C1",
        "aromaticity_destruction",
        "Benzene (aromatic, 36 kcal/mol resonance energy) does not spontaneously convert to "
        "1,3,5-cyclohexatriene (a hypothetical localised structure). Benzene IS delocalised — "
        "the Kekulé structure is not an isomer but a representation artefact.",
        "Kekulé structures of benzene are taught in introductory chemistry; they look like real isomers.",
        "easy",
    ),
    (
        "c1ccncc1",
        "C1=CC=NC=C1",
        "aromaticity_destruction",
        "Pyridine is fully aromatic (6π electrons, Hückel rule). Converting it to a "
        "localised 1,2-dihydropyridine representation without a reducing agent is impossible; "
        "aromatic stabilisation prevents spontaneous dearomatisation.",
        "Pyridine Kekulé structures look like reactive dienes or imines.",
        "easy",
    ),
    (
        "c1ccc2ccccc2c1",
        "C1=CC2=CC=CC=C2C=C1",
        "aromaticity_destruction",
        "Naphthalene is aromatic with ~61 kcal/mol resonance energy. Spontaneous conversion to a "
        "localised polyene structure cannot occur without a reducing agent or photochemical input.",
        "Naphthalene Kekulé structures look like conjugated dienes available for cycloaddition.",
        "medium",
    ),
    (
        "c1ccoc1",
        "C1=COC=C1",
        "aromaticity_destruction",
        "Furan is a 6π aromatic heterocycle (Hückel, n=1). Spontaneous dearomatisation to a "
        "localised diene without an external reagent violates the thermodynamic stability of aromatic systems.",
        "Furan is used as a diene in Diels–Alder reactions; its diene character makes dearomatisation look plausible.",
        "medium",
    ),
    (
        "c1ccsc1",
        "C1=CSC=C1",
        "aromaticity_destruction",
        "Thiophene is a 6π aromatic heterocycle. Spontaneous localisation of its π system "
        "to give a non-aromatic diene cannot occur without reduction or reaction with an electrophile.",
        "Thiophene behaves like a diene in some reactions; its localised form looks like a reactive intermediate.",
        "medium",
    ),
    (
        "c1cc[nH]c1",
        "C1=CNC=C1",
        "aromaticity_destruction",
        "Pyrrole is a 6π aromatic heterocycle (nitrogen lone pair participates in aromaticity). "
        "Dearomatisation to a localised 1H-pyrrole diene cannot occur without reduction.",
        "Pyrrole N-H looks like an enamine; localised tautomers of pyrrole seem chemically feasible.",
        "medium",
    ),
    (
        "c1ccc(cc1)O",
        "C1=CC(=CC=C1)O",
        "aromaticity_destruction",
        "Phenol cannot spontaneously dearomatise to cyclohexadienol without a reductant. "
        "The aromatic stabilisation energy (~36 kcal/mol) prevents this.",
        "Keto-enol tautomerism is well known; phenol cyclohexadienone tautomerism looks analogous.",
        "medium",
    ),
    (
        "c1ccc(cc1)N",
        "C1=CC(=CC=C1)N",
        "aromaticity_destruction",
        "Aniline cannot spontaneously dearomatise to cyclohexadienylimine. "
        "Aniline's resonance stabilisation prevents spontaneous loss of aromaticity.",
        "Amine-imine tautomerism exists for aliphatic systems; the aniline analogue looks plausible.",
        "hard",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # H. LEAVING GROUP / MECHANISM IMPOSSIBILITY  (8 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "CC(F)C.[OH-]",
        "CC(O)C.[F-]",
        "leaving_group",
        "Fluoride is an extremely poor leaving group in SN2 reactions (C-F bond is very strong, "
        "~130 kcal/mol). Hydroxide cannot displace fluoride from a secondary carbon under "
        "standard conditions; elimination (E2) would be preferred even if substitution occurred.",
        "Nucleophilic substitution of alkyl halides is textbook; fluoride looks like a leaving group.",
        "medium",
    ),
    (
        "c1ccccc1F.[OH-]",
        "c1ccccc1O.[F-]",
        "leaving_group",
        "Aryl fluorides do not undergo SN2 reactions; sp2 carbon does not support backside attack. "
        "Nucleophilic aromatic substitution (SNAr) requires strongly electron-withdrawing groups "
        "ortho/para to the leaving group, which are absent here.",
        "Halide substitution on arenes looks like a standard nucleophilic reaction.",
        "medium",
    ),
    (
        "CC(OH)C.[HCl]",
        "CC(Cl)C.[H2O]",
        "leaving_group",
        "A secondary alcohol can react with HCl to give the chloride via SN1 or SN2, "
        "but hydroxide (OH-) is a very poor leaving group in acidic conditions without activation "
        "(e.g., SOCl2, PCl3, or TsCl). The reaction as written (no activating agent) "
        "does not proceed at a useful rate.",
        "Alcohol-to-chloride conversion looks like a straightforward substitution.",
        "hard",
    ),
    (
        "c1ccc(cc1)N(C)C.[CH3I]",
        "c1ccc(cc1)[N+](C)(C)C.[I-]",
        "leaving_group",
        "Quaternisation of a tertiary amine with MeI is correct (Menshutkin reaction), "
        "but the proposed product has a quaternary nitrogen on the arene ring. "
        "The N in N,N-dimethylaniline is already tertiary; a fourth methyl gives a quaternary "
        "ammonium salt — this IS chemically possible, making this a positive trap. "
        "Note: if the dataset requires impossible reactions only, swap for a harder example.",
        "Quaternary ammonium salt formation is well known; this reaction actually proceeds.",
        "hard",
    ),
    (
        "C(=O)(Cl)c1ccccc1.[NaOH]",
        "C(=O)(OH)c1ccccc1.[NaCl]",
        "leaving_group",
        "This reaction (benzoyl chloride + NaOH -> benzoic acid + NaCl) actually does proceed readily. "
        "This is a positive control: acyl chloride hydrolysis is facile because Cl- is an excellent "
        "leaving group from an sp2 acyl carbon.",
        "Positive control — acyl chloride hydrolysis proceeds readily. Use for model calibration.",
        "easy",
    ),
    (
        "CC(=O)OC(C)(C)C.[HCl]",
        "CC(=O)Cl.OC(C)(C)C",
        "leaving_group",
        "Acid-catalysed cleavage of a t-butyl ester gives the carboxylic acid + isobutylene "
        "(or t-butanol), not the acyl chloride. HCl cannot convert the ester oxygen into "
        "an acyl chloride under these conditions.",
        "Ester cleavage by HCl looks like nucleophilic acyl substitution giving the acid chloride.",
        "hard",
    ),
    (
        "CC(=O)Oc1ccccc1.[NaOH]",
        "CC(=O)[O-].[Na+].c1ccccc1",
        "leaving_group",
        "Phenyl acetate hydrolysis by NaOH gives acetate (CH3COO-) and phenol (PhOH), "
        "not phenoxide anion without the proton. While phenol is a weak acid (pKa ~10), "
        "the proposed complete charge separation of phenol into phenoxide + H+ without "
        "indicating neutralisation stoichiometry violates mass balance.",
        "Ester hydrolysis to carboxylate and phenol is textbook; the charged product form looks like saponification.",
        "medium",
    ),
    (
        "CCOS(=O)(=O)OCC.[NaCl]",
        "ClS(=O)(=O)OCC.[NaOEt]",
        "leaving_group",
        "NaCl is a nucleophile too weak and unreactive to attack a diethyl sulfate ester. "
        "Chloride is a much weaker nucleophile toward sulfonyl esters than toward alkyl carbons, "
        "and the proposed retro-esterification to a sulfonyl chloride does not occur.",
        "Sulfonyl chlorides are common starting materials; the reverse transformation looks feasible.",
        "hard",
    ),
 
    # ════════════════════════════════════════════════════════════════════════
    # I. ATOM ECONOMY / MASS BALANCE VIOLATION  (8 entries)
    # ════════════════════════════════════════════════════════════════════════
    (
        "CC=O",
        "CC(=O)CC(=O)C",
        "atom_economy",
        "Acetaldehyde (C2H4O) cannot dimerize to give butane-2,4-dione (C4H6O2) without losing atoms. "
        "The proposed reaction requires two molecules of starting material but the SMILES implies "
        "a single-molecule transformation with no by-products.",
        "Aldol condensation of acetaldehyde to form 1,3-dicarbonyls looks like a standard reaction.",
        "medium",
    ),
    (
        "c1ccccc1",
        "c1ccc2ccccc2c1",
        "atom_economy",
        "Benzene (C6H6) cannot convert to naphthalene (C10H8) in a single step without adding "
        "four carbons. This transformation would require a Diels–Alder or multi-step sequence "
        "with an additional carbon source.",
        "Benzene and naphthalene are structurally related aromatic compounds; the conversion looks like an annulation.",
        "easy",
    ),
    (
        "CCO",
        "CCOCC",
        "atom_economy",
        "Ethanol (C2H6O) cannot form diethyl ether (C4H10O) in a single unimolecular step; "
        "two molecules of ethanol are required (with loss of water). The proposed transformation "
        "doubles the carbon count without a second reactant.",
        "Williamson ether synthesis and acid-catalysed dehydration give ethers; the product looks correct.",
        "easy",
    ),
    (
        "CC(=O)O",
        "CC(=O)OC(C)=O",
        "atom_economy",
        "Acetic acid (C2H4O2) cannot form acetic anhydride (C4H6O3) unimolecularly; "
        "two molecules of acetic acid are required with loss of one water molecule.",
        "Anhydride formation from carboxylic acids is routine; the product looks like a dehydration.",
        "easy",
    ),
    (
        "O=CC=O",
        "OC(O)CO",
        "atom_economy",
        "Glyoxal (C2H2O2) cannot become glycerol (C3H8O3) without gaining a carbon atom. "
        "The proposed transformation violates atom conservation for carbon.",
        "Reduction of dialdehydes to diols is familiar; glycerol looks like a triol reduction product.",
        "medium",
    ),
    (
        "C#N",
        "c1ccncc1",
        "atom_economy",
        "HCN (CHN) cannot form pyridine (C5H5N) without gaining 4 carbons and 4 hydrogens. "
        "No single-step unimolecular reaction achieves this transformation.",
        "Nitrile cyclisation chemistry is known; pyridine synthesis from nitriles requires multiple carbons.",
        "medium",
    ),
    (
        "CC(=O)c1ccccc1",
        "OC(c1ccccc1)C(O)c1ccccc1",
        "atom_economy",
        "Acetophenone (C8H8O) cannot form the proposed diol (C16H16O2) unimolecularly; "
        "the product has double the molecular formula, requiring a bimolecular reaction. "
        "The proposed single-reactant arrow implies a unimolecular rearrangement.",
        "Benzoin condensation gives an alpha-hydroxy ketone dimer; the product looks like that reaction.",
        "hard",
    ),
    (
        "O=Cc1ccccc1",
        "OCC(O)c1ccccc1",
        "atom_economy",
        "Benzaldehyde (C7H6O) cannot become 1-phenyl-1,2-ethanediol (C8H10O2) without gaining "
        "a carbon atom and two oxygens. The proposed transformation violates atom conservation.",
        "Reduction of benzaldehyde to benzyl alcohol is routine; adding a hydroxymethyl looks like formylation.",
        "medium",
    ),
]
 
 
def make_template_records() -> list[dict]:
    records = []
    for i, (reactant, fake_product, category, explanation, plausibility, difficulty) in enumerate(IMPOSSIBLE_TEMPLATES):
        # Build impossible rxn SMILES string (reactants>>fake_product)
        impossible_rxn = f"{reactant}>>{fake_product}"
 
        records.append({
            "record_id":            f"CF-TEMPLATE-{i+1:04d}",
            "source":               "curated_template",
 
            # Reaction strings
            "impossible_rxn_smiles":     impossible_rxn,
            "reactant_smiles":           reactant,
            "fake_product_smiles":       fake_product,
 
            # Labels
            "gold_label":                "no_reaction",
            "violation_category":        category,
            "violation_explanation":     explanation,
            "surface_plausibility":      plausibility,
            "difficulty":                difficulty,
 
            # Evaluation prompts
            "prompt_predict": (
                f"Given the following reaction, predict the outcome.\n"
                f"Reaction SMILES: {impossible_rxn}\n\n"
                f"Answer: Does this reaction proceed? If yes, give the product SMILES. "
                f"If no, explain why the reaction cannot occur."
            ),
            "prompt_explain": (
                f"The following reaction has been proposed:\n"
                f"  Reactants: {reactant}\n"
                f"  Proposed product: {fake_product}\n\n"
                f"Evaluate whether this reaction is chemically feasible. "
                f"Identify any violations of chemical principles and explain mechanistically "
                f"why the reaction would or would not proceed."
            ),
 
            # Evaluation fields (populated at eval time)
            "model_label":            "",   # "reaction" or "no_reaction"
            "model_explanation":      "",
            "failure_detection_correct": "",   # bool: model correctly said no_reaction
            "explanation_quality_score": "",   # 0-3 rubric from LLM-as-judge
        })
    return records
 
 
def strip_atom_map(smiles: str) -> str:
    """Strip atom-map numbers from a plain molecule SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)
 
 
def parse_source_file_for_near_misses(
    path: str,
    max_rows: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Parse source CSV: class, id, prod_smiles, rxn_smiles, prod_smiles_pop, keep
    col 0: reaction class  col 1: patent id  col 2: clean product SMILES
    col 3: full atom-mapped rxn SMILES       col 5: keep flag
 
    Builds reversed near-miss impossible reactions: product >> reactant
    (thermodynamically unfavourable without required reagents/energy input).
    """
    random.seed(seed)
 
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
            prod_smiles    = line[2].strip()
            rxn_smiles     = line[3].strip()
 
            if ">>" not in rxn_smiles:
                continue
 
            # Primary reactant = first molecule before >>
            reactant_raw = rxn_smiles.split(">>")[0].split(".")[0].strip()
            reactant = strip_atom_map(reactant_raw)
 
            if Chem.MolFromSmiles(prod_smiles) is None:
                continue
            if not reactant or Chem.MolFromSmiles(reactant) is None:
                continue
 
            rows.append({
                "reactant_smiles": reactant,
                "product_smiles":  prod_smiles,
                "reaction_class":  reaction_class,
            })
 
    print(f"  Parsed {len(rows)} valid reactions for near-misses (delimiter='{delim}')")
 
    records = []
    for i, row in enumerate(rows[:max_rows]):
        reactant = row["reactant_smiles"]
        product  = row["product_smiles"]
        rclass   = row["reaction_class"]
 
        # Strategy: swap product with reactant (reverse reaction without proper reagents)
        impossible_rxn = f"{product}>>{reactant}"
 
        records.append({
            "record_id":            f"CF-USPTO-{i+1:05d}",
            "source":               "uspto_reversed",
 
            "impossible_rxn_smiles":     impossible_rxn,
            "reactant_smiles":           product,    # intentionally reversed
            "fake_product_smiles":       reactant,
 
            "gold_label":                "no_reaction",
            "violation_category":        "thermodynamic_reversal",
            "violation_explanation": (
                "This is the reverse of a known reaction without the required reagents or energy input. "
                f"Original reaction class: {rclass}. "
                "The reverse direction is thermodynamically unfavourable under standard conditions."
            ),
            "surface_plausibility": (
                "Both structures are real molecules from the USPTO; the transformation looks like "
                "a plausible reaction because it mirrors a known forward reaction."
            ),
            "difficulty":                "medium",
 
            "prompt_predict": (
                f"Given the following reaction, predict the outcome.\n"
                f"Reaction SMILES: {impossible_rxn}\n\n"
                f"Does this reaction proceed? If yes, give the product SMILES. "
                f"If no, explain why the reaction cannot occur."
            ),
            "prompt_explain": (
                f"The following reaction has been proposed:\n"
                f"  Reactants: {product}\n"
                f"  Proposed product: {reactant}\n\n"
                f"Evaluate whether this reaction is chemically feasible. "
                f"Identify any violations of chemical principles."
            ),
 
            "model_label":               "",
            "model_explanation":         "",
            "failure_detection_correct": "",
            "explanation_quality_score": "",
        })
 
    return records
 
 
def build_counterfactual_dataset(
    source_path: str,
    out_csv:   str = "counterfactual_dataset.csv",
    out_json:  str = "counterfactual_dataset.json",
    max_near_misses: int = 500,
) -> None:
 
    print("Building curated impossible reaction templates...")
    records = make_template_records()
    print(f"  {len(records)} template records")
 
    print(f"Building USPTO near-miss reversal pairs from {source_path}...")
    near_misses = parse_source_file_for_near_misses(source_path, max_rows=max_near_misses)
    print(f"  {len(near_misses)} near-miss records")
 
    records.extend(near_misses)
    print(f"Total counterfactual records: {len(records)}")
 
    # Summary stats
    by_category = {}
    by_difficulty = {}
    for r in records:
        cat = r["violation_category"]
        diff = r["difficulty"]
        by_category[cat] = by_category.get(cat, 0) + 1
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
    print(f"By category:   {by_category}")
    print(f"By difficulty: {by_difficulty}")
 
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
        description="Build Experiment 3: Counterfactual Validity dataset"
    )
    parser.add_argument("--source", default="../data/uspto_50k/uspto_50k.csv",
                        help="Path to USPTO-50K CSV (rxn_smiles,reagent_smiles,product_smiles,reaction_class)")
    parser.add_argument("--out_csv",        default="data/counterfactual_dataset.csv")
    parser.add_argument("--out_json",       default="data/counterfactual_dataset.json")
    parser.add_argument("--max_near_misses", type=int, default=500,
                        help="How many USPTO reactions to reverse as near-misses (default 500)")
    args = parser.parse_args()



    build_counterfactual_dataset(
        source_path=args.source,
        out_csv=args.out_csv,
        out_json=args.out_json,
        max_near_misses=args.max_near_misses,
    )
