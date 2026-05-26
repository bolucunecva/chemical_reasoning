This repository is code for **The Illusion of Chemical Reasoning in Large Language Models**

### Dataset
*ReactionPerturb* dataset evaluates chemical reasoning through three components:
- **Regioselectivity Stress Testing**
- **Chain-of-Chemistry (CoC)**
- **Counterfactual Validity Test**

#### 1. Regioselectivity Stress Testing

Evaluates robustness of regioselective reaction prediction under molecular perturbations.

| Type | Samples |
|---|---:|
| Isosteric | 4,868 |
| Bioisosteric | 722 |
| **Total** | **5,590** |

Examples:
- `-OH ↔ -SH`
- `-F ↔ -Cl`
- `-COOH ↔ -SO3H`

---

#### 2. Chain-of-Chemistry (CoC)

Evaluates multi-step chemical reasoning across USPTO-50K reaction classes.

| Reaction Type | Samples |
|---|---:|
| Heteroatom alkylation/arylation | 839 |
| Acylation | 697 |
| C–C bond formation | 514 |
| Heterocycle formation | 352 |
| Protection / Deprotection | 426 |
| Redox reactions | 115 |
| FGI / FGA | 57 |
| **Total** | **3,000** |

---

#### 3. Counterfactual Validity Test

Evaluates recognition of chemically invalid or impossible reactions.

Categories include:
- Aromaticity destruction
- Impossible valence
- Charge conservation
- Orbital symmetry violation
- Reagent mismatch
- Thermodynamic reversal

| Category Count | Samples |
|---|---:|
| Curated invalid reactions | 74 |
| Thermodynamic reversals | 500 |
| **Total** | **574** |

---

### Experiments

#### Experiment 1: Regioselectivity Stress Testing

```bash
python exp1.py \
    --model   $MODEL \
    --dataset dataset/regioselectivity_dataset.json \
    --out_dir results/exp1_output
```
#### Experiment 2: Chain-of-Chemistry
```bash
python exp2.py \
    --mode inference  \
    --model   $MODEL \
    --dataset_path dataset/coc_dataset.json \
    --out_dir results/exp2_output \

python exp2.py \
    --mode eval  \
    --model   $MODEL \
    --judge_model  $JUDGE_MODEL \
    --out_dir results/exp2_output \
```


#### Experiment 3: Counterfactual Validity Test
```bash
python exp3.py \
    --model   $MODEL \
    --dataset dataset/counterfactual_dataset.json \
    --out_dir results/exp3_output \
```
