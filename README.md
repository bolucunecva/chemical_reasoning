## Chemical Reasoning

### Dataset

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
