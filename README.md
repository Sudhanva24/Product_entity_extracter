# Product Entity Extraction

A multimodal fine-tuning pipeline to extract entity values (weight, dimensions, voltage, etc.) directly from product images using Qwen2-VL.

---

## Problem Statement

Given a product image and an entity name (e.g., `item_weight`, `height`, `voltage`), extract the exact value of that entity as it appears on the product. The output must be a numeric value followed by a valid unit (e.g., `500.0 gram`, `27.5 centimetre`).

---

## Dataset

**Source:** Amazon ML Challenge 2024 kaggle

| Split | Samples |
|-------|---------|
| Train | ~9,996  |
| Val   | ~1,996  |
| Test  | ~1,996  |

**Total original dataset size:** ~263,858 samples

Each sample contains:
- `image_link` — URL to the product image on Amazon CDN
- `group_id` — product category group identifier
- `entity_name` — the attribute to extract (e.g., `item_weight`, `height`, `wattage`)
- `entity_value` — ground truth value with unit (e.g., `500.0 gram`)

**Entity types in dataset:** depth, height, item\_volume, item\_weight, maximum\_weight\_recommendation, voltage, wattage, width

**Splitting strategy:** Stratified split by `group_id` to ensure all product categories are represented across all splits. Groups with fewer than 4 samples were moved directly to the training set to avoid data leakage while maintaining category diversity.

---

## Evaluation Metric

Submissions are evaluated using **F1 Score** based on exact string match between prediction and ground truth.

Given prediction `OUT` and ground truth `GT`:

| Condition | Classification |
|-----------|---------------|
| `OUT != ""` and `GT != ""` and `OUT == GT` | True Positive |
| `OUT != ""` and `GT != ""` and `OUT != GT` | False Positive |
| `OUT != ""` and `GT == ""` | False Positive |
| `OUT == ""` and `GT != ""` | False Negative |
| `OUT == ""` and `GT == ""` | True Negative |

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

---

## Model

**Base Model:** `Qwen2-VL-2B-Instruct`

Qwen2-VL was chosen for this task because:
- Pre-trained on VQA and OCR datasets, making it naturally suited for reading product labels
- Strong Vision-Language alignment for grounding visual features to text output
- Dynamic resolution input — handles varying product image sizes natively
- Open-source with strong community support and Unsloth compatibility

---

## Prompt Design

```
What is the {entity_name}?
Reply with only the value and unit from: {allowed_units}.
If not visible, reply with empty string.
```

The prompt is intentionally minimal. Including the allowed units list directly in the prompt enforces valid output format and reduces postprocessing failures. Instructing the model to return an empty string when the entity is not visible prevents hallucinated values which would count as False Positives under the F1 metric.

---

## Fine-Tuning Method

**Method:** QLoRA (Quantized Low-Rank Adaptation) via Unsloth

**Framework:** Unsloth + TRL SFTTrainer

| Hyperparameter | Value |
|----------------|-------|
| Quantization | 4-bit (QLoRA) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Batch Size | 2 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.05 |
| Epochs | 3 |
| Max Sequence Length | 256 |
| Optimizer | AdamW 8-bit |
| Vision Layers Fine-tuned | Yes |
| Language Layers Fine-tuned | Yes |

QLoRA allows fine-tuning a 2B parameter model by quantizing weights to 4-bit and applying trainable low-rank adapter matrices on top. Only ~1.2% of total parameters are actually trained, making it feasible on a single GPU.

---

## Compute

**Platform:** Lightning AI  
**GPU:** NVIDIA L4 (24GB VRAM)  
**Training Time:** ~5 hours for 3 epochs on 10k samples  
**Inference Speed:** ~1.94s per sample

---

## Training Curves

*Loss curve from Weights & Biases:*
<img width="910" height="478" alt="Screenshot 2026-02-25 at 11 22 29 AM" src="https://github.com/user-attachments/assets/2df6e41a-d220-4c76-9f64-9bb67b765f43" />

## Results

### Test Set (1,996 samples)

| Metric | Score |
|--------|-------|
| **F1 Score** | **0.7387** |
| Precision | 0.6466 |
| Recall | 0.8615 |
| True Positives | 1169 |
| False Positives | 639 |
| False Negatives | 188 |

### Per-Entity F1 Breakdown

| Entity | F1 Score |
|--------|----------|
| wattage | 0.8846 |
| height | 0.7950 |
| item_weight | 0.7552 |
| voltage | 0.7500 |
| item_volume | 0.7333 |
| depth | 0.7201 |
| width | 0.6421 |
| maximum_weight_recommendation | 0.3571 |

---

## Post Processing

Model outputs are cleaned using regex before F1 evaluation:
- Extract `<number> <unit>` pattern from raw model output
- Fix common unit typos (`metre` ↔ `meter`, `foot` ↔ `feet`)
- Return empty string for any output not matching the allowed units list
- Replace `NA` / `N/A` / `None` outputs with empty string

This prevents hallucinated units from becoming unnecessary False Positives.

---

## Key Observations

- High recall (0.86) but lower precision (0.65) suggests the model is good at finding entities but sometimes predicts wrong values — a common pattern when training on noisy labels
- `maximum_weight_recommendation` has the lowest F1 (0.36), likely because this entity is rarely visible directly on product images and requires inference from context
- `wattage` has the highest F1 (0.88) as wattage is typically printed prominently on electrical product images
