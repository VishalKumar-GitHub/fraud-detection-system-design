# fraud-detection-system-design

An end-to-end, runnable ML system design for detecting counterfeit products on
an e-commerce platform — I built a reference code, structured system design.

Covers the full lifecycle: business framing → North Star / guardrail metrics →
architecture → feature engineering → model ladder → evaluation →
**guardrail-driven threshold selection** → risk tiering.

<img width="860" height="758" alt="architecture_animated" src="https://github.com/user-attachments/assets/f147d9f9-5608-4e12-ac51-171e306e2a72" />

## Why this exists

Most "ML system design" writeups stop at a whiteboard sketch. This repo is the
sketch turned into code that actually runs, so you can:

- Run each stage live and explain it, instead of talking through slides
- Show real precision/recall/AUC numbers instead of hand-waving metrics
- Demonstrate the part most candidates skip: **tying the classification
  threshold back to a business guardrail**, not just optimizing F1

## Problem statement

Sellers on the platform list counterfeit ("fake") versions of genuine branded
products. The goal is a model that scores every product with a counterfeit
risk level, so the platform can act on it — automatically, via manual review,
or not at all.

| | |
|---|---|
| **North Star metric** | Reduction in counterfeit goods on the platform |
| **Guardrail 1** | Genuine-product upload rate stays constant |
| **Guardrail 2** | Seller churn rate stays constant |
| **Scale** | ~60M products |
| **Scoring mode** | Batch, 1–2x per day (not real-time — risk is a product property, not tied to a live session) |

### Fraud taxonomy

The platform faces four fraud categories. This project designs a system for
the **seller-side / counterfeit goods** branch specifically:

| Category | Examples |
|---|---|
| Buyer-side | Return fraud, refund fraud |
| **Seller-side (this repo)** | **Counterfeit goods**, bad-quality products, review manipulation, fake listings |
| Attack-related | Account takeover, bot attacks |
| Logistics-side | Delivery agent fraud, package theft |

## Architecture

Every product is scored by the model in batch. Risk tier determines what
happens next, and outcomes flow back into the next training run.

```
Product catalog (~60M products)
        |
        v
Counterfeit risk model (batch, 1-2x/day)
        |
   risk score
        |
  -------------------------------
  |             |               |
  v             v               v
Auto-remove  Manual review   Keep live
(very high)  (medium-high)   (low risk)
  |             |               |
  |             v               v
  |       Review outcome   Guardrail monitoring
  |       (confirmed Y/N)  (upload rate, churn)
  |             |
  -------------------------------
        |
        v
  Training labels (1 = counterfeit, 0 = genuine)
        |
        +----> feeds back into model retraining
```

See [`docs/images/architecture.png`](https://github.com/VishalKumar-GitHub/fraud-detection-system-design/blob/main/architecture.png) for the
full diagram with the feedback loop.

## Pipeline — 7 steps

Each stage is a standalone script. Run them in order — each one reads the
previous step's output from `data/` and writes its own. This is deliberate:
in an interview, you run and narrate **one concept at a time** instead of
dumping a single monolithic script.

| Step | Script | What it demonstrates |
|---|---|---|
| 1 | [`step1_simulate_data.py`](src/step1_simulate_data.py) | Data sources: catalog, seller, returns, image-similarity service |
| 2 | [`step2_feature_engineering.py`](src/step2_feature_engineering.py) | Translating entities (product/seller/category/context) into features |
| 3 | [`step3_train_test_split.py`](src/step3_train_test_split.py) | Stratified split, class imbalance awareness |
| 4 | [`step4_train_models.py`](src/step4_train_models.py) | Model ladder: logistic regression → random forest → gradient boosting |
| 5 | [`step5_evaluate_models.py`](src/step5_evaluate_models.py) | Correct metrics for imbalanced fraud problems (not accuracy) |
| 6 | [`step6_threshold_selection.py`](src/step6_threshold_selection.py) | **The senior-level differentiator** — threshold chosen against a precision guardrail, not just best F1 |
| 7 | [`step7_risk_tiering_and_test_eval.py`](src/step7_risk_tiering_and_test_eval.py) | Mapping scores to auto-remove / manual-review / keep tiers + final unbiased test evaluation |

### What's *discussed*, not coded

These require live infrastructure, so they're talking points rather than
runnable code:

- **Online A/B experimentation** — control (existing approach) vs. treatment
  (new model), null/alternative hypothesis, significance level, p-value,
  Type I/II errors, before scaling to 100% of traffic
- **Production monitoring** — feature drift (e.g. PSI), label drift,
  scheduled retraining, alerting if precision on audited "keep" samples
  degrades over time
- **ID embeddings** — in a neural net variant, `seller_id` / `product_id`
  would go through an `nn.Embedding` layer instead of being dropped, letting
  the model learn latent seller/product risk representations

## Entities and features

| Entity | Features |
|---|---|
| **Product** | Price, discount %, return/refund rate, counterfeit return ratio, image quality, image/text/brand similarity to known counterfeits, unauthorized celebrity-image flag, review sentiment signal |
| **Seller** | Reputation, rating, tenure, number of listings, return rate, violation history |
| **Category** | Demand level, average value, counterfeit susceptibility, ease of replication, category return/incident rate |
| **Context** | Seasonal trend flag (counterfeit activity rises during sales periods) |


### Run step by step

```bash
python src/step1_simulate_data.py
python src/step2_feature_engineering.py
python src/step3_train_test_split.py
python src/step4_train_models.py
python src/step5_evaluate_models.py
python src/step6_threshold_selection.py
python src/step7_risk_tiering_and_test_eval.py
```

Each script prints its own results and saves intermediate output to `data/`.

## Sample output

```
Validation set results (threshold=0.5):
              model  precision  recall     f1  roc_auc
logistic_regression     0.4713  0.8367 0.6029   0.9246
  gradient_boosting     0.7316  0.5624 0.6359   0.9187
      random_forest     0.5165  0.7823 0.6222   0.9173

Chosen threshold: 0.9191
  -> precision: 0.8504 (meets >= 0.85 guardrail)
  -> recall:    0.4512 (this is the % of counterfeits we'll catch)

Risk tier distribution on test set:
keep             2229
manual_review     430
auto_remove       341
```

> Numbers come from a synthetic dataset (`step1_simulate_data.py`) generated
> with a fixed random seed, built to mimic realistic, correlated fraud
> signal — not real platform data.

## Repo structure

```
counterfeit-fraud-detection/
├── README.md
├── requirements.txt
├── run_pipeline.py          # runs all 7 steps in sequence
├── src/
│   ├── step1_simulate_data.py
│   ├── step2_feature_engineering.py
│   ├── step3_train_test_split.py
│   ├── step4_train_models.py
│   ├── step5_evaluate_models.py
│   ├── step6_threshold_selection.py
│   └── step7_risk_tiering_and_test_eval.py
├── data/                     # generated at runtime (gitignored)
└── docs/
    └── images/
        └── architecture.png
```

## Notes

- Models use `scikit-learn`'s `GradientBoostingClassifier` rather than
  XGBoost/LightGBM for zero-dependency portability. In production, swap in
  XGBoost for speed and `scale_pos_weight` support — the API is a near
  drop-in replacement.
- Labels are generated via **weak supervision** (manual review outcomes +
  sampled genuine products), mirroring how labels are actually sourced for
  this kind of problem — and a real deployment should budget for periodic
  re-audits to catch label noise.

## License

MIT — use freely for interview prep, learning, or as a starting point for
your own system design writeups.

## About Me

**Vishal Kumar**
- [GitHub](https://github.com/VishalKumar-GitHub)

📫 **Follow me** on [Xing](https://www.xing.com/profile/Vishal_Kumar055381/web_profiles?expandNeffi=true) | [LinkedIn](https://www.linkedin.com/in/vishal-kumar-819585275/)
