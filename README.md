# MedSeg-AI: Efficient & Accurate Algorithms for Medical Image Segmentation

**Project 10 · Husain Naqvi · 2026** · polyp segmentation in colonoscopy · [findings write-up](docs/FINDINGS.md) · [figure gallery](https://engr-husainnaqvi.github.io/MedSeg-AI/)

The study asks which segmentation algorithm gives **the most accuracy per millisecond of CPU**, keeps it **on another hospital's data**, and **knows when it is wrong**. It benchmarks 5 candidate families (unsupervised clustering, gradient-boosted pixel classifiers, mobile CNN U-Nets, a SegFormer transformer and a ResNet-34 U-Net) and 5 hybrid methodologies (knowledge distillation, boundary loss, ConvCRF, TTA + MC-dropout uncertainty, and ensembling + INT8 deployment). Models are trained on **Kvasir-SEG** and tested on a held-out Kvasir split and on **CVC-ClinicDB**, an external centre whose 612 frames are never seen in training or model selection.

<p align="center"><img src="docs/figures/clinicdb_video_segmentation.gif" width="92%" alt="Chosen model segmenting consecutive CVC-ClinicDB colonoscopy frames"></p>

## Headline result

A **3.2 M-parameter MobileNetV3-Large U-Net with flip test-time augmentation** matches a 24 M-parameter ResNet-34 U-Net at **1/19 of the compute**. Exported to ONNX Runtime, it segments a frame in **17 ms on a single CPU thread**.

| Model | Params | GMACs | CPU ms / pass (PyTorch · ONNX), 1 thread | Kvasir test Dice [95% CI] | ClinicDB (external) Dice | Test HD95 (px) |
|---|---|---|---|---|---|---|
| **MobileNetV3-L U-Net + TTA (chosen)** | **3.2 M** | **0.42** | **52 · 17** | **0.885** [0.855–0.911] | **0.851** | 26.2 |
| ResNet-34 U-Net + TTA (reference) | 24.4 M | 7.84 | 335 · 185 | 0.886 [0.857–0.912] | 0.855 | 28.5 |
| SegFormer-B0 + TTA | 3.7 M | 1.76 | 129 · 99 | 0.888 [0.859–0.914] | 0.853 | 24.6 |
| EfficientNet-B0 U-Net + TTA | 4.4 M | 1.11 | 82 · n/a | 0.885 [0.853–0.912] | 0.849 | 26.6 |
| MobileNetV3-S U-Net + TTA | 1.0 M | 0.14 | 29 · 8 | 0.866 [0.833–0.895] | 0.815 | 29.0 |
| Top-3 ensemble + TTA (Hybrid E) | 32.0 M | 9.37 | 467 · n/a | **0.889** [0.858–0.916] | **0.860** | 24.4 |
| U-Net from scratch (no pre-training) | 2.0 M | 3.53 | 195 | 0.688 | 0.542 | 71.5 |
| LightGBM pixel classifier | n/a | n/a | 558 | 0.498 | 0.433 | 117.4 |
| k-means (unsupervised) | n/a | n/a | 72 | 0.272 | 0.273 | 137.0 |

TTA costs 3 passes per frame; the ensemble costs 9.

- **Statistically level with the heavy model.** Against ResNet-34 + TTA, the paired difference is −0.0008 Dice on the test set (95% CI −0.016 to +0.015, Wilcoxon p = 0.85) and −0.005 externally (p = 0.054).
- **The ensemble is the most accurate, at a price.** It beats the chosen model by +0.004 on the test set (p = 0.006) and +0.009 externally (p < 1e-7), but costs 9× the compute.
- **It knows when it is unsure.** MC-dropout entropy rank-correlates with error (Spearman ρ = −0.77 on the external centre). Deferring the 20% least-certain frames to a clinician lifts external Dice from 0.851 to **0.905**.
- **Honest negatives.**
  - Distillation and the boundary loss gave no significant gain.
  - ConvCRF *lowered* Dice by 1.4 points (p < 1e-11).
  - Static INT8 quantisation broke the MobileNets (MNv3-L 0.878 → 0.827) but was safe for ResNet-34 and SegFormer.

Latency is the median of batch-1 runs at 256×256 on an idle laptop CPU (Intel, 4 cores). Every choice (checkpoint, ensemble members, CRF weights, threshold) was made on the validation split. The test sets were scored once.

## Visual tour

Twenty-three figures plus three animations.

| | |
|---|---|
| ![](docs/figures/01_dataset_examples.png) **The data:** Kvasir-SEG and CVC-ClinicDB frames with expert masks | ![](docs/figures/02_dataset_statistics.png) **What makes it hard:** polyp size, position and colour |
| ![](docs/figures/03_augmentation.png) **Augmentation** used in training | ![](docs/figures/04_study_design.png) **Study design:** 5 candidates × 5 hybrids, internal and external tests |
| ![](docs/figures/05_leaderboard.png) **Leaderboard** with bootstrap CIs | ![](docs/figures/06_pareto_accuracy_latency.png) **Pareto front:** Dice vs. CPU latency |
| ![](docs/figures/07_model_cost.png) **Cost:** params, GMACs, latency | ![](docs/figures/08_training_curves.png) **Training curves** (validation Dice) |
| ![](docs/figures/09_metric_heatmap.png) **All metrics** for all methods | ![](docs/figures/10_dice_distributions.png) **Per-image Dice distributions** |
| ![](docs/figures/11_domain_shift.png) **Domain shift:** Kvasir → ClinicDB | ![](docs/figures/12_qualitative_comparison.png) **Side by side:** every family on the same frames |
| ![](docs/figures/13_dice_by_polyp_size.png) **Small polyps are hardest** | ![](docs/figures/14_failure_cases.png) **Failure cases**, examined honestly |
| ![](docs/figures/15_hybrid_ablation.png) **Hybrid ablation** on the MobileNetV3-L U-Net | ![](docs/figures/16_knowledge_distillation.png) **Hybrid A:** teacher → student distillation |
| ![](docs/figures/17_convcrf_refinement.png) **Hybrid C:** ConvCRF refinement | ![](docs/figures/18_uncertainty.png) **Hybrid D:** MC-dropout uncertainty and risk–coverage |
| ![](docs/figures/19_ensemble.png) **Hybrid E:** ensemble | ![](docs/figures/20_deployment_int8.png) **Deployment:** ONNX fp32 vs. INT8 |
| ![](docs/figures/21_resolution_scaling.png) **Resolution** as an efficiency knob | ![](docs/figures/22_paired_statistics.png) **Paired statistics** (Wilcoxon + bootstrap) |
| ![](docs/figures/23_classical_methods.png) **Classical ML baselines** and their features | ![](docs/figures/methods_cycle.gif) **Animation:** every method on the same frames |

## Method

- **Data:** Kvasir-SEG (1,000 images) is split 780 / 100 / 120, with near-duplicate frames grouped so that none cross splits. CVC-ClinicDB (612 frames, 29 videos) is used only as an external test. Images are cached at 320 px. Training uses random 224 px crops, inference uses 256 px, and metrics are computed at 320 px.
- **Training:** Dice + BCE loss; AdamW with warm-up + cosine schedule; EMA 0.995; 30 epochs; the checkpoint with the best validation Dice is kept. All 10 deep models were trained on a Colab L4 GPU (about 35 minutes in total). Inference and every efficiency measurement ran on CPU.
- **Hybrids:**
  - A: teacher soft-map distillation (ResNet-34 → MobileNetV3)
  - B: Kervadec boundary loss
  - C: dense ConvCRF tuned on validation
  - D: 3-flip TTA × MC-dropout, with per-pixel entropy and image-level deferral
  - E: probability-averaging ensemble of the top-3 validation models, plus ONNX Runtime fp32 / static INT8 (QDQ, per-channel)
- **Statistics:** per-image metrics, 2,000-sample bootstrap CIs, paired Wilcoxon signed-rank tests, and Spearman ρ for uncertainty vs. Dice.

The full problem statement, candidate and hybrid rationale, chosen methodology and abstract are in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

## Interactive app: PolypSeg Studio

The app (in the private repository) lets you pick a model, upload a colonoscopy frame (or use the examples), and see:
- the predicted mask
- the probability and uncertainty maps
- the inference time
- Dice against the expert mask, when one is available

The app also has TTA, MC-dropout and ConvCRF toggles, a threshold slider, and mask download.

## Code and models

This public repository contains the results, figures and write-up. The full source code (data pipeline, training, evaluation, analysis and figure scripts), the trained checkpoints and the Streamlit app are in a private repository. Access is available on request to reviewers and prospective supervisors via GitHub ([Engr-HusainNaqvi](https://github.com/Engr-HusainNaqvi)).

## Limitations

- These are single-frame models with no temporal smoothing across video frames.
- There is one training dataset of 1,000 images.
- ClinicDB frames come from 29 videos, so they are correlated and the external CIs are optimistic.
- Latency was measured on one laptop CPU.

**This is a research prototype, not a medical device.**

## Data and licence

Kvasir-SEG (Jha et al., MMM 2020) and CVC-ClinicDB (Bernal et al., CMIG 2015) are used under their research-use terms. They are not redistributed, except the nine example frames in `app/samples`.

Code, models, figures and text: **Copyright © 2026 Husain Naqvi. All Rights Reserved.** See [LICENSE](LICENSE).
