# Efficient & Accurate Algorithms for Medical Image Segmentation: Findings

**Task:** binary polyp segmentation in colonoscopy frames.
**Training data:** Kvasir-SEG (1,000 expert masks). Split: 780 train / 100 validation / 120 test, with near-duplicate frames grouped so that no near-duplicate crosses a split.
**External test:** CVC-ClinicDB (612 frames from 29 colonoscopy videos, a different hospital and a different camera). It is never used for training or model selection.
**Metrics:** Dice, IoU, precision, recall, HD95 (px at 320×320), and Boundary-F1 (2 px tolerance), per image, with 95% bootstrap CIs. Paired Wilcoxon signed-rank tests compare methods.
**Efficiency:** parameters, GMACs at 256×256, and the median CPU latency of a batch-1 frame on one thread (idle Intel 4-core laptop CPU), in PyTorch and in ONNX Runtime fp32 / INT8.

---

## 1. Problem summary

Colonoscopy is the reference test for colorectal cancer screening. Meta-analyses of tandem colonoscopy put the adenoma miss rate at roughly one in four, often small or flat lesions at the edge of the field of view. A segmentation model that outlines polyps in real time can act as a second observer. The clinical setting imposes three requirements at once:

1. **Accuracy:** the outline must match an expert's (high Dice and IoU) and must have accurate *boundaries* (low HD95, high Boundary-F1), because size estimation and resection margins depend on the boundary.
2. **Efficiency:** endoscopy video runs at 25–30 fps, and many endoscopy towers or edge boxes have no GPU. A model that needs a datacentre GPU is not deployable in most of the world.
3. **Robustness and trust:** a model trained at one hospital must keep working on another hospital's camera (domain shift), and it should *know when it is unsure* so that a clinician can review those frames.

The research question is therefore not "which model has the highest Dice?". It is: **which algorithm gives the best accuracy per unit of compute, keeps that accuracy under domain shift, and reports calibrated uncertainty?**

## 2. Five candidate AI solutions (all implemented and benchmarked)

| # | Candidate | Family | Why it is a candidate | Params | Test Dice | External Dice |
|---|---|---|---|---|---|---|
| 1 | **Unsupervised clustering**: k-means and GMM on Lab colour + texture | classical / unsupervised | zero labels, trivially cheap | n/a | 0.272 / 0.284 | 0.273 / 0.272 |
| 2 | **Pixel classifiers**: Random Forest and LightGBM on 54 hand-crafted colour/texture/position features | classical supervised ML | interpretable, no GPU needed | n/a | 0.486 / 0.498 | 0.427 / 0.433 |
| 3 | **Light CNN U-Nets**: MobileNetV3-Small / MobileNetV3-Large / EfficientNet-B0 encoder, ImageNet-pretrained | efficient CNN | designed for mobile inference | 1.0 / 3.2 / 4.4 M | 0.859 / 0.878 / 0.890 | 0.810 / 0.838 / 0.845 |
| 4 | **Vision transformer**: SegFormer-B0 (MiT-b0 encoder + all-MLP decoder) | transformer | global context, strong boundaries | 3.7 M | 0.881 | 0.845 |
| 5 | **Heavy CNN reference**: U-Net with ResNet-34 encoder | standard CNN | the widely used accuracy baseline | 24.4 M | 0.880 | 0.842 |

A small U-Net trained from scratch (2.0 M parameters) was included as a control for the effect of pre-training. It reached only 0.688 test Dice and 0.542 external Dice, which shows that **ImageNet pre-training, not model size, drives most of the accuracy** on a 780-image training set.

## 3. Five hybrid methodologies (all implemented and tested)

| Hybrid | Combination | What it is meant to fix | Inference cost | Result (vs. plain MobileNetV3-L U-Net) |
|---|---|---|---|---|
| **A: Knowledge distillation** | ResNet-34 teacher → MobileNetV3-L student (BCE + MSE to the teacher's soft maps, α = 0.5) | transfer heavy-model accuracy to a light model | **zero** (training only) | test +0.001, external −0.002; **n.s.** (p = 0.79) |
| **B: Boundary-aware loss** | Dice + BCE + Kervadec boundary loss (signed-distance-weighted) | sharper boundaries, lower HD95 | **zero** (training only) | test +0.001, external −0.004; HD95 29.6 → 25.9 on test; **n.s.** (p = 0.51) |
| **C: CNN + ConvCRF** | CNN probabilities refined by a dense CRF (bilateral appearance + smoothness kernels, weights tuned on validation) | snap boundaries to image edges | a few ms | **−0.014 test, −0.014 external (p < 1e-11)**: hurts |
| **D: TTA + MC-dropout** | 3-flip test-time augmentation × T MC-dropout samples → mean prediction + per-pixel entropy | accuracy plus a *calibrated uncertainty* signal | ×3 (TTA) to ×24 (with MC) | TTA: **+0.007 test, +0.013 external (p < 1e-4)**; uncertainty ρ = −0.59 / −0.77 with Dice |
| **E: Ensemble + edge deployment** | probability averaging of the top-3 validation models; ONNX Runtime fp32 and INT8 static quantisation | maximum accuracy; then compress for the edge | ×9 (ensemble); ONNX fp32 −67% latency | ensemble (EffB0 + MNv3-L-boundary + ResNet-34): best overall (**0.889 test / 0.860 external**), +0.004 / +0.009 over Hybrid D (p = 0.006 / < 1e-7), but at 9× the cost |

**Negative results are reported as they came out.**
- Distillation and the boundary loss did not improve the MobileNetV3-L student beyond noise, and combining them (KD + boundary) was slightly worse. With a pretrained encoder and only 780 images, the student already matches the teacher (see section 4), so there is little left to distil.
- ConvCRF consistently *hurt*: the CNN boundaries are already better than the colour edges the CRF snaps to. Polyp borders are often low-contrast, and specular highlights produce false edges.

## 4. Best methodology and justification

**Chosen: Hybrid D on candidate 3.** This is a **MobileNetV3-Large U-Net (3.2 M params, 0.42 GMACs) + flip TTA + MC-dropout uncertainty**, deployed through ONNX Runtime fp32.

| | MobileNetV3-L + TTA (chosen) | ResNet-34 U-Net + TTA | Top-3 ensemble + TTA | SegFormer-B0 + TTA |
|---|---|---|---|---|
| Test Dice [95% CI] | **0.885** [0.855–0.911] | 0.886 [0.857–0.912] | 0.889 [0.858–0.916] | 0.888 [0.859–0.914] |
| External Dice (ClinicDB) | **0.851** | 0.855 | 0.860 | 0.853 |
| Test Boundary-F1 / HD95 | 0.707 / 26.2 | 0.711 / 28.5 | 0.727 / 24.4 | 0.674 / 24.6 |
| Parameters | **3.2 M** | 24.4 M (7.7×) | 32.0 M | 3.7 M |
| GMACs @256 | **0.42** | 7.84 (19×) | 9.37 | 1.76 |
| CPU per frame, 1 thread, ×3 TTA | **~156 ms PyTorch / ~52 ms ONNX** | ~1,005 ms / ~555 ms ONNX | ~1,400 ms | ~387 ms / ~297 ms ONNX |

Justification:
1. **Accuracy level with the heavy reference.** Against ResNet-34 + TTA, the paired difference is −0.0008 Dice on the test set (95% CI −0.016 to +0.015, Wilcoxon p = 0.85, 61 wins / 59 losses). Externally it is −0.005 (95% CI −0.014 to +0.004, p = 0.054). That is no statistically detectable loss at **1/19 of the compute and 1/7.7 of the parameters**. Against SegFormer-B0 + TTA the external gap is only −0.002 Dice. It is consistent enough to reach p < 1e-5 over 612 frames, but clinically negligible, and SegFormer needs 4× the MACs.
2. **Best accuracy per unit of compute.** It lies on or next to the Pareto front of Dice against CPU time on both test sets (figure 06). EfficientNet-B0 without TTA is an equally valid point: 0.890 test / 0.845 external at 82 ms. Nothing more accurate costs less.
3. **Domain robustness.** The test-to-external drop is 0.034 Dice, similar to the 24 M-parameter ResNet-34 (0.031) and SegFormer (0.035). It is far smaller than the from-scratch U-Net (0.146) or the classical models.
4. **It knows when it is unsure.** MC-dropout entropy correlates strongly with error (Spearman ρ = −0.59 test, **−0.77 external**, p < 1e-12). Flagging the 20% least-certain frames for review raises mean Dice on the remaining 80% to **0.905** on the external hospital (0.901 internally). This is the clinically meaningful operating mode: automatic on confident frames, human review on uncertain ones.
5. **Deployable on CPU.** Exported to ONNX Runtime fp32, the model runs in **17 ms on one CPU thread (about 58 fps)** with Dice identical to PyTorch, and in about 52 ms with 3-flip TTA. That is real-time on a laptop without a GPU.

If maximum accuracy matters more than cost, the top-3 ensemble is the best model measured (0.889 / 0.860, significantly better than the chosen model on both sets), at 9× the compute.

A deployment caveat found in Hybrid E: **static INT8 quantisation is not safe for the MobileNet family.** MobileNetV3-L dropped from 0.878 to 0.827 test Dice, and MobileNetV3-S collapsed to 0.315. This fits the known sensitivity of depthwise convolutions and hard-swish activations to per-tensor activation ranges. INT8 *is* usable for ResNet-34 (−0.015 Dice, −38% latency, 4× smaller) and SegFormer (−0.006 test / −0.019 external Dice, −16% latency). For the chosen model, ONNX fp32 is the right deployment format, and at 12 MB it is already small.

**Resolution study (figure 21).** For MobileNetV3-L, the input size trades accuracy for time:
- 160 px: 0.828 test / 0.753 external Dice in 25 ms
- 224 px: 0.869 / 0.828 in 38 ms
- 256 px: 0.878 / 0.838 in 50 ms
- 320 px: 0.879 / 0.834 in 75 ms

So 256 px is the sweet spot, and going larger costs 50% more time for no gain. On a weaker CPU, 224 px keeps 99% of the Dice at 77% of the cost.

## 5. Mini abstract

> **Efficient and uncertainty-aware polyp segmentation for CPU-only colonoscopy.** We benchmarked 5 candidate families and 5 hybrid methodologies for polyp segmentation: unsupervised clustering, pixel-wise gradient boosting, mobile CNN U-Nets, a SegFormer transformer and a ResNet-34 U-Net, plus knowledge distillation, boundary loss, ConvCRF, TTA/MC-dropout and ensembling with INT8 deployment. Models were trained on Kvasir-SEG (780 images) and tested on a held-out Kvasir split (n = 120) and on CVC-ClinicDB from another centre (n = 612). A 3.2 M-parameter MobileNetV3-Large U-Net with flip test-time augmentation reached Dice 0.885 (95% CI 0.855–0.911) internally and 0.851 externally. It was statistically indistinguishable from a 24 M-parameter ResNet-34 U-Net (paired Wilcoxon p = 0.85 internal, 0.054 external) at 1/19 of the multiply-accumulates, and ran in 17 ms per frame on a single CPU thread through ONNX Runtime. A 9×-costlier three-model ensemble was the most accurate (0.889 / 0.860). Distillation and boundary loss gave no significant gain, and ConvCRF significantly reduced Dice. Monte-Carlo-dropout entropy was strongly rank-correlated with segmentation error (ρ = −0.77 externally): deferring the 20% least-certain frames lifted external Dice to 0.905. Static INT8 quantisation degraded the MobileNet models but not ResNet-34. Pretrained mobile encoders combined with cheap test-time ensembling and uncertainty deferral are therefore a strong, deployable alternative to heavy segmentation networks.

## 6. Implementation outline (reproducible)

```
scripts/prepare_data.py   Kvasir-SEG + CVC-ClinicDB → 320 px cache; near-duplicate-grouped split (780/100/120)
src/medseg/models.py      U-Net (timm encoders: mobilenetv3_small/large, efficientnet_b0, resnet34), SegFormer-B0, tiny U-Net
src/medseg/train.py       224 px random crops + flips/rotations/colour jitter; Dice+BCE (+KD, +boundary); AdamW, warm-up + cosine,
                     EMA 0.995, 30 epochs, checkpoint chosen on validation Dice (trained on a Colab L4 GPU, ~35 min for all 10)
scripts/run_classical.py  k-means, GMM, Random Forest, LightGBM on 54 per-pixel features
scripts/evaluate.py  CPU inference at 256 px, metrics at 320 px; TTA; ConvCRF tuning on validation; bootstrap CIs; efficiency
scripts/analysis.py  ensemble, MC-dropout uncertainty and risk–coverage, ONNX fp32/INT8, resolution study, paired statistics
scripts/make_figures.py  23 figures + 3 animations
app/app.py           Streamlit "PolypSeg Studio": upload a frame → mask, probability, uncertainty, Dice vs. expert
```

Selection discipline: every choice (checkpoint epoch, ensemble members, CRF weights, threshold 0.5) was made on the **validation** split only. The Kvasir test split and CVC-ClinicDB were scored once.

**Limitations.**
- Single-frame models with no temporal smoothing across video frames.
- One training dataset of 1,000 images.
- ClinicDB frames come from 29 videos, so they are correlated: the external CIs are optimistic, and the effective sample size is nearer to the number of videos.
- Latency was measured on one laptop CPU.
- This is a research prototype, **not a medical device**.
