# 2026S1 COMP8221 Assignment 1 - Notebook Outline

## 1. Title
Cold Diffusion for Flower Image Generation:
Reversing Deterministic Blur from Scratch with a Time-Conditioned U-Net

## 2. Objective
- Implement a non-standard diffusion variant from scratch
- Use deterministic blur degradation instead of standard DDPM noise
- Train and evaluate on Oxford-102 Flowers
- Report quantitative and qualitative results

## 3. Dataset
- Oxford-102 Flowers
- Official split from setid.mat
- Image size: 64x64
- Normalization to [-1, 1]

## 4. Method
- Cold Diffusion blur degradation
- Time-conditioned U-Net
- L1 training objective
- EMA checkpoint

## 5. Training Setup
- epochs = 25
- batch_size = 8
- lr = 2e-4
- base_channels = 32
- time_dim = 128
- num_steps = 100
- sigma range = [0.01, 4.0]
- kernel_size = 19
- schedule = linear

## 6. Qualitative Results
- blur progression
- reconstruction examples
- reverse trajectory

## 7. Quantitative Results
- FID main = 85.049
- 25-step = 83.005
- 50-step = 84.282
- 100-step = 85.049

## 8. Discussion
- 25-step performed best
- restoration-based evaluation
- shorter reverse path gave better trade-off

## 9. Reproducibility
- commands to train
- commands to export fake/real
- commands to compute FID

## 10. Conclusion
- Cold Diffusion works as a valid non-standard diffusion variant
- qualitative restoration is clear
- quantitative evaluation and ablation completed
