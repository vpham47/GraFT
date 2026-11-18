"""Per-sensor predictors (backbones) for GraFT.

main/        the three backbones evaluated in the paper: MLP, PatchTST, TSMixer
additional/  further forecasters anyone can plug in (not evaluated in the paper, untuned defaults)

Every predictor maps [B*N, T, C] -> [B*N, horizon, d]. The registry is in graft/registry.py.
"""
