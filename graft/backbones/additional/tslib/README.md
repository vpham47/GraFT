# Vendored from Time-Series-Library

Source: https://github.com/thuml/Time-Series-Library (commit 4e938a1767106324dd753b2a44832bf870a0252e), MIT License (see LICENSE).

The model and layer files are unchanged except:
- imports rewritten as package-relative (`from ..layers.X import ...`);
- `layers/SelfAttention_Family.py`: `reformer_pytorch` and `einops` are imported optionally. They are used only by the Reformer and Crossformer layers, which are not included here;
- `layers/AutoCorrelation.py`: an unused `matplotlib` import is removed;
- `models/iTransformer.py`, `models/TimeMixer.py`: `x_enc /= stdev` became `x_enc = x_enc / stdev`. In GraFT the inputs include learned embeddings, and the in-place division breaks autograd.

`graft/backbones/additional/tslib_predictor.py` wraps each model as a GraFT per-sensor predictor.
