# Certifier model weight files (`.txt`)

These `.txt` files are the weight inputs to the certifier, one per model. They are
**sound by construction**: each is generated directly from the deployed Keras
model and encodes *exactly* the float32 weights the model executes with.

## Why this directory exists

The original CAV 2025 (Tobler et al.) pipeline produced its certifier `.txt`
files with a script that rounded weights to **5 decimal places**
(`verified-certified-robustness/scripts/make_certifier_format.py`, in its
Jan-2025 form). But the models that actually run — the ones that produce the
`y1` logits and against which the counter-examples were generated — use the
full-precision float32 CSV weights. So the certifier was computing norms and
Lipschitz constants for a *5-dp-rounded model* while the deployed model was the
full-precision one (~100% of weights differ, by up to ~5e-6). That makes the
certified Lipschitz bounds apply to a different model than the one executing —
a soundness gap independent of floating-point execution semantics.

Rather than patch the upstream repo, we regenerate the certifier input **here**,
from the exact deployed weights, so the gap is not inherited.

## How they are produced (and why it's sound)

`../make_certifier_format_from_model.py`:
1. builds the actual model via `doitlib` under a float32 mixed-precision policy
   and `load_and_set_weights` (exactly how it is run / attacked), then
2. writes `model.get_weights()` — the precise float32 values the forward pass
   multiplies with — as exact decimals (`:.150f`, which is exact for float32),
   in the certifier's nested-bracket `(in,out)` format, and
3. **self-checks**: it reloads the file through the certifier's own
   `load_network_from_file` and asserts every parsed rational equals the exact
   rational of the corresponding `get_weights()` float32 value, raising an error
   on any mismatch. A non-sound file therefore cannot be produced silently.

Requires the TF 2.13.0 environment pinned in `../requirements.txt` (matched to
the cav2025 environment so the Keras execution reproduces the original `y1`
logits bit-for-bit).

## Not committed — regenerate on demand

The `.txt` files are large derived artifacts (~360 MB total at exact-float32
precision) and would duplicate the cav2025 CSV weights, so they are
**gitignored**. The committed source of truth is the generator
(`../make_certifier_format_from_model.py`) plus `provision_image_model_weights.sh` and this README.
Regenerate them before running the test suite or building the artifact:

```bash
bash models/provision_image_model_weights.sh
```

Run it in the TF 2.13.0 venv (see `../requirements.txt`). All paths are derived
from the script location; the cav2025 models are expected in the sibling
`../verified-certified-robustness/` repo (override with
`CAV_MODELS=/path/to/cav2025-models`).
