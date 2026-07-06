# Committed provisioned certifier artifacts

These are the **gitignored derived artifacts** the certifier and the artifact
build consume, committed here (compressed + split) so a clean build needs **no
provisioning step and no TensorFlow/tfds venv** — it just extracts these and
builds the Docker image.

Each bundle is `certifier-<what>.tar.gz`, split into `<=45 MB` chunks named
`.part-aa`, `.part-ab`, … so every file stays under GitHub's 100 MB limit.
Reassemble with `cat certifier-<what>.tar.gz.part-* > certifier-<what>.tar.gz`.

| bundle | expands to (relative to `python-certifier/`) |
|---|---|
| `certifier-nets.tar.gz` | `models/neural_net_*.txt` — the 9 sound-by-construction weight files |
| `certifier-inputs-higgs-sweep-10k.tar.gz` | `tests/inputs_higgs_w{128,256,512,1024}_n10000/` |
| `certifier-inputs-higgs-1024-500k.tar.gz` | `tests/inputs_higgs_w1024_n500000/` (the RQ3 500k row) |
| `certifier-inputs-emnist-balanced.tar.gz` | `tests/inputs_emnist_balanced_full/` |
| `certifier-inputs-emnist-byclass.tar.gz` | `tests/inputs_emnist_byclass_full/` |

## Extract (what the build does)

```bash
./extract.sh        # reassembles the parts and extracts into ../{models,tests}
```

`docker/build_artifact.sh` runs this automatically in its preflight, so a build
from clean checkouts requires no provisioning.

## Source of truth / regenerating

The bundles are a committed **cache** of the outputs of
`../models/provision_image_model_weights.sh` and
`../models/provision_higgs_emnist_inputs.sh`, which build each model from its
committed CSV weights and generate the certification inputs. Those scripts remain
the regenerator of record. To refresh the bundles after re-provisioning:

```bash
# from python-certifier/, having provisioned models/*.txt and tests/inputs_*/:
./provisioned/repack.sh
```
