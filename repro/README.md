# burn-cubek-drift-repro

Minimal reproducer for a numerical drift in `cubek`-backed `burn-wgpu`
introduced by cubek PR #191 ([commit 662610b7][pr191], "Refactor
matmul types for a Tile enum"). No weights, no images — the inputs,
conv weights, biases, and head weights are all synthesized from a
tiny LCG seed so the run is bit-for-bit deterministic across machines.

This was originally surfaced by Brush's LPIPS VGG test: PyTorch
reference `0.6571019887924194` drifts to `~0.52` on wgpu after #191.
With cubek rolled back to `be1fef47` (parent of #191) the Brush test
matches PyTorch to ~1e-4.

## What it does

1. Builds two 1×3×128×128 NCHW tensors from an LCG seed and applies
   LPIPS's published mean/stdev normalization.
2. Runs the same LPIPS-shaped pipeline on two backends:
   - `burn::backend::Wgpu`  (under test)
   - `burn::backend::NdArray` (CPU reference)

   Pipeline per block (5 blocks total):
   - 2–3 × `conv2d(3×3, pad 1) + relu` (VGG-16 channel counts 64…512)
   - per-pixel L2 normalization across channels
   - `(norm(a) − norm(b))²`
   - learned 1×1 head conv reducing channels → 1
   - spatial mean
   - plus a 2×2 `max_pool2d` between blocks.
3. Sums the per-block scalars into one LPIPS-like output, computes
   `|wgpu - ndarray|`, and asserts it's below 1e-5.

## Verified results

Measured on an M-series Mac, with burn pinned to `0e05dc6e`
(the bisect baseline), cubek patched via `[patch]` to a local
checkout of the named rev:

| cubek rev                         |  \|wgpu − ndarray\|  | verdict |
| --------------------------------- | :------------------: | :-----: |
| `be1fef47` (parent of #191)       | **2.91e-10**         | PASS    |
| `662610b7` (#191 itself)          | **3.28e-5**          | FAIL    |
| `ac4b2e48` (burn 0e05dc6e's pin)  | **3.28e-5**          | FAIL    |

The drop from ~1e-10 to ~1e-5 (5 orders of magnitude) is specific
to #191. Pre-#191 wgpu agrees with ndarray essentially bit-for-bit
on this pipeline; post-#191 it drifts consistently.

Dropping the 1×1 head conv from each block — or going below ~128
channels — makes the drift vanish. Consistent with the regression
living in the tiled-matmul path #191 refactored.

## Running

```bash
cargo run --release
```

Deps default to `burn = 0e05dc6e02e18e2bd586d4119f5bb014c16adcb3`
with cubek pulled transitively (post-#191). On this config the
assertion fires with `|wgpu - ndarray| = 3.28e-5`.

## Confirming against pre-#191 cubek

`cargo [patch]` can't override a git source with another rev of the
same URL, so use a local checkout:

```bash
# In some workspace, e.g. /tmp:
git clone https://github.com/tracel-ai/cubek /tmp/cubek-pre-191
cd /tmp/cubek-pre-191
git checkout be1fef47dcbb152f8c62730e31d4176ac2bcd368

# be1fef47's cubek expects cubecl 34625c911...; burn 0e05dc6e uses
# cubecl 96d5f722... Pin cubek's cubecl dep to match burn's, or
# cargo will fail to resolve:
sed -i '' 's/34625c911c3e07b0438d91bea9af0f2233050589/96d5f7224a56979baca6dae0afbc0bc4d83a7c2f/g' Cargo.toml
```

Then uncomment the `[patch."https://github.com/tracel-ai/cubek"]`
block in this crate's `Cargo.toml`, pointing each `path = ...` at
`/tmp/cubek-pre-191/crates/<name>`, and run again. The assertion
should pass with `|wgpu - ndarray| < 1e-9`.

## Environment used for verification

- macOS (M-series), wgpu via Metal
- burn `0e05dc6e02e18e2bd586d4119f5bb014c16adcb3`
- cubecl `96d5f7224a56979baca6dae0afbc0bc4d83a7c2f`
- cubek — see table above

[pr191]: https://github.com/tracel-ai/cubek/commit/662610b79b969504f3ddc47dab719c9bba51a5b0
