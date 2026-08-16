# sdt-llm-6g

A reference implementation of the pipeline from Huawei's
**"Semantic Digital Twins: Enhancing Performance in Wireless Communication
and LLM Inference"**
([source](https://www.huawei.com/en/huaweitech/future-technologies/semantic-digital-twins-wireless-communication-llm-inference)),
plus your extension: **replacing the camera with 6G radio (ISAC/CSI) sensing**.

```
Paper:      vision  ────────────────► SDT ────► LLM inference
This repo:  vision  ─┐
            6G radio ─┴──────────────► SDT ────► LLM inference   (radio branch is new)
```

Both branches share the *same* SDT (clustering + fusion + tracking) and LLM
code — only the sensor encoder at the front differs. Step 2 (wiring in real
ns-3 + Sionna RT ray tracing in place of the synthetic radio generator) is
scoped out below and is a small, contained swap given how this is built.

---

## 1. What's real here, and what isn't

Being upfront about this before you build on it:

- **There is no public pretrained "SDT" model.** The Huawei page is a
  research/concept writeup, not a code or weights release. Everything under
  `src/sdt_llm/fusion/` and `src/sdt_llm/encoders/radio_encoder.py` is *our*
  implementation of the paper's described architecture and equations — real
  code, but not reproducing anyone's trained checkpoint (none exists to
  reproduce).
- **The transformer fusion block and the modality projections are
  architecturally faithful but untrained** (Xavier-initialised from a fixed
  seed). The paper mentions training happens ("multiple semantic token
  attributes... are aggregated into a cluster using semantic graphs" during
  training) but publishes no weights or training procedure. A training loop
  is the natural next step once you have a task loss — see §6.
- **The vision and LLM branches *do* have a real-pretrained-model option**
  (CLIP for vision tagging, any Hugging Face instruct LLM, or the
  Anthropic/OpenAI APIs) — gated behind `requirements-full.txt` since they
  need real weight downloads / an API key. Mock (offline, deterministic,
  zero-download) backends are the default so the whole pipeline runs
  instantly out of the box, which is what "check it on synthetic data" was
  asking for.
- **The DPC-KNN equations (1)-(3) as published on the Huawei page appear to
  have transcription issues** (an inequality that would invert the meaning
  of "k nearest neighbours", and a density formula that plugs in raw cosine
  *similarity* where a *distance* is needed for density-peaks clustering to
  behave sensibly). We implement the standard, well-behaved density-peaks
  formulation and document the discrepancy in code rather than silently
  guessing — see the long comment at the top of
  `src/sdt_llm/fusion/dpc_knn.py`. A `metric="cosine_similarity_literal"`
  mode is also provided so you can reproduce the equations exactly as
  written and compare.

If you're citing/reproducing this for a thesis or paper, please read that
`dpc_knn.py` docstring — it's short and matters.

---

## 2. Quickstart (Ubuntu)

```bash
cd sdt-llm-6g
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh              # installs numpy/scipy/pillow/requests, runs the test suite

PYTHONPATH=src python3 scripts/run_vision_sdt_llm_demo.py   # paper's own pipeline
PYTHONPATH=src python3 scripts/run_radio_sdt_llm_demo.py    # <- your idea: 6G radio -> SDT -> LLM
PYTHONPATH=src python3 scripts/run_fused_sdt_llm_demo.py    # paper's full method: both branches fused
```

No GPU, no API key, no model download needed for any of the above — the
default `mock` LLM and vision backends make the whole thing runnable in
seconds. See §5 to swap in a real LLM.

---

## 3. Paper → code map

| Paper section | What it describes | Code |
|---|---|---|
| 2.1 Semantic Sensor Data | camera → semantic tokens T^s | `encoders/vision_encoder.py` |
| 2.2 Tokenized Radio Channel Measurement | CSI → semantic tokens T^c ("obstacle"/"interference"/"movement" + location/loss/direction) | `encoders/radio_encoder.py` **(your new branch)** |
| 2.3 SDT Representation, Eq. (1)-(3) | DPC-KNN clustering | `fusion/dpc_knn.py` |
| 2.3 "transformer block ... fused token clusters" | per-cluster transformer + pooling | `fusion/token_fusion.py` |
| 2.3 "clusters ... matched if similarity distance < d_c" | cross-timestamp tracking | `fusion/temporal_alignment.py` |
| 2.3 "timestamp and location stamp... time, space, semantics" | stateful twin | `sdt/digital_twin.py` (`SemanticDigitalTwin`) |
| 3.2 Effective Prompt Engineering / Context-Aware Prediction | SDT context → LLM prompt, historical recall | `llm/prompt_builder.py`, `SemanticDigitalTwin.last_seen()` |
| 3.2 (LLM inference itself) | prompt → answer | `llm/{mock_llm,local_hf_llm,api_llm}.py` |
| — (glue) | end-to-end orchestration | `pipeline.py` (`SDTLLMPipeline`) |

Synthetic data generators (stand in for real sensors, see §7 for the radio
one's real-world swap-in path):
- `data/synthetic_vision.py` — procedurally drawn camera-schematic PNGs
- `data/synthetic_radio.py` — synthetic multipath components shaped exactly
  like real **Sionna RT** ray-tracing output

---

## 4. The three demos

**`run_vision_sdt_llm_demo.py`** — the paper's own pipeline. A synthetic
"person enters, sits, reads a book" scenario (mirroring the paper's own
example) is rendered as PNGs, tagged by a real (if simple) pixel-level
colour-blob detector — not a peek at ground truth — clustered/fused/tracked
by the SDT, then queried through the LLM.

**`run_radio_sdt_llm_demo.py`** — **your idea.** No camera anywhere. A
person walks across a room while a static obstacle and a box (both outside
any camera's hypothetical field of view) sit in the scene. A 6G monostatic
sensor (co-located Tx/Rx, e.g. a gNB doing radar-style ISAC sensing)
illuminates the room; `RadioChannelEncoder` turns the resulting multipath
components into "obstacle"/"movement"/"interference" tokens via a
CFAR-style detector (no ground truth is read — see §7), the same SDT code
clusters/fuses/tracks them, and the LLM answers questions purely from
radio-derived semantics.

**`run_fused_sdt_llm_demo.py`** — the paper's *actual* full method: the
same room, same objects, same instants, observed by **both** a camera and
the 6G sensor at once. Where a cluster ends up containing both a vision
token and a radio token for the same physical entity, its modality becomes
`"fused"` — that's the paper's central mechanism, working. Useful if you
want to *add* 6G sensing alongside a camera (e.g. so the twin still knows
where something is once it leaves the camera's view) rather than replace
the camera outright.

All three accept `--llm-backend {mock,hf_local,anthropic_api,openai_compatible_api}`.

---

## 5. Swapping in a real LLM (or real CLIP)

```bash
# on your own machine, with internet access:
pip install -r requirements-full.txt --break-system-packages
# CPU-only torch is much smaller if you don't have a GPU — see the note
# at the top of requirements-full.txt before the line above.

PYTHONPATH=src python3 scripts/run_radio_sdt_llm_demo.py --llm-backend hf_local
```

`LocalHFLLM` (`llm/local_hf_llm.py`) defaults to `Qwen/Qwen2.5-1.5B-Instruct`
— small enough for CPU inference, works with `apply_chat_template`. Any
similar instruct model works; check the Hub for what's current.

For a hosted model instead of local weights:

```bash
export ANTHROPIC_API_KEY=...       # or OPENAI_API_KEY for openai_compatible_api
PYTHONPATH=src python3 scripts/run_radio_sdt_llm_demo.py --llm-backend anthropic_api
```

For real (rather than colour-blob-mock) vision tagging, pass
`vision_backend="clip"` to `PipelineConfig` / `--vision-backend clip` on the
vision or fused demo — uses `openai/clip-vit-base-patch32` for zero-shot
tagging + real image embeddings.

---

## 6. Known limitations (and the fixes already applied)

Built and tested honestly, including the rough edges:

- **Untrained embeddings ⇒ clustering/tracking need a spatial assist.**
  A random linear projection has no guarantee that physically-related
  detections land near each other in embedding space. Both `dpc_knn_cluster`
  and `align_clusters_across_time` accept a `location_weight` that blends in
  physical (x, y, z) proximity — well-motivated by the paper's own "time,
  space, semantics" framing, not just a hack — and it's what makes track
  identity stable across a session (see the tuned defaults in `SDTConfig`).
  Set `location_weight=0` to see pure (noisier) embedding-only behaviour.
- **CFAR-style noise-floor estimation uses a *low* percentile, not the
  median.** With only a handful of radio paths per frame, real detections
  can be ~40% of the population; a median-based "noise floor" gets dragged
  up by real signals and self-defeats the detection threshold. See the
  comment in `RadioEncoderConfig.noise_floor_percentile`.
- **Monocular depth-from-size is scale-ambiguous** (a classic, real
  limitation of single-camera vision, not unique to this toy renderer): a
  small object (e.g. the book) can be estimated as farther away than it is.
  Locations are clamped to the room's physical bounds so this degrades
  gracefully rather than producing nonsense coordinates.
- **DPC-KNN equation ambiguity** — see §1 and `fusion/dpc_knn.py`.
- **Track continuity is good but not perfect** — under simultaneous fading
  dips and sparse detections, an identity swap is still possible in
  principle (this is a real, actively-researched problem in multi-object
  tracking generally, not something fully "solved" by any simple system).
  `SemanticDigitalTwin.query_track(track_id)` lets you inspect a track's
  full history if you want to audit this yourself.

---

## 7. Connecting to real 6G data — two paths

**Path A (built, verified, ready to use): `data/sionna_bridge.py` + real Sionna RT ray tracing.**

```bash
pip install -r requirements-sionna.txt --break-system-packages   # ~1-2 min, no GPU needed
PYTHONPATH=src python3 scripts/run_sionna_radio_demo.py
```

This calls **real** ray tracing (`sionna.rt.PathSolver`) against a scene shipped
with Sionna RT itself, moves a scatterer through it across several timestamps,
and feeds the *actual* ray-traced delays/angles/Doppler/gains into your
existing `RadioChannelEncoder` → `SemanticDigitalTwin` → LLM pipeline
completely unchanged. No GPU required — Mitsuba's `llvm_ad_mono_polarized`
variant ray-traces on CPU (slower than CUDA, but functional for scenes this
small; relevant if you're on a laptop without an NVIDIA GPU).

Every field mapping, array shape, and API call in `sionna_bridge.py` was
**verified by actually running it**, not inferred from docs — Sionna RT has
had several breaking API changes recently (`scene.compute_paths()` was
replaced by `PathSolver()`; `Paths.a` has a different shape than
`Paths.tau`; adding movable geometry needs `scene.edit()`, not `scene.add()`).
Read the module docstring before you build on it — it also documents one
thing I could *not* fully verify (Doppler on an oblique-angle moving
scatterer looked off in one quick test) so you know to sanity-check it.

To use your own room instead of a shipped scene: build it in Blender with the
[Mitsuba-Blender add-on](https://github.com/mitsuba-renderer/mitsuba-blender)
and export to Mitsuba XML — this is currently the reliable way to author
custom Sionna RT scenes (hand-writing scene XML without Blender has known,
currently-open rough edges in the Sionna community). `MovingScatterer` in
`sionna_bridge.py` then adds programmatic movable objects (spheres by
default) into whatever static scene you load, with no XML editing needed for
those.

**Path B: the official ns-3 + Sionna RT *online* integration.** Very recently
added directly to ns-3-dev (documented at
`nsnam.org/docs/installation/html/sionna-rt.html`) — ns-3's C++ core calls
into Sionna RT live, per channel-update event, via a pybind11 bridge, so
node mobility and ray-traced propagation stay in lockstep inside one
simulation. This is the "real" ns-3+Sionna combo if you want full
network-protocol simulation (packets, MAC/PHY, multi-node scheduling) with
ray-traced channels, not just mobility + channel data:

```bash
python3.12 -m venv sionna_env && source sionna_env/bin/activate
pip install sionna==1.2.0 sionna-rt==1.2.0 pybind11==2.11.1 cppyy==3.5.0
git clone https://gitlab.com/nsnam/ns-3-dev.git ns3dev && cd ns3dev
./ns3 configure --enable-examples --enable-tests --enable-python-bindings
./ns3 build
./ns3 run sionna-rt-channel-example   # verifies the install; writes snr-trace.txt
```

Notes if you go this route:
- It needs `ns-3-dev` (the git development branch), **not** a numbered
  release tarball — this feature is only days old as of this writing and
  isn't in a tagged release yet.
- The shipped example (`src/spectrum/examples/sionna-rt-channel-example.cc`)
  only prints an aggregate SNR trace. To get the per-path delay/angle/
  Doppler/gain detail `RadioChannelEncoder` actually needs, you'll want to
  extend that example (or write a similar one) to also export the raw
  per-path values — since ns-3's C++ side is calling the *same* `sionna-rt`
  Python package via pybind11, the extraction logic is the same as
  `paths_to_multipath_components()` in `sionna_bridge.py`; you're just
  invoking it from ns-3's embedded Python call site instead of a standalone
  script.
- Given the build complexity (full ns-3 C++ compile + pybind11 + pinned
  versions) versus Path A giving you real ray-traced data into the exact
  same downstream pipeline in a few minutes, Path A is the more practical
  starting point — reach for Path B once you specifically need ns-3's
  network-protocol simulation in the loop, not just realistic channels.

Either path produces the same `MultipathComponent`/`RadioScene` objects
(see the field-mapping table below), so `RadioChannelEncoder` and everything
downstream never needs to change:

| `MultipathComponent` field | Sionna RT (`sionna.rt.Paths`) |
|---|---|
| `delay_s` | `tau` |
| `aoa_az_rad` / `aoa_el_rad` | `phi_r` / `(pi/2 - theta_r)` — Sionna's `theta_r` is zenith, not elevation |
| `aod_az_rad` / `aod_el_rad` | `phi_t` / `(pi/2 - theta_t)` |
| `path_gain_re` / `path_gain_im` | `a[0][...]` / `a[1][...]` — `a` is a `(real, imag)` tuple, shape differs from `tau` |
| `doppler_hz` | `doppler` (from the scatterer's `.velocity`, or a Transmitter/Receiver's own `velocity=` param) |

A couple of things worth knowing once you're on real data: the CFAR threshold
and clustering tolerances in `RadioEncoderConfig` were tuned against the
synthetic generator's noise model, not real RF noise — expect to retune
`cluster_range_tol_m` / `cluster_angle_tol_rad` / `detection_margin_db`
against real traces. Real ray tracing also often returns several bounces per
object rather than one dominant path; `_group_paths_by_proximity` already
groups multi-path clusters, so this should mostly work as-is, but validate
against your own scenes.

## 7.5. Live/streaming mode: connecting a running ns-3 simulation

`data/ns3_stream_bridge.py` + `scripts/run_ns3_live_demo.py` let the SDT+LLM
pipeline consume data from a **live** ns-3 simulation instead of a batch
Python script — it tails a CSV file that ns-3 appends one row per multipath
component to (schema documented in the module docstring), groups rows into
`RadioScene`s by timestamp, and feeds each one into the pipeline as soon as
it's written, so you get near-real-time results while ns-3 is still running.

```bash
python3 scripts/run_ns3_live_demo.py --demo-producer   # proves the plumbing works, zero ns-3 needed
python3 scripts/run_ns3_live_demo.py --csv /tmp/sdt_radio_stream.csv --llm-backend mock   # the real thing
```

The Python (consumer) side is fully built and tested (`tests/test_ns3_stream_bridge.py`,
`tests/test_bistatic_localization.py`). The C++ (producer) side — the actual
patch to `SionnaRtChannelModel` that appends one CSV row per resolved path —
is in `ns3-patch/SDT_EXPORT_PATCH.md`, written against the real headers from
this project's own ns-3+5G-LENA+Sionna-RT build (not guessed).

One correctness note this surfaced: a real ns-3 NR link is **bistatic** (gNB
and UE at different positions), not monostatic like the earlier synthetic
demos. The naive round-trip range formula (`range = c*delay/2` from one
point) is a genuine physics error under bistatic geometry, not just an
approximation — `bistatic_scatterer_location()` in `synthetic_radio.py`
solves the correct tx/rx ellipse geometry instead (verified to reduce
exactly to the monostatic formula when tx and rx coincide, so the earlier
synthetic demos are unaffected — see `tests/test_bistatic_localization.py`).

Another robustness note: `RadioChannelEncoder`'s CFAR-style noise-floor
estimate needs a reasonable population of paths to be statistically
meaningful. With very few paths per update (realistic for a low-multipath
scene), the percentile-based floor can degenerate and filter out every
detection. `RadioEncoderConfig.min_paths_for_cfar` (default 5) now falls
back to a fixed `absolute_noise_floor_db` below that count.

---

## 8. Project layout

```
src/sdt_llm/
  tokens.py                    SemanticToken / FusedCluster data model
  fusion/
    dpc_knn.py                 Eq (1)-(3): density-peaks clustering
    token_fusion.py            transformer fusion block + attention pooling
    temporal_alignment.py      cross-timestamp track matching
  encoders/
    base.py                    shared encoder interface + seeded projection
    vision_encoder.py          T^s: image -> tokens (mock colour-blob | real CLIP)
    radio_encoder.py           T^c: multipath -> tokens (CFAR-style, rule-based labels)
  sdt/digital_twin.py           SemanticDigitalTwin: orchestrates the above, keeps history
  llm/
    base.py, factory.py         backend interface + selector
    mock_llm.py                 deterministic offline stand-in (default)
    local_hf_llm.py             real local HF transformers backend (optional)
    api_llm.py                  Anthropic / OpenAI-compatible API backend (optional)
    prompt_builder.py           SDT tokens -> structured LLM prompt (Sec 3.2)
  data/
    synthetic_vision.py         synthetic camera scenes + renderer
    synthetic_radio.py          synthetic multipath, Sionna-RT-shaped
    sionna_bridge.py             REAL Sionna RT ray tracing -> MultipathComponent (optional, needs requirements-sionna.txt)
    ns3_stream_bridge.py         Live CSV-tailing bridge for a running ns-3 simulation (see §7.5)
  pipeline.py                   SDTLLMPipeline: the one class demos wrap
scripts/                        the three synthetic demos + real-Sionna demo + live ns-3 demo + dataset regeneration
tests/                          pytest suite (23 tests, all passing)
ns3-patch/SDT_EXPORT_PATCH.md   Exact C++ patch for SionnaRtChannelModel (§7.5)
```

## 9. Tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

Covers: DPC-KNN correctness on synthetic blobs (incl. the location-blend
behaviour) and edge cases (n=0, n=1); the fusion transformer's shapes,
determinism, and single-token passthrough; temporal alignment's
match/no-match behaviour; and end-to-end smoke tests for all three
pipelines (vision-only, radio-only, fused) including a check that
cross-modal fusion actually happens in the fused demo, and that the three
real, persistent objects in the radio demo each keep one stable track ID
across the session.
