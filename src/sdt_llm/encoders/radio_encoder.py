"""
Radio channel encoder — Section 2.2 "Tokenized Radio Channel Measurement".

  "the CSI can reveal insights about the current environment, such as the
  presence of obstacles, sources of interference, or moving objects...
  extracting distance, velocity, and angle information from the wireless
  signal... tokenized into semantic concepts such as 'obstacle',
  'interference', or 'movement', accompanied by relevant parameters such as
  location, path loss, and direction."

There is no pretrained model to plug in here — CSI-to-semantics is exactly
the novel piece the paper proposes and does not release weights for, and it
is also the branch *you* are replacing the camera with. What we implement
is a transparent, physics-grounded pipeline that does NOT peek at ground
truth (it never reads `scatterer_id`):

  1. Noise-floor estimation (order-statistics / CA-CFAR-style): use the
     *population of paths in the current frame itself* to estimate what
     "background clutter" looks like, so detection doesn't require any
     hand-tuned absolute power threshold.
  2. Detection + spatial clustering: paths clearly above the noise floor are
     grouped by proximity in (range, azimuth, elevation) — paths close in
     all three likely bounced off the same physical scatterer.
  3. Feature extraction per cluster: range, radial velocity (from Doppler),
     direction, aggregate gain -> a world-frame `location`, exactly like the
     vision branch, so radio- and vision-derived tokens can later be fused
     by the same DPC-KNN + transformer stage (Sec. 2.3).
  4. Rule-based semantic labelling matching the paper's three example
     concepts: "movement" (significant radial velocity), "obstacle"
     (static, clearly-above-noise reflector), "interference" (detected,
     but only marginally above the noise floor / geometrically diffuse —
     i.e. we're not confident it's a discrete physical object).

Swap-in point for ns-3 + Sionna RT: replace the `RadioScene`/
`MultipathComponent` objects consumed by `encode()` with ones built from
real ray-traced paths (see data/synthetic_radio.py's module docstring for
the exact field mapping) — nothing in this file needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sdt_llm.encoders.base import BaseSensorEncoder, SeededLinearProjection
from sdt_llm.tokens import SemanticToken
from sdt_llm.data.synthetic_radio import MultipathComponent, RadioScene, C, bistatic_scatterer_location


def doppler_to_radial_velocity(doppler_hz: float, carrier_hz: float) -> float:
    """v_radial = doppler * c / (2 * fc)  (monostatic sensing)."""
    return doppler_hz * C / (2.0 * carrier_hz)


def _direction_vector(az: float, el: float) -> np.ndarray:
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


@dataclass
class RadioEncoderConfig:
    embed_dim: int = 256
    velocity_threshold_mps: float = 0.15     # |v_radial| above this -> "movement"
    cluster_range_tol_m: float = 0.4         # grouping tolerance in range
    cluster_angle_tol_rad: float = 0.12      # grouping tolerance in az/el (~7 deg)
    # CFAR-style noise floor: a LOW percentile, not the median. With only a
    # handful of paths per frame, real targets can easily be ~40-50% of the
    # population; using the 50th percentile lets strong real returns
    # contaminate the "noise" estimate and self-defeat the threshold (a
    # target can end up needing to beat *itself*). A low percentile (default:
    # 25th) stays inside the true clutter/thermal-noise population in the
    # common case where genuine detections are a minority of all paths.
    noise_floor_percentile: float = 25.0
    detection_margin_db: float = 6.0         # must beat noise floor by this much to count as a detection
    interference_margin_db: float = 3.0      # within [margin, margin+this] of the floor -> "interference", not "obstacle"
    # With very few paths in a frame, a percentile-based noise floor is
    # statistically meaningless (e.g. with 1-2 paths, "the 25th percentile"
    # essentially just measures the signal against itself and can filter out
    # every real detection -- this is not hypothetical, it's what happens
    # with a sparse, low-multipath monostatic scene, which is a realistic
    # case for real ns-3/Sionna RT output, not just noisy synthetic data).
    # Below this many paths, skip percentile-based CFAR entirely and accept
    # every path as a detection (still labelled movement/obstacle/interference
    # by velocity+SNR-vs-a-fixed-floor below) rather than silently dropping
    # everything.
    min_paths_for_cfar: int = 5
    absolute_noise_floor_db: float = -80.0   # fallback "floor" when min_paths_for_cfar isn't met
    seed: int = 23


def _group_paths_by_proximity(
    ranges: np.ndarray, azs: np.ndarray, els: np.ndarray, cfg: RadioEncoderConfig
) -> List[np.ndarray]:
    """Union-find style grouping of path indices whose (range, az, el) are all
    within tolerance of each other — a simple stand-in for a proper CFAR +
    DBSCAN detector, adequate for the modest path counts ISAC scenes produce."""
    n = len(ranges)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if (abs(ranges[i] - ranges[j]) <= cfg.cluster_range_tol_m
                    and abs(azs[i] - azs[j]) <= cfg.cluster_angle_tol_rad
                    and abs(els[i] - els[j]) <= cfg.cluster_angle_tol_rad):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(v) for v in groups.values()]


class RadioChannelEncoder(BaseSensorEncoder):
    modality = "radio"

    def __init__(self, config: Optional[RadioEncoderConfig] = None):
        self.cfg = config or RadioEncoderConfig()
        # feature vector: [range, radial_v, az, el, gain_db, snr_db, doppler_hz(scaled), n_members, 0,0,0,0]
        self._proj = SeededLinearProjection(in_dim=12, out_dim=self.cfg.embed_dim, seed=self.cfg.seed)

    def encode(self, raw_input: RadioScene, timestamp: float) -> List[SemanticToken]:
        cfg = self.cfg
        scene = raw_input
        paths = scene.paths
        if not paths:
            return []

        ranges = np.array([p.range_m for p in paths])
        delays = np.array([p.delay_s for p in paths])
        azs = np.array([p.aoa_az_rad for p in paths])
        els = np.array([p.aoa_el_rad for p in paths])
        dopplers = np.array([p.doppler_hz for p in paths])
        gains = np.array([abs(p.path_gain) for p in paths])
        gains_db = 20 * np.log10(np.clip(gains, 1e-12, None))

        if len(paths) < cfg.min_paths_for_cfar:
            noise_floor_db = cfg.absolute_noise_floor_db
        else:
            noise_floor_db = float(np.percentile(gains_db, cfg.noise_floor_percentile))

        groups = _group_paths_by_proximity(ranges, azs, els, cfg)

        tokens: List[SemanticToken] = []
        for idx in groups:
            # aggregate via the strongest path in the group (dominant scatterer response)
            dom = idx[np.argmax(gains[idx])]
            peak_db = gains_db[dom]
            snr_db = peak_db - noise_floor_db
            if snr_db < cfg.detection_margin_db:
                continue  # below detection threshold -> not emitted as a token at all

            mean_range = float(ranges[idx].mean())
            mean_delay = float(delays[idx].mean())
            mean_az = float(np.angle(np.exp(1j * azs[idx]).mean()))  # circular mean
            mean_el = float(els[idx].mean())
            v_radial = doppler_to_radial_velocity(float(dopplers[dom]), scene.carrier_hz)

            # Exact for both monostatic (tx==rx, e.g. a dedicated ISAC sensor)
            # and bistatic (tx!=rx, e.g. a real gNB<->UE NR link) geometry —
            # see data/synthetic_radio.py's bistatic_scatterer_location() for
            # the derivation. Using the naive range/2-from-receiver formula
            # here would be a real physics error whenever tx_pos != sensor_pos.
            location = bistatic_scatterer_location(
                scene.effective_tx_pos, scene.sensor_pos, mean_delay, mean_az, mean_el,
            )

            if abs(v_radial) > cfg.velocity_threshold_mps:
                label = "movement"
            elif snr_db < cfg.detection_margin_db + cfg.interference_margin_db:
                label = "interference"
            else:
                label = "obstacle"

            feat = np.array([
                mean_range / 10.0, v_radial / 5.0, mean_az / np.pi, mean_el / np.pi,
                peak_db / 40.0, snr_db / 40.0, float(dopplers[dom]) / 1000.0,
                len(idx) / 5.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32)
            embedding = self._proj(feat)

            tokens.append(SemanticToken(
                embedding=embedding,
                label=label,
                modality="radio",
                timestamp=timestamp,
                location=(float(location[0]), float(location[1]), float(location[2])),
                attributes={
                    "range_m": round(mean_range, 2),
                    "radial_velocity_mps": round(v_radial, 3),
                    "aoa_az_deg": round(np.degrees(mean_az), 1),
                    "aoa_el_deg": round(np.degrees(mean_el), 1),
                    "snr_db": round(snr_db, 1),
                    "n_paths": int(len(idx)),
                },
                confidence=float(np.clip(snr_db / 20.0, 0.05, 0.99)),
            ))
        return tokens
