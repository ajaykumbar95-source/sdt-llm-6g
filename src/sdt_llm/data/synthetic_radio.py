"""
Synthetic 6G radio-channel data — stands in for Section 2.2's "Tokenized
Radio Channel Measurement" branch, and for what you'll eventually feed in
from ns-3 + Sionna RT ray tracing (your Step 2).

Field naming is chosen to mirror Sionna RT's actual `Paths` outputs so that
swapping this generator for real ray-traced data later is (close to) a
drop-in replacement:

    this module's field      Sionna RT equivalent (sionna.rt.Paths)
    ------------------------  --------------------------------------
    delay_s                   tau            (per-path delay, seconds)
    aoa_az_rad / aoa_el_rad   phi_r / theta_r (angle of arrival, azimuth/zenith)
    aod_az_rad / aod_el_rad   phi_t / theta_t (angle of departure, azimuth/zenith)
    path_gain (complex)       a              (complex path coefficient)
    doppler_hz                doppler        (Doppler shift; in Sionna RT this
                                               comes from object velocities via
                                               `scene.compute_paths(..., doppler=True)`
                                               or from the time-variation of `a`
                                               across time steps)

To actually wire in ns-3/Sionna RT later: replace `simulate_radio_scene()`
with code that calls `scene.compute_paths()` (or reads ns-3's Sionna-RT
mobility-integrated CSI trace) and repacks its `tau`/`theta_r`/`phi_r`/
`theta_t`/`phi_t`/`a` arrays into `MultipathComponent` objects with the same
field names used here — everything downstream (radio_encoder.py onward)
is agnostic to *how* the multipath components were produced.

Physics used here is deliberately simple (free-space path loss + a per-
category radar-cross-section proxy + i.i.d. small-scale Rayleigh-ish fading
+ a thermal noise floor) — good enough to produce *plausible, structured*
synthetic CSI for pipeline testing, not a substitute for Sionna's actual
ray tracing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from sdt_llm.data.synthetic_vision import SyntheticObject, ROOM_W_M, ROOM_D_M, ROOM_H_M

C = 299_792_458.0  # speed of light, m/s

# Per-category radar-cross-section proxy (relative, unitless) — bigger/denser
# objects reflect more energy back to the sensor. Purely illustrative.
CATEGORY_RCS: dict[str, float] = {
    "person": 1.0,
    "chair": 0.5,
    "table": 1.4,
    "book": 0.05,
    "box": 0.8,
    "wall_obstacle": 3.0,
}


@dataclass
class MultipathComponent:
    """One ray-traced (or here, synthetically generated) multipath component
    between a sensing node (Tx/Rx, monostatic) and a scatterer."""

    delay_s: float
    doppler_hz: float
    aoa_az_rad: float
    aoa_el_rad: float
    aod_az_rad: float
    aod_el_rad: float
    path_gain_re: float
    path_gain_im: float
    scatterer_id: Optional[str] = None   # ground truth, NOT visible to the encoder — eval/debug only

    @property
    def path_gain(self) -> complex:
        return complex(self.path_gain_re, self.path_gain_im)

    @property
    def range_m(self) -> float:
        """Monostatic round-trip range proxy: delay = 2*range / c. Used for
        rough proximity-based grouping of raw paths (radio_encoder.py's
        `_group_paths_by_proximity`), where exact correctness under bistatic
        geometry matters less than being monotonic in true distance, which
        this still is. For the actual REPORTED location of a detected
        object, see `bistatic_scatterer_location()` below, which is exact
        for both monostatic and bistatic geometry."""
        return self.delay_s * C / 2.0

    @property
    def radial_velocity_mps(self) -> float:
        """Doppler shift -> radial velocity at a given carrier (needs fc, so
        this is computed by the caller — see radio_encoder.py)."""
        raise NotImplementedError("use doppler_to_velocity(fc) in radio_encoder.py")


@dataclass
class RadioScene:
    timestamp: float
    sensor_pos: Tuple[float, float, float]   # receiver position — AoA/AoD and reported locations are relative to this
    carrier_hz: float
    paths: List[MultipathComponent]
    tx_pos: Optional[Tuple[float, float, float]] = None
    """Transmitter position, if different from `sensor_pos` (bistatic link,
    e.g. a real ns-3 gNB<->UE NR link). None means monostatic — tx and rx
    co-located (e.g. a dedicated ISAC sensing node), which is what
    simulate_radio_scene() below produces and what all the synthetic demos
    use. `bistatic_scatterer_location()` handles both cases correctly from
    one formula; it isn't a separate code path."""

    @property
    def effective_tx_pos(self) -> Tuple[float, float, float]:
        return self.tx_pos if self.tx_pos is not None else self.sensor_pos

    def to_manifest(self) -> dict:
        d = asdict(self)
        return d


def bistatic_scatterer_location(
    tx_pos: Tuple[float, float, float],
    rx_pos: Tuple[float, float, float],
    delay_s: float,
    aoa_az_rad: float,
    aoa_el_rad: float,
) -> Tuple[float, float, float]:
    """
    Exact scatterer location from a bistatic total path length (tx -> scatterer
    -> rx) and the angle of arrival at the receiver. Reduces exactly to the
    familiar monostatic round-trip formula (range = c*delay/2) when
    tx_pos == rx_pos — this is one formula, not two, so it's always safe to
    call even for a monostatic scene.

    Derivation: the scatterer S = rx + r*d for AoA unit direction d, subject
    to the ellipse constraint |tx - S| + |S - rx| = c*delay =: L. Substituting
    and solving the resulting linear equation for r gives:
        r = (|D|^2 - L^2) / (2 * (D . d - L)),   D = tx - rx
    (Verified against both the monostatic-reduction case and a genuinely
    bistatic case with tx and rx several metres apart — see tests/test_bistatic_localization.py.)
    """
    tx = np.asarray(tx_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)
    L = delay_s * C
    d = np.array([
        np.cos(aoa_el_rad) * np.cos(aoa_az_rad),
        np.cos(aoa_el_rad) * np.sin(aoa_az_rad),
        np.sin(aoa_el_rad),
    ])
    D = tx - rx
    denom = 2.0 * (float(np.dot(D, d)) - L)
    if abs(denom) < 1e-9:
        # tx/rx (nearly) co-located and/or a degenerate geometry -> monostatic fallback
        r = L / 2.0
    else:
        r = (float(np.dot(D, D)) - L * L) / denom
    r = max(r, 0.0)  # a negative solution is unphysical (can happen with noisy inputs near the degenerate case)
    scatterer = rx + r * d
    return (float(scatterer[0]), float(scatterer[1]), float(scatterer[2]))


def _direction_to_object(sensor_pos, obj_pos) -> Tuple[float, float, float]:
    dx = obj_pos[0] - sensor_pos[0]
    dy = obj_pos[1] - sensor_pos[1]
    dz = obj_pos[2] - sensor_pos[2]
    r = float(np.sqrt(dx**2 + dy**2 + dz**2))
    az = float(np.arctan2(dy, dx))
    el = float(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
    return r, az, el


def simulate_radio_scene(
    objects: List[SyntheticObject],
    timestamp: float,
    sensor_pos: Tuple[float, float, float] = (0.1, 0.1, 2.6),
    carrier_hz: float = 28e9,
    n_noise_paths: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> RadioScene:
    """
    Monostatic ISAC sensor (co-located Tx/Rx, e.g. a 6G gNB doing radar-style
    sensing) illuminates `objects`; returns one multipath component per
    object (its dominant reflection) plus a handful of noise-floor paths.
    """
    rng = rng or np.random.default_rng(0)
    paths: List[MultipathComponent] = []

    for obj in objects:
        r, az, el = _direction_to_object(sensor_pos, obj.world_pos)
        delay = 2 * r / C
        # radial velocity = component of object velocity along the sensor->object direction
        dir_vec = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        v_radial = float(np.dot(np.array(obj.velocity), dir_vec))
        doppler = 2 * v_radial * carrier_hz / C  # monostatic Doppler

        rcs = CATEGORY_RCS.get(obj.category, 0.3)
        # simplified monostatic radar equation (relative units): gain ~ sqrt(rcs) / r^2
        mean_amp = np.sqrt(rcs) / max(r, 0.3) ** 2
        # log-normal shadow fading (mild, ~3 dB std) + random phase. We use
        # shadowing rather than full Rayleigh small-scale fading here because
        # Rayleigh's occasional deep fades (>15 dB) would intermittently drop
        # real, persistent objects below the noise floor for a single frame,
        # which is physically real but makes for a confusing first demo of
        # the *tokenization/tracking* logic. Swap back to
        # `rng.rayleigh(scale=...)` if you specifically want to study
        # detection-under-fading behaviour.
        fade_db = rng.normal(0.0, 3.0)
        phase = rng.uniform(-np.pi, np.pi)
        amp = mean_amp * (10 ** (fade_db / 20.0))
        gain = amp * np.exp(1j * phase)

        paths.append(MultipathComponent(
            delay_s=delay, doppler_hz=doppler,
            aoa_az_rad=az, aoa_el_rad=el, aod_az_rad=az, aod_el_rad=el,
            path_gain_re=float(gain.real), path_gain_im=float(gain.imag),
            scatterer_id=obj.object_id,
        ))

    # thermal-noise-floor paths: random weak, incoherent reflections (multipath clutter)
    for _ in range(n_noise_paths):
        r = rng.uniform(0.5, max(ROOM_W_M, ROOM_D_M) * 1.4)
        az = rng.uniform(-np.pi, np.pi)
        el = rng.uniform(-0.3, 0.3)
        delay = 2 * r / C
        amp = rng.rayleigh(scale=1.0) * 0.02 / max(r, 0.3) ** 2
        phase = rng.uniform(-np.pi, np.pi)
        gain = amp * np.exp(1j * phase)
        paths.append(MultipathComponent(
            delay_s=delay, doppler_hz=float(rng.normal(0, 5)),
            aoa_az_rad=az, aoa_el_rad=el, aod_az_rad=az, aod_el_rad=el,
            path_gain_re=float(gain.real), path_gain_im=float(gain.imag),
            scatterer_id=None,
        ))

    return RadioScene(timestamp=timestamp, sensor_pos=sensor_pos, carrier_hz=carrier_hz, paths=paths)


def indoor_isac_scenario() -> List[RadioScene]:
    """
    A purely radio-sensed scenario with NO camera at all — this is the
    scenario `run_radio_sdt_llm_demo.py` uses to demonstrate
    "6G radio -> SDT -> LLM inference". A person walks across the room while
    a static obstacle (e.g. a pillar/shelving unit outside any camera's
    view) sits in a corner the whole time.
    """
    wall = SyntheticObject("wall_obstacle_1", "wall_obstacle", (5.4, 4.8, 1.2), size_m=0.6)
    box = SyntheticObject("box_1", "box", (1.0, 4.5, 0.5), size_m=0.4)
    rng = np.random.default_rng(42)

    scenes = []
    walk_path = [(0.8, 1.0), (1.6, 1.8), (2.4, 2.6), (3.2, 3.2), (4.0, 3.6), (4.6, 3.4)]
    n = len(walk_path)
    for t, (x, y) in enumerate(walk_path):
        # central difference (forward/backward at the boundaries) so velocity
        # stays non-zero at the last frame instead of artificially vanishing
        prev_i, next_i = max(t - 1, 0), min(t + 1, n - 1)
        dt = next_i - prev_i
        vx = (walk_path[next_i][0] - walk_path[prev_i][0]) / dt
        vy = (walk_path[next_i][1] - walk_path[prev_i][1]) / dt
        person = SyntheticObject(
            "person_1", "person", (x, y, 0.0), velocity=(vx, vy, 0.0),
            size_m=0.5, action="walking",
        )
        scene = simulate_radio_scene([wall, box, person], timestamp=float(t), rng=rng)
        scenes.append(scene)
    return scenes


def save_radio_scene(scene: RadioScene, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(scene.to_manifest(), indent=2))
    return p


def generate_and_save(out_dir: str = "data/synthetic/radio") -> List[Path]:
    out = Path(out_dir)
    paths = []
    for i, scene in enumerate(indoor_isac_scenario()):
        paths.append(save_radio_scene(scene, out, f"radio_{i:02d}"))
    return paths


if __name__ == "__main__":
    for p in generate_and_save():
        print("wrote", p)
