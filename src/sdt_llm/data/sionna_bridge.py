"""
Bridge: real Sionna RT ray-traced paths -> this project's MultipathComponent /
RadioScene format (see data/synthetic_radio.py, which this is a drop-in
replacement data source for).

Requires `sionna-rt` (NOT bundled in requirements.txt/-full.txt — it's a
large, fast-moving, hardware-sensitive dependency you install separately;
see the chat message this file was delivered with for exact commands).
This module does not import sionna at module load time, only inside
functions, so the rest of sdt_llm keeps working without it installed.

--------------------------------------------------------------------------
Verified against a real installation (sionna-rt 2.0.1) — not guessed
--------------------------------------------------------------------------
The public docs/examples are easy to misread here, so these shapes/behaviours
were confirmed by actually running PathSolver() on CPU (LLVM/Dr.Jit backend,
no GPU needed) and inspecting the returned Paths object directly:

  * Under the default `synthetic_array=True`, `paths.tau`, `.theta_t`,
    `.phi_t`, `.theta_r`, `.phi_r`, `.doppler`, and `.valid` all have shape
    (num_rx, num_tx, num_paths) — 3-D.
  * `paths.a` is a *tuple* `(real, imag)`, and EACH element has shape
    (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths) — 5-D, i.e. two MORE
    axes than tau/angles/doppler even under synthetic_array=True (per-antenna
    gain is still tracked even when the array is treated as one synthetic
    element for geometry purposes). Indexing `a` with the same 3 indices you
    use for `tau` will silently give you the wrong numbers — you need the
    antenna axes too (index 0 for both if you're using a 1x1 PlanarArray,
    which is the simplest useful configuration for an ISAC sensing node).
  * `theta_r`/`theta_t` are *zenith* angles (0 = straight up/+z, pi/2 =
    horizontal) in Sionna's convention — NOT the "elevation from horizontal"
    convention this project's MultipathComponent uses. Convert with
    `elevation = pi/2 - theta`.
  * To add a movable scatterer (e.g. a person) to a loaded scene, you must
    use `scene.edit(add=obj)` — NOT `scene.add(obj)`, which is reserved for
    Transmitter/Receiver/RadioMaterialBase and raises a ValueError for a
    SceneObject. `.position`, `.velocity` (both `mitsuba.Vector3f`/`Point3f`)
    and `.scaling` are real, working properties on the resulting SceneObject
    (confirmed via read-back), and setting `.velocity` is what feeds Sionna's
    own Doppler computation — you do not need to finite-difference position
    across frames yourself.
  * Valid `ITURadioMaterial(itu_type=...)` values (checked directly against
    the installed package): concrete, brick, plasterboard, wood, glass,
    ceiling_board, chipboard, plywood, marble, floorboard, metal,
    very_dry_ground, medium_dry_ground, wet_ground. There is no built-in
    "human" type — 'wood' is used below as a rough dielectric stand-in for a
    person-scale scatterer; if you need this to be quantitatively realistic,
    build a custom `sionna.rt.RadioMaterial(relative_permittivity=...,
    conductivity=...)` tuned to human-body RCS literature instead.

--------------------------------------------------------------------------
One thing NOT fully verified — check this yourself before trusting it
--------------------------------------------------------------------------
In a quick oblique-angle test (sensor and moving object not aligned on a
simple axis), the Doppler values Sionna returned for the moving object's
reflection came back as 0.0 Hz, which did not obviously match a
back-of-envelope radial-velocity calculation for that geometry. This may be
correct for reasons specific to that scene/geometry (e.g. the returned paths
may not have been the ones reflecting off the moving object at all — with
only a couple of valid paths found, it's hard to tell without visualizing
the scene), or it may indicate a setup mistake on my part. Before relying on
Sionna's Doppler output quantitatively, sanity-check it yourself on a
simple, unambiguous case (e.g. one object moving directly toward the
sensor, where radial velocity == speed) and compare against
`doppler = 2 * v_radial * carrier_hz / C`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sdt_llm.data.synthetic_radio import C, MultipathComponent, RadioScene


def paths_to_multipath_components(
    paths,
    carrier_hz: float,
    rx_index: int = 0,
    tx_index: int = 0,
    rx_ant_index: int = 0,
    tx_ant_index: int = 0,
    min_valid_only: bool = True,
) -> List[MultipathComponent]:
    """
    Convert a `sionna.rt.Paths` object (as returned by `PathSolver()(scene, ...)`)
    into a flat list of MultipathComponent for ONE (rx, tx, rx_ant, tx_ant)
    combination — the simplest useful case is a single-element PlanarArray on
    each side (num_rx_ant = num_tx_ant = 1), i.e. rx_ant_index=tx_ant_index=0.

    Assumes the default `synthetic_array=True` (see module docstring for the
    shape implications if you set it to False).
    """
    tau = np.asarray(paths.tau)
    theta_r = np.asarray(paths.theta_r)
    phi_r = np.asarray(paths.phi_r)
    theta_t = np.asarray(paths.theta_t)
    phi_t = np.asarray(paths.phi_t)
    doppler = np.asarray(paths.doppler) if hasattr(paths, "doppler") else np.zeros_like(tau)
    valid = np.asarray(paths.valid) if hasattr(paths, "valid") else np.ones_like(tau, dtype=bool)
    a_re, a_im = paths.a
    a_re, a_im = np.asarray(a_re), np.asarray(a_im)

    n_paths = tau.shape[-1]
    components: List[MultipathComponent] = []
    for p in range(n_paths):
        if min_valid_only and not bool(valid[rx_index, tx_index, p]):
            continue
        gain_re = float(a_re[rx_index, rx_ant_index, tx_index, tx_ant_index, p])
        gain_im = float(a_im[rx_index, rx_ant_index, tx_index, tx_ant_index, p])
        components.append(MultipathComponent(
            delay_s=float(tau[rx_index, tx_index, p]),
            doppler_hz=float(doppler[rx_index, tx_index, p]),
            # Sionna's theta is *zenith* (0=up); this project's *_el_rad is
            # elevation-from-horizontal (0=horizontal, +pi/2=up) -- convert.
            aoa_az_rad=float(phi_r[rx_index, tx_index, p]),
            aoa_el_rad=float(np.pi / 2 - theta_r[rx_index, tx_index, p]),
            aod_az_rad=float(phi_t[rx_index, tx_index, p]),
            aod_el_rad=float(np.pi / 2 - theta_t[rx_index, tx_index, p]),
            path_gain_re=gain_re,
            path_gain_im=gain_im,
            scatterer_id=None,  # Sionna doesn't label paths by which object they hit
        ))
    return components


def solve_to_radio_scene(
    scene,
    timestamp: float,
    sensor_pos: Tuple[float, float, float],
    carrier_hz: float,
    max_depth: int = 3,
    samples_per_src: int = 200_000,
) -> RadioScene:
    """Run PathSolver on `scene` right now and package the result as a
    RadioScene, ready to feed into RadioChannelEncoder.encode() unchanged."""
    from sionna.rt import PathSolver

    solver = PathSolver()
    paths = solver(scene, max_depth=max_depth, samples_per_src=samples_per_src)
    components = paths_to_multipath_components(paths, carrier_hz=carrier_hz)
    return RadioScene(timestamp=timestamp, sensor_pos=sensor_pos, carrier_hz=carrier_hz, paths=components)


@dataclass
class MovingScatterer:
    """One object you want to move through the scene across timestamps."""
    name: str
    itu_type: str = "wood"  # see module docstring for the valid list; 'wood' is a rough person-scale stand-in
    radius_m: float = 0.5
    thickness_m: float = 0.3
    base_mesh: Optional[str] = None  # defaults to sionna.rt.scene.sphere if None


def add_moving_scatterers(scene, scatterers: Sequence[MovingScatterer]):
    """
    Create SceneObjects for each MovingScatterer and add them to `scene` via
    `scene.edit()` (the correct call for geometry — `scene.add()` is only for
    Transmitter/Receiver/RadioMaterialBase and will raise ValueError here).
    Returns {name: SceneObject} so you can set .position/.velocity per
    timestep afterwards.
    """
    import sionna.rt.scene as scene_mod
    from sionna.rt import ITURadioMaterial, SceneObject
    from sionna.rt.utils.meshes import load_mesh

    objects = {}
    for s in scatterers:
        mesh = load_mesh(s.base_mesh or scene_mod.sphere)
        mat = ITURadioMaterial(name=f"{s.name}-mat", itu_type=s.itu_type, thickness=s.thickness_m)
        obj = SceneObject(mi_mesh=mesh, name=s.name, radio_material=mat)
        objects[s.name] = obj
    scene.edit(add=list(objects.values()))
    for s in scatterers:
        objects[s.name].scaling = s.radius_m
    return objects


def simulate_moving_scenario(
    base_scene_path: str,
    sensor_pos: Tuple[float, float, float],
    scatterer_tracks: dict,   # {name: [(t0, x,y,z, vx,vy,vz), (t1, ...), ...]}
    scatterer_specs: Sequence[MovingScatterer],
    carrier_hz: float = 28e9,
    max_depth: int = 3,
    samples_per_src: int = 200_000,
    variant: str = "llvm_ad_mono_polarized",
) -> List[RadioScene]:
    """
    End-to-end helper: load a scene, add your moving scatterers, step through
    `scatterer_tracks`, and run a fresh PathSolver at each timestep -> a list
    of RadioScene, drop-in compatible with RadioChannelEncoder.encode().

    `variant='llvm_ad_mono_polarized'` runs on CPU (no GPU needed — this is
    what you want on a laptop without an NVIDIA GPU). If you do have a CUDA
    GPU available, `'cuda_ad_mono_polarized'` will be much faster for larger
    scenes/sample counts.
    """
    import mitsuba as mi
    mi.set_variant(variant)
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(base_scene_path)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter(name="sensor_tx", position=list(sensor_pos)))
    scene.add(Receiver(name="sensor_rx", position=list(sensor_pos)))  # monostatic: co-located

    objects = add_moving_scatterers(scene, scatterer_specs)

    timestamps = sorted({row[0] for rows in scatterer_tracks.values() for row in rows})
    scenes: List[RadioScene] = []
    for t in timestamps:
        for name, rows in scatterer_tracks.items():
            row = next((r for r in rows if r[0] == t), None)
            if row is None:
                continue
            _, x, y, z, vx, vy, vz = row
            objects[name].position = mi.Point3f(x, y, z)
            objects[name].velocity = mi.Vector3f(vx, vy, vz)
        scenes.append(solve_to_radio_scene(
            scene, timestamp=t, sensor_pos=sensor_pos, carrier_hz=carrier_hz,
            max_depth=max_depth, samples_per_src=samples_per_src,
        ))
    return scenes


if __name__ == "__main__":
    # Minimal smoke test using a scene shipped with sionna-rt itself, so this
    # runs with zero custom-scene authoring. Needs `pip install -r
    # requirements-sionna.txt` (see chat) first.
    import sionna.rt.scene as scene_mod

    scenes = simulate_moving_scenario(
        base_scene_path=scene_mod.floor_wall,
        sensor_pos=(0.1, 0.1, 2.6),
        scatterer_tracks={
            "person": [
                (0.0, 1.0, 1.0, 1.0, 0.8, 0.0, 0.0),
                (1.0, 1.8, 1.0, 1.0, 0.8, 0.0, 0.0),
                (2.0, 2.6, 1.0, 1.0, 0.8, 0.0, 0.0),
            ],
        },
        scatterer_specs=[MovingScatterer(name="person", radius_m=0.4)],
    )
    for sc in scenes:
        print(f"t={sc.timestamp}: {len(sc.paths)} multipath components")
        for p in sc.paths:
            print(f"    range={p.range_m:.2f}m  doppler={p.doppler_hz:.1f}Hz  "
                  f"|gain|={abs(p.path_gain):.3e}  aoa_az={np.degrees(p.aoa_az_rad):.1f}deg")
