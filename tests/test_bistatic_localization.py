import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdt_llm.data.synthetic_radio import C, bistatic_scatterer_location  # noqa: E402


def _aoa_from(rx, scatterer):
    d = (np.asarray(scatterer) - np.asarray(rx))
    d = d / np.linalg.norm(d)
    az = float(np.arctan2(d[1], d[0]))
    el = float(np.arcsin(d[2]))
    return az, el


def test_monostatic_reduces_to_round_trip_formula():
    rx = tx = (0.1, 0.1, 2.6)
    true_scatterer = np.array([3.0, 3.0, 1.0])
    rng_true = np.linalg.norm(true_scatterer - np.array(rx))
    delay = 2 * rng_true / C
    az, el = _aoa_from(rx, true_scatterer)
    loc = bistatic_scatterer_location(tx, rx, delay, az, el)
    assert np.allclose(loc, true_scatterer, atol=1e-6)


def test_genuinely_bistatic_recovers_exact_location():
    tx = np.array([0.0, 0.0, 2.5])
    rx = np.array([6.0, 0.0, 2.5])
    scatterer = np.array([3.0, 4.0, 1.0])
    total_len = np.linalg.norm(scatterer - tx) + np.linalg.norm(scatterer - rx)
    delay = total_len / C
    az, el = _aoa_from(rx, scatterer)
    loc = bistatic_scatterer_location(tuple(tx), tuple(rx), delay, az, el)
    assert np.allclose(loc, scatterer, atol=1e-6)


def test_applying_monostatic_formula_to_bistatic_data_would_be_wrong():
    """Guards against silently regressing to the naive (wrong for bistatic)
    range/2 formula: confirms it disagrees with the correct answer whenever
    tx != rx with genuinely asymmetric geometry, so a future refactor can't
    quietly swap back to it unnoticed. (Note: a scatterer exactly equidistant
    from tx and rx is a degenerate case where the two formulas coincidentally
    agree — deliberately avoided here by using an off-center scatterer.)"""
    tx = np.array([0.0, 0.0, 2.5])
    rx = np.array([6.0, 0.0, 2.5])
    scatterer = np.array([1.0, 4.0, 1.0])  # off-center: much closer to tx than to rx
    total_len = np.linalg.norm(scatterer - tx) + np.linalg.norm(scatterer - rx)
    delay = total_len / C
    az, el = _aoa_from(rx, scatterer)

    correct = np.array(bistatic_scatterer_location(tuple(tx), tuple(rx), delay, az, el))
    naive_range = delay * C / 2.0
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    naive = rx + naive_range * d

    assert not np.allclose(correct, naive, atol=0.1), \
        "monostatic formula should visibly disagree with the correct bistatic answer here"
    assert np.allclose(correct, scatterer, atol=1e-6)


def test_degenerate_denominator_falls_back_gracefully():
    # tx == rx and az/el pointed anywhere should never raise or return nan/inf
    loc = bistatic_scatterer_location((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), delay_s=1e-8, aoa_az_rad=0.7, aoa_el_rad=0.1)
    assert all(np.isfinite(v) for v in loc)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
