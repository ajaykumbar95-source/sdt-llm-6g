import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdt_llm.data.ns3_stream_bridge import (  # noqa: E402
    LiveBridgeConfig, rows_to_scenes, run_live, tail_csv_stream, write_csv_row,
)
from sdt_llm.data.synthetic_radio import MultipathComponent  # noqa: E402
from sdt_llm.pipeline import PipelineConfig, SDTLLMPipeline  # noqa: E402
from sdt_llm.sdt.digital_twin import SDTConfig  # noqa: E402


def _tmp_csv() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    f.close()
    return f.name


def test_tail_picks_up_rows_written_concurrently():
    path = _tmp_csv()
    comp = MultipathComponent(delay_s=1e-8, doppler_hz=0.0, aoa_az_rad=0.0, aoa_el_rad=0.0,
                               aod_az_rad=0.0, aod_el_rad=0.0, path_gain_re=0.01, path_gain_im=0.0)

    def producer():
        time.sleep(0.15)
        with open(path, "a") as f:
            for t in range(3):
                write_csv_row(f, float(t), (0.0, 0.0, 2.0), (0.0, 0.0, 2.0), 28e9, comp)
                time.sleep(0.15)

    threading.Thread(target=producer, daemon=True).start()
    rows = list(tail_csv_stream(path, poll_interval_s=0.05, stop_after_idle_s=0.6))
    assert len(rows) == 3
    assert [r[0] for r in rows] == [0.0, 1.0, 2.0]


def test_rows_to_scenes_groups_by_timestamp():
    comp = MultipathComponent(delay_s=1e-8, doppler_hz=0.0, aoa_az_rad=0.0, aoa_el_rad=0.0,
                               aod_az_rad=0.0, aod_el_rad=0.0, path_gain_re=0.01, path_gain_im=0.0)
    path = _tmp_csv()
    with open(path, "a") as f:
        write_csv_row(f, 0.0, (0, 0, 0), (0, 0, 0), 28e9, comp)
        write_csv_row(f, 0.0, (0, 0, 0), (0, 0, 0), 28e9, comp)
        write_csv_row(f, 1.0, (0, 0, 0), (0, 0, 0), 28e9, comp)
    rows = tail_csv_stream(path, poll_interval_s=0.05, stop_after_idle_s=0.3)
    scenes = list(rows_to_scenes(rows))
    assert len(scenes) == 2
    assert len(scenes[0].paths) == 2
    assert len(scenes[1].paths) == 1


def test_malformed_lines_are_skipped_not_fatal():
    path = _tmp_csv()
    with open(path, "a") as f:
        f.write("# a comment line, should be ignored\n")
        f.write("not,enough,columns\n")
        comp = MultipathComponent(delay_s=1e-8, doppler_hz=0.0, aoa_az_rad=0.0, aoa_el_rad=0.0,
                                   aod_az_rad=0.0, aod_el_rad=0.0, path_gain_re=0.01, path_gain_im=0.0)
        write_csv_row(f, 0.0, (0, 0, 0), (0, 0, 0), 28e9, comp)
    rows = list(tail_csv_stream(path, poll_interval_s=0.05, stop_after_idle_s=0.3))
    assert len(rows) == 1  # only the well-formed row survives


def test_run_live_end_to_end_with_pipeline():
    path = _tmp_csv()

    def producer():
        time.sleep(0.15)
        with open(path, "a") as f:
            for t in range(3):
                obstacle = MultipathComponent(delay_s=2 * 3.0 / 3e8, doppler_hz=0.0,
                                               aoa_az_rad=0.3, aoa_el_rad=0.0, aod_az_rad=0.3, aod_el_rad=0.0,
                                               path_gain_re=0.05, path_gain_im=0.0)
                write_csv_row(f, float(t), (0.1, 0.1, 2.6), (0.1, 0.1, 2.6), 28e9, obstacle)
                time.sleep(0.15)

    threading.Thread(target=producer, daemon=True).start()
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=64, k=3, cluster_ratio=0.7), llm_backend="mock",
    ))
    run_live(pipeline, LiveBridgeConfig(
        csv_path=path, poll_interval_s=0.05, stop_after_idle_s=0.6, ask_every_n_scenes=0,
    ))
    assert len(pipeline.twin.history) == 3
    persistent = [tid for tid in pipeline.twin.all_track_ids() if len(pipeline.twin.query_track(tid)) >= 2]
    assert len(persistent) == 1  # the one static obstacle should keep one stable track id


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
