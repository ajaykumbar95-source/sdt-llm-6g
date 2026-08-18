import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdt_llm.data.synthetic_radio import indoor_isac_scenario, simulate_radio_scene  # noqa: E402
from sdt_llm.data.synthetic_vision import book_reading_scenario, save_scene  # noqa: E402
from sdt_llm.fusion.temporal_alignment import align_clusters_across_time  # noqa: E402
from sdt_llm.pipeline import PipelineConfig, SDTLLMPipeline  # noqa: E402
from sdt_llm.sdt.digital_twin import SDTConfig  # noqa: E402


def test_temporal_alignment_matches_near_duplicates_and_flags_new():
    rng = np.random.default_rng(0)
    prev = rng.normal(size=(3, 16)).astype(np.float32)
    curr = np.concatenate([prev + rng.normal(size=(3, 16)) * 0.01, rng.normal(size=(1, 16)) * 5], axis=0)
    matches = align_clusters_across_time(prev, curr, d_c=0.35)
    assert len(matches) == 4
    assert {m.curr_index: m.prev_index for m in matches}[3] is None  # the novel point
    for i in range(3):
        assert {m.curr_index: m.prev_index for m in matches}[i] == i


def test_radio_pipeline_end_to_end_mock_llm():
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=128, k=3, cluster_ratio=0.7), llm_backend="mock",
    ))
    for scene in indoor_isac_scenario():
        record = pipeline.ingest_radio(scene)
        assert record.timestamp == scene.timestamp
        for fc in record.fused_clusters:
            assert fc.fused_token.embedding.shape == (128,)
            assert fc.track_id is not None

    result = pipeline.ask("Is anything moving in the room?", k_history=3)
    assert "MOCK LLM" in result["answer"]
    assert len(result["prompt"]) > 0

    # the three real, persistent objects should each keep one stable track id
    persistent = [tid for tid in pipeline.twin.all_track_ids() if len(pipeline.twin.query_track(tid)) >= 4]
    assert len(persistent) == 3


def test_vision_pipeline_end_to_end_mock_llm(tmp_path):
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=64, k=3, cluster_ratio=0.6), vision_backend="mock", llm_backend="mock",
    ))
    for i, scene in enumerate(book_reading_scenario()):
        img_path, _ = save_scene(scene, tmp_path, f"s{i}")
        record = pipeline.ingest_vision(img_path, timestamp=float(i))
        assert len(record.raw_tokens) > 0

    result = pipeline.ask("What is the person doing?")
    assert "MOCK LLM" in result["answer"]


def test_fused_pipeline_produces_at_least_one_cross_modal_cluster(tmp_path):
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=96, k=4, cluster_ratio=0.55, location_weight=0.7), llm_backend="mock",
    ))
    any_fused = False
    for i, vscene in enumerate(book_reading_scenario()):
        img_path, _ = save_scene(vscene, tmp_path, f"f{i}")
        rscene = simulate_radio_scene(vscene.objects, timestamp=vscene.timestamp, sensor_pos=(0.1, 0.1, 2.6))
        record = pipeline.ingest_multimodal(timestamp=vscene.timestamp, image_path=img_path, radio_scene=rscene)
        if any(fc.fused_token.modality == "fused" for fc in record.fused_clusters):
            any_fused = True
    assert any_fused, "expected at least one vision+radio cross-modal fused cluster across the session"


def test_empty_token_set_does_not_crash():
    pipeline = SDTLLMPipeline(PipelineConfig(sdt=SDTConfig(embed_dim=32), llm_backend="mock"))
    record = pipeline.twin.ingest([], timestamp=0.0)
    assert record.fused_clusters == []
    result = pipeline.ask("anything here?")
    assert isinstance(result["answer"], str)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_prompt_grounding_rules_prevent_radio_label_contradiction():
    from sdt_llm.llm.prompt_builder import SYSTEM_PREAMBLE

    assert "semantic labels as authoritative classifications" in SYSTEM_PREAMBLE
    assert "degraded_radio_link" in SYSTEM_PREAMBLE
    assert "good_radio_link" in SYSTEM_PREAMBLE
    assert "Do not contradict a semantic label" in SYSTEM_PREAMBLE


def test_prompt_forbids_invented_numeric_severity_labels():
    from sdt_llm.llm.prompt_builder import SYSTEM_PREAMBLE

    assert "Do not invent qualitative descriptors" in SYSTEM_PREAMBLE
    assert "actual measured value and unit" in SYSTEM_PREAMBLE

def test_canonical_ue_evidence_groups_measurements_by_rnti():
    from sdt_llm.llm.evidence import build_canonical_ue_evidence
    from sdt_llm.tokens import SemanticToken
    import numpy as np

    tokens = [
        SemanticToken(
            embedding=np.zeros(4, dtype=np.float32),
            label="good_radio_link",
            modality="fused",
            timestamp=0.49,
            location=(0.0, 0.0, 0.0),
            attributes={
                "rnti": 1,
                "sinr_db": 16.93,
                "packet_loss_pct": 0.0,
            },
        ),
        SemanticToken(
            embedding=np.zeros(4, dtype=np.float32),
            label="degraded_radio_link",
            modality="fused",
            timestamp=0.49,
            location=(0.0, 0.0, 0.0),
            attributes={
                "rnti": 2,
                "sinr_db": 6.89,
                "packet_loss_pct": 57.0,
            },
        ),
    ]

    evidence = build_canonical_ue_evidence(tokens)

    assert "UE / RNTI=1" in evidence
    assert "sinr_db=16.93" in evidence
    assert "UE / RNTI=2" in evidence
    assert "sinr_db=6.89" in evidence


def test_grounding_validator_rejects_contradictory_radio_condition():
    from sdt_llm.llm.grounding import (
        UEGroundTruth,
        validate_answer,
    )

    ground_truth = {
        "7.0.0.3": UEGroundTruth(
            ue_ip="7.0.0.3",
            rnti=2,
            radio_label="degraded_radio_link",
            network_label="high_packet_loss",
            sinr_db=6.89,
            packet_loss_pct=57.0,
            throughput_mbps=44.03,
            delay_ms=40.77,
        )
    }

    bad_answer = (
        "UE IP 7.0.0.3 has a good radio link and "
        "strong radio link."
    )

    valid, reason = validate_answer(
        bad_answer,
        ground_truth,
    )

    assert not valid
    assert "contradict" not in reason.lower() or "SDT" in reason


def test_grounding_validator_rejects_rnti_only_radio_contradiction():
    from sdt_llm.llm.grounding import (
        UEGroundTruth,
        validate_answer,
    )

    ground_truth = {
        "7.0.0.3": UEGroundTruth(
            ue_ip="7.0.0.3",
            rnti=2,
            radio_label="degraded_radio_link",
            network_label="high_packet_loss",
            sinr_db=6.89,
            packet_loss_pct=57.0,
            throughput_mbps=44.03,
            delay_ms=40.77,
        )
    }

    bad_answer = (
        "UE / RNTI=2 has a good_radio_link and "
        "a strong radio link."
    )

    valid, reason = validate_answer(
        bad_answer,
        ground_truth,
    )

    assert not valid
    assert "RNTI=2" in reason
