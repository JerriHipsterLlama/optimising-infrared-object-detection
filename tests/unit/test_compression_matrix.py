import pytest

from infrared_detection.training.compression_matrix import run_variant


def test_run_variant_dispatches_known_variant_in_dry_run(tmp_path):
    result = run_variant(
        "dense_fp32",
        base_checkpoint=tmp_path / "best.pt",
        config={"experiment": {"output_dir": str(tmp_path)}, "runtime": {"precision": "fp32"}, "dry_run": True},
    )

    assert result["variant"] == "dense_fp32"
    assert result["status"] == "dry_run"


def test_run_variant_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="Unknown compression variant"):
        run_variant("not_a_variant", tmp_path / "best.pt", {"experiment": {"output_dir": str(tmp_path)}})
