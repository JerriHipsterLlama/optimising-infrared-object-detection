from __future__ import annotations

from pathlib import Path

import pytest
import yaml
import onnx
from onnx import TensorProto, helper

from infrared_detection.evaluation.model_testing_matrix import (
    ModelTestingMatrixAdapters,
    discover_onnx_models,
    _build_local_engine,
    _benchmark_jetson_cpp,
    planned_test_rows,
    run_model_testing_matrix,
)


def _artifact(root: Path, candidate: str, precision: str) -> Path:
    path = root / candidate / precision / "model.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    graph_input = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("Identity", ["images"], ["output"])
    onnx.save(helper.make_model(helper.make_graph([node], candidate, [graph_input], [graph_output])), path)
    return path


def _config(tmp_path: Path, *, backend: str = "python") -> Path:
    artifacts = tmp_path / "onnx"
    _artifact(artifacts, "dense", "fp32")
    _artifact(artifacts, "dense", "fp16")
    _artifact(artifacts, "cluster-8-ratio-0.1", "int8")
    path = tmp_path / "testing.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {"directory": str(artifacts), "pattern": "**/model.onnx"},
                "testing": {
                    "backend": backend,
                    "dataset_yaml": str(tmp_path / "dataset.yaml"),
                    "split": "test",
                    "image_size": 352,
                    "device": "0",
                    "confidence": 0.25,
                    "iou": 0.7,
                    "output_dir": str(tmp_path / "results"),
                    "hardware_label": "test-device",
                    "workspace_mb": 2048,
                    "runner": str(tmp_path / "jetson_benchmark"),
                    "image_dir": str(tmp_path / "test-images"),
                    "warmup": 3,
                    "iterations": 9,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _adapters(*, fail_candidate: str | None = None):
    calls = []

    def evaluate_python(onnx_path, config):
        candidate = onnx_path.parent.parent.name
        calls.append(("python", candidate))
        if candidate == fail_candidate:
            raise RuntimeError("python evaluation failed")
        return {
            "map50": 0.8,
            "map50_95": 0.6,
            "precision": 0.75,
            "recall": 0.7,
            "per_class_ap": {"camel": 0.6},
        }

    def build_engine(onnx_path, engine_path, config):
        calls.append(("build", onnx_path.parent.parent.name))
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(b"target engine")
        return {"engine_path": str(engine_path), "engine_size_bytes": engine_path.stat().st_size}

    def benchmark_jetson(engine_path, config):
        calls.append(("benchmark", engine_path.parent.name))
        return {
            "latency_p50_ms": 12.5,
            "latency_p95_ms": 14.0,
            "fps": 80.0,
            "runtime": "jetson_cpp",
        }

    return ModelTestingMatrixAdapters(evaluate_python, build_engine, benchmark_jetson), calls


def test_discovery_is_sorted_and_respects_configured_glob(tmp_path):
    root = tmp_path / "onnx"
    second = _artifact(root, "z-candidate", "fp16")
    first = _artifact(root, "a-candidate", "fp32")
    _artifact(root, "ignored", "fp32").rename(root / "ignored" / "fp32" / "other.onnx")
    discovered = discover_onnx_models(root, "**/model.onnx")
    assert discovered == [first, second]


def test_ambiguous_precision_is_rejected(tmp_path):
    root = tmp_path / "onnx"
    ambiguous = root / "dense" / "converted" / "model.onnx"
    ambiguous.parent.mkdir(parents=True)
    ambiguous.write_bytes(b"onnx")
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"]["directory"] = str(root)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="precision"):
        planned_test_rows_from_path(config_path)


def planned_test_rows_from_path(path: Path):
    from infrared_detection.evaluation.model_testing_matrix import load_testing_config

    return planned_test_rows(load_testing_config(path))


def test_python_backend_reports_detection_metrics(tmp_path):
    adapters, _calls = _adapters()
    rows = run_model_testing_matrix(_config(tmp_path), adapters=adapters)

    row = next(row for row in rows if row["candidate_id"] == "dense" and row["precision"] == "fp32")
    assert row["python_status"] == "completed"
    assert row["python_metrics"]["map50"] == 0.8
    assert row["python_metrics"]["map50_95"] == 0.6
    assert row["python_metrics"]["precision"] == 0.75
    assert row["python_metrics"]["recall"] == 0.7
    assert row["python_metrics"]["per_class_ap"] == {"camel": 0.6}


def test_jetson_backend_builds_engine_from_onnx_then_invokes_runner(tmp_path):
    adapters, calls = _adapters()
    config_path = _config(tmp_path, backend="jetson_cpp")
    one_model = tmp_path / "one-model"
    _artifact(one_model, "dense", "fp32")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"]["directory"] = str(one_model)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    rows = run_model_testing_matrix(config_path, adapters=adapters)

    row = next(row for row in rows if row["precision"] == "fp32")
    assert calls[0] == ("build", "dense")
    assert calls[1] == ("benchmark", "fp32")
    assert row["jetson_cpp_status"] == "completed"
    assert row["jetson_cpp_metrics"]["latency_p50_ms"] == 12.5
    assert "map50" not in row["jetson_cpp_metrics"]
    assert Path(row["engine_path"]).is_file()


def test_both_backend_joins_metrics_by_candidate(tmp_path):
    adapters, _calls = _adapters()
    rows = run_model_testing_matrix(_config(tmp_path, backend="both"), adapters=adapters)

    row = next(row for row in rows if row["candidate_id"] == "dense" and row["precision"] == "fp16")
    assert row["status"] == "completed"
    assert row["python_metrics"]["map50"] == 0.8
    assert row["jetson_cpp_metrics"]["latency_p50_ms"] == 12.5


def test_backend_failure_isolated_per_model(tmp_path):
    adapters, _calls = _adapters(fail_candidate="dense")
    rows = run_model_testing_matrix(_config(tmp_path), adapters=adapters)

    dense = [row for row in rows if row["candidate_id"] == "dense"]
    cluster = next(row for row in rows if row["candidate_id"].startswith("cluster"))
    assert all(row["status"] == "failed" for row in dense)
    assert "python evaluation failed" in dense[0]["python_error"]
    assert cluster["status"] == "completed"


def test_jetson_failure_preserves_python_metrics_and_sibling_rows(tmp_path):
    base, _calls = _adapters()

    def build_engine(onnx_path, engine_path, config):
        if onnx_path.parent.parent.name == "dense":
            raise FileNotFoundError("trtexec unavailable")
        return base.build_engine(onnx_path, engine_path, config)

    adapters = ModelTestingMatrixAdapters(
        evaluate_python=base.evaluate_python,
        build_engine=build_engine,
        benchmark_jetson=base.benchmark_jetson,
    )
    rows = run_model_testing_matrix(_config(tmp_path, backend="both"), adapters=adapters)

    dense = next(row for row in rows if row["candidate_id"] == "dense" and row["precision"] == "fp32")
    cluster = next(row for row in rows if row["candidate_id"].startswith("cluster"))
    assert dense["status"] == "partial"
    assert dense["python_metrics"]["map50"] == 0.8
    assert dense["jetson_cpp_status"] == "failed"
    assert "trtexec unavailable" in dense["jetson_cpp_error"]
    assert cluster["status"] == "completed"


def test_engine_artifact_is_not_reused_across_targets(tmp_path):
    artifacts = tmp_path / "onnx"
    _artifact(artifacts, "dense", "fp32")
    outputs = []
    for label in ("rtx", "jetson"):
        config_path = _config(tmp_path, backend="jetson_cpp")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["models"]["directory"] = str(artifacts)
        config["testing"]["output_dir"] = str(tmp_path / label)
        config["testing"]["hardware_label"] = label
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        adapters, _calls = _adapters()
        row = run_model_testing_matrix(config_path, adapters=adapters)[0]
        outputs.append(Path(row["engine_path"]))
    assert outputs[0] != outputs[1]
    assert outputs[0].parent.parent.parent.name == "rtx"
    assert outputs[1].parent.parent.parent.name == "jetson"


def test_local_engine_build_uses_typed_onnx_without_precision_requantization(tmp_path, monkeypatch):
    from types import SimpleNamespace

    commands = []
    onnx_path = tmp_path / "dense" / "int8" / "model.onnx"
    engine_path = tmp_path / "engine" / "model.engine"

    def fake_run(command, **kwargs):
        commands.append(command)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(b"engine")
        return SimpleNamespace(stdout="built", stderr="")

    monkeypatch.setattr("infrared_detection.evaluation.model_testing_matrix.shutil.which", lambda name: "trtexec")
    monkeypatch.setattr("infrared_detection.evaluation.model_testing_matrix.subprocess.run", fake_run)
    config = {"testing": {"workspace_mb": 1024}}

    metadata = _build_local_engine(onnx_path, engine_path, config)

    assert "--stronglyTyped" in commands[0]
    assert not any(flag in commands[0] for flag in ("--fp16", "--int8", "--calib"))
    assert metadata["strongly_typed"] is True


def test_jetson_cpp_runner_receives_engine_images_and_iteration_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    runner = tmp_path / "jetson_benchmark"
    runner.write_bytes(b"runner")
    engine = tmp_path / "dense" / "fp16" / "model.engine"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    image_dir = tmp_path / "test-images"
    image_dir.mkdir()
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output-json") + 1])
        output.write_text(
            '{"latency_p50_ms": 9.0, "latency_p95_ms": 11.0, "fps": 100.0}',
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="runner output", stderr="")

    monkeypatch.setattr("infrared_detection.evaluation.model_testing_matrix.subprocess.run", fake_run)
    config = {
        "testing": {
            "runner": str(runner),
            "image_dir": str(image_dir),
            "warmup": 4,
            "iterations": 12,
        }
    }

    metrics = _benchmark_jetson_cpp(engine, config)

    assert commands[0][commands[0].index("--engine") + 1] == str(engine)
    assert commands[0][commands[0].index("--input-dir") + 1] == str(image_dir)
    assert commands[0][commands[0].index("--warmup") + 1] == "4"
    assert commands[0][commands[0].index("--iterations") + 1] == "12"
    assert metrics["latency_p50_ms"] == 9.0
    assert "map50" not in metrics


def test_missing_external_weights_are_reported_for_only_that_artifact(tmp_path):
    root = tmp_path / "onnx"
    broken = root / "broken" / "int8" / "model.onnx"
    broken.parent.mkdir(parents=True)
    graph_input = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
    weight.data_location = TensorProto.EXTERNAL
    entry = weight.external_data.add()
    entry.key = "location"
    entry.value = "weights.bin"
    node = helper.make_node("Add", ["images", "weight"], ["output"])
    onnx.save(helper.make_model(helper.make_graph([node], "broken", [graph_input], [graph_output], [weight])), broken)
    _artifact(root, "good", "fp32")
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"]["directory"] = str(root)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    rows = run_model_testing_matrix(config_path, adapters=_adapters()[0])

    broken_row = next(row for row in rows if row["candidate_id"] == "broken")
    good_row = next(row for row in rows if row["candidate_id"] == "good")
    assert broken_row["status"] == "failed"
    assert "missing external-weight sidecar" in broken_row["error"]
    assert str(broken) in broken_row["error"]
    assert good_row["status"] == "completed"


def test_unsupported_parser_error_names_artifact_and_isolated(tmp_path):
    config_path = _config(tmp_path, backend="jetson_cpp")

    def build_engine(onnx_path, engine_path, config):
        if "dense" in str(onnx_path):
            raise RuntimeError(f"Unsupported ONNX opset/parser for {onnx_path}")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(b"engine")
        return {"engine_path": str(engine_path), "engine_size_bytes": 6}

    adapters = ModelTestingMatrixAdapters(
        evaluate_python=lambda *args: {},
        build_engine=build_engine,
        benchmark_jetson=lambda *args: {"latency_p50_ms": 1.0},
    )
    rows = run_model_testing_matrix(config_path, adapters=adapters)

    failed = [row for row in rows if row["candidate_id"] == "dense"]
    successful = next(row for row in rows if row["candidate_id"].startswith("cluster"))
    assert all(row["jetson_cpp_status"] == "failed" for row in failed)
    assert str(failed[0]["onnx_path"]) in failed[0]["jetson_cpp_error"]
    assert "parser" in failed[0]["jetson_cpp_error"]
    assert successful["status"] == "completed"


def test_empty_artifact_directory_is_a_configuration_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"]["directory"] = str(root)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="No ONNX artifacts matched"):
        planned_test_rows_from_path(config_path)
