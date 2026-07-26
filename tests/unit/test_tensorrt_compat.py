from pathlib import Path

import ultralytics.utils.export.engine as engine_export


def test_installed_ultralytics_export_uses_default_network_flags_for_tensorrt_10_plus():
    source = Path(engine_export.__file__).read_text(encoding="utf-8")

    assert "flag = 0 if is_trt10 else" in source
    assert "NetworkDefinitionCreationFlag.EXPLICIT_BATCH" in source


def test_installed_ultralytics_export_handles_removed_tensor_capability_properties():
    source = Path(engine_export.__file__).read_text(encoding="utf-8")

    assert "getattr(builder, \"platform_has_fast_fp16\", True)" in source
    assert "getattr(builder, \"platform_has_fast_int8\", True)" in source
