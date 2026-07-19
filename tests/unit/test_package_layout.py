from pathlib import Path


def test_infrared_detection_package_is_importable_from_src():
    import infrared_detection

    package_path = Path(infrared_detection.__file__).resolve().parent
    expected_path = Path(__file__).resolve().parents[2] / "src" / "infrared_detection"

    assert package_path == expected_path
