import pytest

from wa_whisper.compute_mode import (
    ComputeMode,
    ComputeModeError,
    compute_mode_file,
    device_settings_for_mode,
    read_compute_mode,
    resolve_compute_device,
    write_compute_mode,
)


def test_mode_parser_accepts_supported_modes():
    assert ComputeMode.parse("gpu") is ComputeMode.GPU
    assert ComputeMode.parse("ram") is ComputeMode.RAM


@pytest.mark.parametrize("value", ["", "cpu", "cuda", "GPU", "gpu ram"])
def test_mode_parser_rejects_unsupported_modes(value):
    with pytest.raises(ComputeModeError, match="Expected one of: gpu, ram"):
        ComputeMode.parse(value)


def test_missing_mode_file_defaults_to_gpu(tmp_path):
    assert read_compute_mode(config_dir=tmp_path) is ComputeMode.GPU


def test_invalid_mode_file_fails_clearly(tmp_path):
    path = compute_mode_file(config_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("cuda\n", encoding="utf-8")

    with pytest.raises(ComputeModeError, match="Invalid compute mode"):
        read_compute_mode(config_dir=tmp_path)


def test_atomic_write_stores_selected_mode(tmp_path):
    path = write_compute_mode(ComputeMode.RAM, config_dir=tmp_path)

    assert path == compute_mode_file(config_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "ram\n"
    assert read_compute_mode(config_dir=tmp_path) is ComputeMode.RAM


def test_ram_mode_maps_to_cpu_without_fp16():
    settings = device_settings_for_mode(ComputeMode.RAM)

    assert settings.compute_mode is ComputeMode.RAM
    assert settings.device == "cpu"
    assert settings.fp16 is False


def test_gpu_mode_maps_to_cuda_with_fp16():
    settings = device_settings_for_mode(ComputeMode.GPU)

    assert settings.compute_mode is ComputeMode.GPU
    assert settings.device == "cuda"
    assert settings.fp16 is True


def test_device_override_still_works_for_direct_runs(tmp_path):
    settings = resolve_compute_device(device_override="cpu", config_dir=tmp_path)

    assert settings.compute_mode is None
    assert settings.device == "cpu"
    assert settings.fp16 is False


def test_device_override_and_compute_mode_are_rejected(tmp_path):
    with pytest.raises(ComputeModeError, match="--device cannot be used with --compute-mode"):
        resolve_compute_device(
            compute_mode_override="ram",
            device_override="cuda",
            config_dir=tmp_path,
        )
