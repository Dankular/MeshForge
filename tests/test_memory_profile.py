import torch

from avatar_pipeline.runtime.memory import (
    create_shared_profile,
    validate_cpu_constructed,
    validate_shared_profile,
)


def test_shared_mmgp_profile_owns_cpu_constructed_modules():
    first = torch.nn.Linear(16, 16)
    second = torch.nn.Linear(16, 8)
    modules = {"first": first, "second": second}

    validate_cpu_constructed(modules)
    manager, report = create_shared_profile(modules)
    validate_shared_profile(manager, modules)

    assert report.module_names == ("first", "second")
    assert manager.models["first"] is first
    assert manager.models["second"] is second
    assert all(parameter.device.type == "cpu" for parameter in first.parameters())
    assert all(parameter.device.type == "cpu" for parameter in second.parameters())


def test_shared_mmgp_profile_rejects_duplicate_module_registration():
    module = torch.nn.Linear(4, 4, device="cpu")
    try:
        validate_cpu_constructed({"first": module, "duplicate": module})
    except RuntimeError as exc:
        assert "registered twice" in str(exc)
    else:
        raise AssertionError("duplicate heavy module registration was accepted")
