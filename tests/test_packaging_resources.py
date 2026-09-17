from __future__ import annotations

import pytest

from vadbench.resources import package_resource_path


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/configs/encoders/videomaev2-base.yaml",
        "C:\\\\configs\\\\encoders\\\\videomaev2-base.yaml",
        "../registry/checkpoints.yaml",
        "configs/../registry/checkpoints.yaml",
    ],
)
def test_package_resource_path_rejects_paths_outside_the_package(value: str) -> None:
    with pytest.raises(ValueError):
        package_resource_path(value)
