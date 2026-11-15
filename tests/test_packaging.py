"""Regressions for base-install dependency checks with optional extras."""

import runpy
import zipfile
from pathlib import Path

import pytest

SMOKE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke_wheel.py"
CHECK_METADATA = runpy.run_path(str(SMOKE_SCRIPT))["check_metadata"]


@pytest.mark.parametrize(
    ("requirements", "has_base_dependencies"),
    [
        ([], False),
        (["optional-package; extra == 'example'"], False),
        (["optional-package; extra == 'example' and python_version >= '3.10'"], False),
        (["required-package"], True),
        (["required-package; python_version >= '3.10'"], True),
        (["required-package; extra == 'example' or python_version >= '3.10'"], True),
    ],
)
def test_base_dependencies_respect_extra_markers(tmp_path, requirements, has_base_dependencies):
    wheel = tmp_path / "example.whl"
    metadata = "Metadata-Version: 2.4\nName: example\nVersion: 1\n"
    metadata += "".join(f"Requires-Dist: {value}\n" for value in requirements)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("example-1.dist-info/METADATA", metadata)
    if has_base_dependencies:
        with pytest.raises(RuntimeError, match="Base install has runtime dependencies"):
            CHECK_METADATA(wheel)
    else:
        CHECK_METADATA(wheel)
