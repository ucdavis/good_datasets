import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the pipeline uses paths relative to the repository root


@pytest.fixture(scope="session")
def process():
    """The processing module, loaded the way build.py loads it."""

    import src

    return src.inputs.write_data.load_module(str(ROOT / "Data" / "US" / "process.py"))
