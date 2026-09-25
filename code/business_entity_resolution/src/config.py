"""Central paths and constants. Override paths with env vars ER_DATA_DIR / ER_WORK_DIR if needed."""
import os
from pathlib import Path

# <root>/code/business_entity_resolution/src/config.py -> parents[3] = <root>
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))  # intermediate files (gitignored)
SOURCES = ["source1", "source2", "source3"]
SEED = 42
