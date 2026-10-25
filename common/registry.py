"""Load a model folder by path, e.g. load("task1_reservoir_monthly/05_lightgbm_quantile").Model()."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(rel):
    path = REPO / rel / "model.py"
    spec = importlib.util.spec_from_file_location("m_" + rel.replace("/", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def folders(task):
    """Model folders of one task, in order."""
    return [f"{task}/{p.name}" for p in sorted((REPO / task).iterdir()) if (p / "model.py").exists()]
