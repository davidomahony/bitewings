"""Central place for the data, checkpoint and work-dir locations.

Every script takes these from command-line options first, then from the
environment variables below, and finally falls back to the original
repository layout (data and checkpoints next to the repository).

    BITEWINGS_DATA_ROOT    folder with images/, annotations.json and splits/
    BITEWINGS_CHECKPOINTS  folder with the *.pth checkpoints
    BITEWINGS_WORK_DIRS    folder where training runs and predictions are written
    BITEWINGS_ODONTO_ROOT  folder with the OdontoAI panoramic data (pre-training only)

The configuration files read the same variables through mmengine's
``{{$VAR:default}}`` substitution, so a single set of variables drives both.
"""
import os
from pathlib import Path

DEFAULT_DATA_ROOT = '../data/Netherlands'
DEFAULT_CHECKPOINTS = '../checkpoints'
DEFAULT_WORK_DIRS = 'work_dirs'
DEFAULT_ODONTO_ROOT = '../data/odontoai'


def data_root() -> Path:
    return Path(os.environ.get('BITEWINGS_DATA_ROOT', DEFAULT_DATA_ROOT))


def checkpoints() -> Path:
    return Path(os.environ.get('BITEWINGS_CHECKPOINTS', DEFAULT_CHECKPOINTS))


def work_dirs() -> Path:
    return Path(os.environ.get('BITEWINGS_WORK_DIRS', DEFAULT_WORK_DIRS))


def odonto_root() -> Path:
    return Path(os.environ.get('BITEWINGS_ODONTO_ROOT', DEFAULT_ODONTO_ROOT))


def method_work_dir(method: str, base: Path = None) -> Path:
    """Work dir of a chart-filing run, e.g. work_dirs/chart_filing_hierarchical."""
    return (base or work_dirs()) / f'chart_filing_{method}'
