import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_csv(path: str | Path, headers: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_csv(path: str | Path, row: Iterable[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(row))


def timestamp_dir(base_dir: str | Path, name: str = "") -> Path:
    import datetime as _dt

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{name}" if name else ""
    return ensure_dir(Path(base_dir) / f"{ts}{suffix}")
