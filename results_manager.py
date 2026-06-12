from __future__ import annotations
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List
import numpy as np


@dataclass
class AnalysisResult:
    result_id: str
    result_type: str
    name: str
    timestamp: float = field(default_factory=time.time)
    source_file: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.get("data", {}).items():
            if isinstance(v, np.ndarray):
                d["data"][k] = v.tolist()
        for k, v in d.get("params", {}).items():
            if isinstance(v, np.ndarray):
                d["params"][k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisResult":
        for k, v in d.get("data", {}).items():
            if isinstance(v, list):
                d["data"][k] = np.array(v)
        for k, v in d.get("params", {}).items():
            if isinstance(v, list):
                d["params"][k] = np.array(v)
        return cls(**d)


class ResultsManager:
    def __init__(self):
        self._results: List[AnalysisResult] = []
        self._counter: int = 0
        self._callbacks: List[Any] = []

    def add_result(self, result_type: str, name: str,
                   data: Dict[str, Any],
                   params: Optional[Dict[str, Any]] = None,
                   source_file: str = "") -> AnalysisResult:
        self._counter += 1
        rid = f"{result_type}_{self._counter}"
        r = AnalysisResult(
            result_id=rid,
            result_type=result_type,
            name=name,
            source_file=source_file,
            params=params or {},
            data=data,
        )
        self._results.append(r)
        self._notify()
        return r

    def remove_result(self, result_id: str):
        self._results = [r for r in self._results if r.result_id != result_id]
        self._notify()

    def get_results(self, result_type: Optional[str] = None) -> List[AnalysisResult]:
        if result_type is None:
            return list(self._results)
        return [r for r in self._results if r.result_type == result_type]

    def get_result(self, result_id: str) -> Optional[AnalysisResult]:
        for r in self._results:
            if r.result_id == result_id:
                return r
        return None

    def clear(self):
        self._results.clear()
        self._notify()

    def _notify(self):
        for cb in self._callbacks:
            try:
                cb(self._results)
            except Exception:
                pass

    def add_callback(self, cb):
        self._callbacks.append(cb)

    def save_to_json(self, path: str):
        data = [r.to_dict() for r in self._results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._results = [AnalysisResult.from_dict(d) for d in data]
        self._notify()
