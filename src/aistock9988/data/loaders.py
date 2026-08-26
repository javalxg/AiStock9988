from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .pit import enforce_available_time


@dataclass(frozen=True)
class LoadRequest:
    required_columns: tuple[str, ...]
    decision_time: pd.Timestamp | None = None
    available_column: str = "available_time"
    event_column: str = "event_time"


class DataLoader:
    def load(self, request: LoadRequest) -> pd.DataFrame:  # pragma: no cover - protocol-like base
        raise NotImplementedError


class FrameLoader(DataLoader):
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame.copy()

    def load(self, request: LoadRequest) -> pd.DataFrame:
        frame = self._frame.copy()
        _validate(frame, request)
        if request.decision_time is not None:
            frame = enforce_available_time(frame, decision_time=request.decision_time.to_pydatetime(),
                                           available_column=request.available_column)
        return _stable(frame)


class FileLoader(DataLoader):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, request: LoadRequest) -> pd.DataFrame:
        if self.path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(self.path)
        elif self.path.suffix.lower() in {".csv", ".txt"}:
            frame = pd.read_csv(self.path)
        else:
            raise ValueError(f"unsupported data file: {self.path.suffix}")
        return FrameLoader(frame).load(request)


class SQLLoader(DataLoader):
    """Read-only SQL adapter. The caller owns the connection and transaction."""

    def __init__(self, connection, table: str, *, where_sql: str = "", params: Sequence[object] = ()):
        if not table.replace("_", "").isalnum():
            raise ValueError("table name must be a simple identifier")
        self.connection = connection
        self.table = table
        self.where_sql = where_sql
        self.params = tuple(params)

    def load(self, request: LoadRequest) -> pd.DataFrame:
        query = f"SELECT * FROM {self.table}"
        if self.where_sql:
            query += f" WHERE {self.where_sql}"
        frame = pd.read_sql_query(query, self.connection, params=self.params)
        return FrameLoader(frame).load(request)


def _validate(frame: pd.DataFrame, request: LoadRequest) -> None:
    missing = [c for c in request.required_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame[request.event_column].isna().any():
        raise ValueError(f"null {request.event_column} is not allowed")
    if request.available_column not in frame.columns:
        raise ValueError(f"missing PIT column: {request.available_column}")


def _stable(frame: pd.DataFrame) -> pd.DataFrame:
    # Stable order is part of the training contract, especially for pairwise ranking.
    keys = [c for c in ("event_time", "ts_code", "trade_date") if c in frame.columns]
    return frame.sort_values(keys or list(frame.columns), kind="mergesort").reset_index(drop=True)
