"""Shared I/O helpers for reading TUAB/BCI2a/TUEV segment parquets."""

import ast

import numpy as np
import pyarrow.parquet as pq


def read_segment_array(path: str) -> np.ndarray:
    """path: local path to a parquet file. Returns (T, C) float32 array."""
    table = pq.read_table(path)
    return table.to_pandas().to_numpy(dtype=np.float32)


def parse_channels(raw) -> list[str]:
    """raw: a stringified Python list, as stored in the metadata CSV's `channels` column."""
    return [str(c) for c in ast.literal_eval(raw)]
