"""Shared I/O helpers for reading TUAB/BCI2a/TUEV segment parquets, local or on S3."""

import ast

import numpy as np
import pyarrow.parquet as pq

_S3FS = None


def get_s3fs():
    global _S3FS
    if _S3FS is None:
        import s3fs
        _S3FS = s3fs.S3FileSystem()
    return _S3FS


def read_segment_array(path: str) -> np.ndarray:
    """path: local path or s3:// URI to a parquet file. Returns (T, C) float32 array."""
    if path.startswith("s3://"):
        table = pq.read_table(path, filesystem=get_s3fs())
    else:
        table = pq.read_table(path)
    return table.to_pandas().to_numpy(dtype=np.float32)


def parse_channels(raw) -> list[str]:
    """raw: a stringified Python list, as stored in the metadata CSV's `channels` column."""
    return [str(c) for c in ast.literal_eval(raw)]
