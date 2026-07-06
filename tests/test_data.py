import pandas as pd

from src.data.tuab import load_metadata as load_tuab_metadata
from src.data.tuev import label_for_window, load_annotations
from src.data.utils import parse_channels


def test_parse_channels():
    raw = "['FP1', 'FP2', 'C3']"
    assert parse_channels(raw) == ["FP1", "FP2", "C3"]


def test_load_tuab_metadata_filters_by_label_and_duration(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame({
        "datalakeID": ["a", "b", "c"],
        "label": ["normal", "abnormal", "seizure"],  # "seizure" isn't a TUAB label -> dropped
        "subset": ["train", "train", "train"],
        "channels": ["['C3']", "['C3']", "['C3']"],
        "s3_data_file": ["a.parquet", "b.parquet", "c.parquet"],
        "segment_duration_sec": [300.0, 10.0, 300.0],  # b is shorter than WINDOW_SEC -> dropped
    }).to_csv(csv_path, index=False)

    df = load_tuab_metadata(str(csv_path))
    assert list(df["datalakeID"]) == ["a"]


def test_tuev_label_for_window_local_parquet(tmp_path):
    annot_path = tmp_path / "annots.parquet"
    pd.DataFrame({"start": [0.0, 5.0], "stop": [5.0, 10.0], "label": ["bckg", "artf"]}).to_parquet(annot_path)

    annots = load_annotations(str(annot_path))
    assert label_for_window(annots, 0.0, 5.0) == "bckg"
    assert label_for_window(annots, 4.0, 6.0) is None  # straddles the boundary at t=5
