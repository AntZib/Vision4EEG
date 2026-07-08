import mne
import numpy as np
import pandas as pd

from src.data.bci2a import FS, TRIAL_SEC, extract_trials
from src.data.tuab import load_metadata as load_tuab_metadata
from src.data.tuev import label_for_window, load_annotations
from src.data.utils import parse_channels


def make_raw_with_events(descriptions, onsets_sec, n_channels=4, duration_sec=20.0, fs=FS):
    rng = np.random.default_rng(0)
    n_samples = int(duration_sec * fs)
    data = rng.normal(size=(n_channels, n_samples)).astype(np.float64)
    info = mne.create_info([f"ch{i}" for i in range(n_channels)], sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=onsets_sec, duration=[0.0] * len(onsets_sec), description=descriptions))
    return raw


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


def test_bci2a_extract_trials_train_session():
    # 3 cues (left, right, tongue) at t=2, 8, 14s -- codes per src/data/bci2a.CUE_EVENT_TO_CLASS
    raw = make_raw_with_events(["769", "770", "772"], [2.0, 8.0, 14.0])
    trials = extract_trials(raw, is_eval=False)

    assert [label for label, _ in trials] == [0, 1, 3]  # left_hand, right_hand, tongue
    for _, window in trials:
        assert window.shape == (int(TRIAL_SEC * FS), 4)


def test_bci2a_extract_trials_eval_session_uses_true_labels_in_order():
    raw = make_raw_with_events(["783", "783", "783"], [2.0, 8.0, 14.0])
    true_labels = np.array([2, 0, 1])  # feet, left_hand, right_hand -- already 0-indexed
    trials = extract_trials(raw, is_eval=True, true_labels=true_labels)

    assert [label for label, _ in trials] == [2, 0, 1]


def test_bci2a_extract_trials_drops_trial_past_end_of_recording():
    raw = make_raw_with_events(["769"], [19.0], duration_sec=20.0)  # only 1s left, needs TRIAL_SEC=4s
    trials = extract_trials(raw, is_eval=False)
    assert trials == []
