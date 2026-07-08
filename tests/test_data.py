"""
Tests for the per-dataset metadata loading and raw ingestion helpers.

TUAB and TUEV tests build small synthetic metadata/annotation files on disk
(tmp_path). BCI2a tests build a synthetic mne Raw object with fake
annotations, so none of this needs a real GDF/EDF file.
"""

import mne
import numpy as np
import pandas as pd

from src.data.bci2a import FS, TRIAL_SEC, extract_trials
from src.data.tuab import load_metadata as load_tuab_metadata
from src.data.tuev import PAD_SEC, extract_events, load_annotations
from src.data.utils import parse_channels


def make_raw_with_events(descriptions, onsets_sec, n_channels=4, duration_sec=20.0, fs=FS):
    """
    Build a fake continuous recording with annotations at given times,
    standing in for a real GDF session.
    """

    rng = np.random.default_rng(0)
    n_samples = int(duration_sec * fs)
    data = rng.normal(size=(n_channels, n_samples)).astype(np.float64)

    info = mne.create_info(
        [f"ch{i}" for i in range(n_channels)],
        sfreq=fs,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    raw.set_annotations(
        mne.Annotations(
            onset=onsets_sec,
            duration=[0.0] * len(onsets_sec),
            description=descriptions,
        )
    )

    return raw


def test_parse_channels():
    raw = "['FP1', 'FP2', 'C3']"
    assert parse_channels(raw) == ["FP1", "FP2", "C3"]


def test_load_tuab_metadata_filters_by_label_and_duration(tmp_path):
    csv_path = tmp_path / "metadata.csv"

    pd.DataFrame({
        "record_id": ["a", "b", "c"],
        "label": ["normal", "abnormal", "seizure"],
        "subset": ["train", "train", "train"],
        "channels": ["['C3']", "['C3']", "['C3']"],
        "data_file": ["a.parquet", "b.parquet", "c.parquet"],
        "segment_duration_sec": [300.0, 10.0, 300.0],
    }).to_csv(csv_path, index=False)

    df = load_tuab_metadata(str(csv_path))

    # "b" is dropped because its segment is shorter than WINDOW_SEC.
    # "c" is dropped because "seizure" isn't a valid TUAB label.
    assert list(df["record_id"]) == ["a"]


def test_tuev_load_annotations_parses_headerless_rec_csv(tmp_path):
    rec_path = tmp_path / "rec001.rec"

    # columns: channel, start_sec, end_sec, label_code
    rec_path.write_text("0,10.0,11.0,6\n3,50.0,51.0,1\n")

    annots = load_annotations(str(rec_path))

    assert list(annots["label_code"]) == [6.0, 1.0]
    assert list(annots["start"]) == [10.0, 50.0]


def test_tuev_extract_events_cuts_padded_window_around_each_event():
    fs = 128.0
    signal = np.zeros((int(20 * fs), 4), dtype=np.float32)

    # one bckg (code 6) event at [10, 11]s, one spsw (code 1) event at [15, 16]s
    annots = pd.DataFrame({
        "channel": [0, 3],
        "start": [10.0, 15.0],
        "stop": [11.0, 16.0],
        "label_code": [6, 1],
    })

    events = extract_events(signal, fs, annots)

    assert [label for label, _ in events] == ["bckg", "spsw"]

    # 1-second event plus PAD_SEC of padding on each side.
    expected_len = int(round((1.0 + 2 * PAD_SEC) * fs))
    for _, window in events:
        assert window.shape == (expected_len, 4)


def test_tuev_extract_events_drops_and_caps():
    fs = 128.0
    signal = np.zeros((int(20 * fs), 4), dtype=np.float32)

    annots = pd.DataFrame({
        "channel": [0, 0, 0, 0],
        "start": [0.5, 5.0, 8.0, 19.5],
        "stop": [1.5, 6.0, 9.0, 19.6],
        "label_code": [6, 6, 6, 6],
    })

    # The first and last events are too close to the recording's edges and
    # are never candidates. Of the two remaining bckg events, only one
    # survives the max_bckg=1 cap.
    events = extract_events(signal, fs, annots, max_bckg=1)

    assert len(events) == 1


def test_bci2a_extract_trials_train_session():
    # 3 cues (left, right, tongue) at t=2, 8, 14s. Codes match
    # src/data/bci2a.CUE_EVENT_TO_CLASS.
    raw = make_raw_with_events(["769", "770", "772"], [2.0, 8.0, 14.0])

    trials = extract_trials(raw, is_eval=False)

    assert [label for label, _ in trials] == [0, 1, 3]
    for _, window in trials:
        assert window.shape == (int(TRIAL_SEC * FS), 4)


def test_bci2a_extract_trials_eval_session_uses_true_labels_in_order():
    raw = make_raw_with_events(["783", "783", "783"], [2.0, 8.0, 14.0])
    true_labels = np.array([2, 0, 1])  # feet, left_hand, right_hand, already 0-indexed

    trials = extract_trials(raw, is_eval=True, true_labels=true_labels)

    assert [label for label, _ in trials] == [2, 0, 1]


def test_bci2a_extract_trials_drops_trial_past_end_of_recording():
    # Only 1 second of signal left after the cue, but a trial needs TRIAL_SEC=4s.
    raw = make_raw_with_events(["769"], [19.0], duration_sec=20.0)

    trials = extract_trials(raw, is_eval=False)

    assert trials == []
