# Datasets

No raw or processed data ships with this repo (TUAB/TUEV can't legally be
redistributed; nothing is committed here at all, see `.gitignore`). This documents
where to get each dataset and the local layout `scripts/preprocess_raw.py` expects.

## TUAB -- normal/abnormal (binary)

Source: [TUH Abnormal EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)
(free registration + data use agreement, then SFTP access).

```
scripts/preprocess_raw.py --dataset tuab \
    --raw-dir data/raw/tuab --labels-csv data/raw/tuab_labels.csv --out-dir data/processed/tuab
```

- `--raw-dir`: one continuous EDF file per recording, named `{record_id}.edf`.
- `--labels-csv`: columns `record_id, label, subset` (`label` in `{normal, abnormal}`,
  `subset` in `{train, validation, test}`) -- this isn't encoded in the EDF itself,
  it comes from TUAB's own train/eval split and per-recording labels.

## BCI2a -- 4-class motor imagery

Source: [BCI Competition IV, dataset 2a](https://www.bbci.de/competition/iv/#dataset2a)
(public, no registration).

```
scripts/preprocess_raw.py --dataset bci2a --raw-dir data/raw/bci2a --out-dir data/processed/bci2a
```

- `--raw-dir`: one EDF file per trial (already 5s-long), with the trial's class
  name (`left_hand`, `right_hand`, `feet`, or `tongue`) in the EDF header's
  `subject_info.his_id` field, and `train`/`test` in the filename
  (e.g. `..._session-0train_...`). If your copy of the dataset ships the original
  `.gdf` format instead, convert each trial to EDF with this same convention first
  (e.g. via `mne.io.read_raw_gdf` + `mne.export.export_raw`).

## TUEV -- 6-class event classification

Source: [TUH EEG Events Corpus](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)
(free registration + data use agreement, same access as TUAB).

```
scripts/preprocess_raw.py --dataset tuev \
    --raw-dir data/raw/tuev/edf --annotations-dir data/raw/tuev/annots \
    --labels-csv data/raw/tuev_labels.csv --out-dir data/processed/tuev
```

- `--raw-dir`: one continuous EDF file per recording, named `{record_id}.edf`.
- `--annotations-dir`: one parquet per recording, named `{record_id}.parquet`, with
  columns `(start, stop, label)` in absolute recording-time seconds, `label` in
  `{bckg, artf, eyem, spsw, gped, pled}` -- convert TUEV's native `.rec` annotation
  files to this format.
- `--labels-csv`: columns `record_id, subset` (TUEV has no whole-recording label,
  only a train/test split to carry over).

## Local samples used during development

Small real samples of all three datasets (kept outside this repo, never committed --
TUAB/TUEV's data use agreement doesn't permit redistribution) were used to validate
`src/data/preprocessing.py` and the ingestion functions in `src/data/{tuab,bci2a,tuev}.py`
end to end before writing the pipeline documented above.
