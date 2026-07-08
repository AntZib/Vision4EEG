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
(public, no registration) -- `BCICIV_2a_gdf.zip`, one continuous GDF file per
subject per session: `A0{1-9}T.gdf` (training) and `A0{1-9}E.gdf` (evaluation).

Evaluation sessions carry no class labels themselves (the cue annotation is
"783", unknown) -- the true labels were released after the competition as a
separate per-session `A0{id}E.mat` file (field `classlabel`). If you only have
the training sessions, that's fine: `--labels-dir` is only needed for `*E.gdf`.

```
scripts/preprocess_raw.py --dataset bci2a \
    --raw-dir data/raw/bci2a --labels-dir data/raw/bci2a_true_labels --out-dir data/processed/bci2a
```

- `--raw-dir`: the `A0{1-9}T.gdf` / `A0{1-9}E.gdf` files, unmodified.
- `--labels-dir`: `A0{id}E.mat` files (one per evaluation session), each with a
  `classlabel` array (1-4, in trial order).

Trials are cut as the 4s motor-imagery window starting at cue onset (t=2s to
t=6s relative to trial start -- see the [dataset description PDF](http://bbci.de/competition/iv/desc_2a.pdf)).
The 3 EOG channels are dropped (provided for artifact-correction methods only,
not meant for classification). The original recording is already
bandpass-filtered 0.5-100Hz with a 50Hz notch (European mains) by the
amplifier -- `scripts/preprocess_raw.py` still re-applies its own 50Hz notch/
high-pass/CAR/resample-to-128Hz on top, harmlessly idempotent-ish, for a
consistent pipeline across all three datasets.

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
