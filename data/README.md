# Data

| Path | In git? | Contents |
|---|---|---|
| `sample/` | Yes (~700 KB) | A small HDFS_v1 sample from [loghub](https://github.com/logpai/loghub) — enough to run the pipeline end-to-end locally without downloading the full dataset. |
| `explainer_dataset/` | Yes (~2 MB) | The curated JSONL chat-format dataset used to fine-tune the explainer model (`logsense_final_train/val.jsonl`), plus the intermediate files it was built from. See `notebooks/01_dataset_preparation.ipynb`. |
| `raw/` | **No** (gitignored, 1.8 GB) | The full HDFS log (11M+ lines) and its preprocessed Loghub artifacts (`Event_traces.csv`, `anomaly_label.csv`, `HDFS.npz`, etc.) used to train the classifier and generate the explainer dataset. |

## Getting `data/raw/`

Download the full HDFS_v1 dataset (raw log + preprocessed CSVs) from
[loghub](https://github.com/logpai/loghub/tree/master/HDFS) or
[Zenodo](https://zenodo.org/records/8196385), and place it at:

```
data/raw/HDFS.log
data/raw/preprocessed/Event_traces.csv
data/raw/preprocessed/anomaly_label.csv
data/raw/preprocessed/Event_occurrence_matrix.csv
data/raw/preprocessed/HDFS.npz
data/raw/preprocessed/HDFS.log_templates.csv
```

For most purposes (demoing the API, running the frontend, testing the
explainer) the small dataset in `sample/` is sufficient — you only need
`raw/` to retrain the classifier or regenerate the explainer dataset from
scratch.
