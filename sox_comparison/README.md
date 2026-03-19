# sox_comparison

Tools for verifying that `sox_tensorflow` produces spectrograms that match the
sox binary, and that those spectrograms yield equivalent PNW-Cnet model
predictions.

---

## Scripts

| File | Description |
|---|---|
| `compare_spectrograms_cli.py` | Generates spectrogram pairs (sox + sox_tensorflow) and compares them pixel-by-pixel |
| `model_comparison_cli.py` | Runs PNW-Cnet on existing spectrogram pairs and compares model predictions |
| `analysis.ipynb` | Notebook that reads the CSVs and visualises the results |
| `compare_spectrograms.py` | Original (no CLI) version of the spectrogram comparison |
| `model_comparison.py` | Original (no CLI) version of the model comparison |

---

## Quickstart

### 1. Generate and compare spectrograms

```bash
python compare_spectrograms_cli.py <audio>
```

`<audio>` can be:
- a single audio file: `audio/20230522_000000.flac`
- a directory: `audio/`
- an S3 URI (file or prefix): `s3://my-bucket/audio/`
- multiple inputs: `audio/file1.flac audio/file2.flac`

Supported formats: FLAC, WAV, MP3, OGG.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--output`, `-o` | `./results` | Output directory for CSVs and spectrograms |
| `--n-samples`, `-n` | 50 | Random 12-second segments to sample per file |
| `--seed` | 42 | Random seed for reproducible segment selection |

**Output files** (written to `--output`):

| File | Description |
|---|---|
| `results/overall_comparison.csv` | Per-segment pixel statistics |
| `results/percentile_comparison.csv` | Per-segment statistics broken down by brightness decile |
| `spectrograms/*.sox.png` | Spectrograms from the sox binary |
| `spectrograms/*.tf.png` | Spectrograms from sox_tensorflow |

### 2. Compare model predictions

```bash
python model_comparison_cli.py
```

Reads the spectrogram pairs already in `spectrograms/` and runs PNW-Cnet on
each pair to measure prediction agreement.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `pnw/PNW-Cnet_v4_TF.h5` | Path to `PNW-Cnet_v4_TF.h5` |
| `--classes`, `-c` | `pnw/target_classes.csv` | Path to `target_classes.csv` |
| `--spectrograms`, `-s` | `./spectrograms` | Directory of `.sox.png` / `.tf.png` pairs |
| `--output`, `-o` | `./results` | Output directory for CSVs |

**Output files:**

| File | Description |
|---|---|
| `results/model_comparison.csv` | Per-pair prediction comparison (top-1 agreement, rank 1–5, probability diffs, high-confidence analysis) |

### 3. Analyse results

Open `analysis.ipynb` and run all cells. Set `RESULTS_DIR` in the first cell
if you used a custom `--output` path.

The notebook covers:
- Overall pixel difference distributions
- Agreement by brightness decile (are errors concentrated in dark or bright pixels?)
- Model top-1 and rank 1–5 agreement rates
- Prediction probability difference distributions
- Sox vs sox_tensorflow confidence scatter plot

---

## S3 support

S3 input requires `boto3`:

```bash
pip install boto3
```

Credentials are read from the standard AWS credential chain (env vars, `~/.aws/credentials`, IAM role, etc.).

---

## Example — full run

```bash
# 1. Compare spectrograms for two files (20 segments each)
python compare_spectrograms_cli.py \
  audio/20230522_000000.flac audio/20230526_000000.flac \
  --n-samples 20 \
  --output run_01

# 2. Compare model predictions (model and classes default to pnw/)
python model_comparison_cli.py \
  --spectrograms run_01/spectrograms \
  --output run_01/results

# 3. Open notebook
jupyter lab analysis.ipynb
```
