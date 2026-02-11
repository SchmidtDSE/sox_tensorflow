# Sox vs TensorFlow Spectrogram Comparison

## Overview

Compared 100 12-second audio segments from 2 FLAC files.
Generated spectrograms using both sox binary and tf_sox_spectrogram.

## Overall Statistics

- **Average Accuracy**: 99.807% (pixels exactly matching)
- **Average Standard Deviation**: 0.045
- **Maximum Difference**: 2 pixel values
- **Pixels within ±5**: 100.000%

## Files Generated

- `overall_comparison.csv`: Per-segment comparison statistics
- `percentile_comparison.csv`: Statistics by brightness percentile
- `spectrograms/`: PNG files for visual comparison
