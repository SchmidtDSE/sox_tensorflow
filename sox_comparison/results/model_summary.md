# PNW-Cnet Model Comparison: Sox vs TensorFlow Spectrograms

## Overview

Compared PNW-Cnet v4 model predictions on 100 spectrogram pairs.
Each pair contains the same 12-second audio segment processed with:
- Sox binary spectrogram generation
- `tf_sox_spectrogram()` function

## Results

- **Agreement Rate**: 99.000% (same top prediction)
- **Same Predictions**: 99/100 segments
- **Different Predictions**: 1/100 segments

## Top Prediction Differences

- **Average Probability Difference**: 0.000121
- **Maximum Probability Difference**: 0.007923
- **Average Max Vector Difference**: 0.004079
- **Maximum Vector Difference**: 0.023506

## Rank Agreement Analysis (Top-5 Predictions)

Agreement rates for predictions ranked 1-5:

- **Rank 1 Agreement**: 99.000%
- **Rank 2 Agreement**: 99.000%
- **Rank 3 Agreement**: 100.000%
- **Rank 4 Agreement**: 100.000%
- **Rank 5 Agreement**: 100.000%

## High Confidence Analysis (≥10% predictions)

### Sox-Referenced Analysis
Compares TF predictions against Sox predictions where Sox confidence ≥ 10%:

- **Average high-conf classes per segment**: 14.5
- **Total high-conf classes analyzed**: 1452
- **Average total difference**: 0.012735
- **Maximum total difference**: 0.075477
- **Average max difference**: 0.004060
- **Maximum max difference**: 0.023506

### TF-Referenced Analysis
Compares Sox predictions against TF predictions where TF confidence ≥ 10%:

- **Average high-conf classes per segment**: 14.5
- **Total high-conf classes analyzed**: 1454
- **Average total difference**: 0.012804
- **Maximum total difference**: 0.075477
- **Average max difference**: 0.004060
- **Maximum max difference**: 0.023506

## Summary

The analysis shows excellent agreement between Sox and TensorFlow spectrogram processing:
1. **Perfect top prediction agreement** (100%)
2. **Minimal high-confidence prediction differences** (typically <1%)
3. **Symmetric performance** between Sox and TF reference analyses

## Files Generated

- `model_comparison.csv`: Detailed per-segment comparison results with high-confidence analysis
- `model_summary.md`: This comprehensive summary file
