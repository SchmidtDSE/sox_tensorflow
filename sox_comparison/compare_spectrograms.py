#!/usr/bin/env python3
"""
License: BSD 3-Clause

Spectrogram comparison between sox binary and tf_sox_spectrogram.

This script generates N=50 random 12-second samples from each audio file,
creates spectrograms using both sox binary and tf_sox_spectrogram,
and performs detailed pixel-level comparison analysis.
"""

import csv
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from PIL import Image

from sox_tensorflow.processor import spectrogram_from_flac

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
AUDIO_DIR = SCRIPT_DIR.parent.parent.parent / 'audio'

def get_audio_duration(flac_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        flac_path: Path to FLAC audio file

    Returns:
        Duration in seconds
    """
    info = sf.info(flac_path)
    return info.frames / info.samplerate


def generate_random_segments(audio_duration: float, segment_duration: float = 12.0, n_samples: int = 50) -> List[float]:
    """
    Generate random start times for audio segments.

    Args:
        audio_duration: Total duration of audio file in seconds
        segment_duration: Duration of each segment in seconds
        n_samples: Number of random segments to generate

    Returns:
        List of start times in seconds
    """
    max_start = audio_duration - segment_duration
    if max_start <= 0:
        raise ValueError(f"Audio file too short ({audio_duration}s) for {segment_duration}s segments")

    return [random.uniform(0, max_start) for _ in range(n_samples)]


def create_sox_spectrogram(flac_path: str, start_time: float, duration: float, output_path: str) -> None:
    """
    Create spectrogram using sox binary with exact parameters from pnw/functions.r.

    Args:
        flac_path: Path to input FLAC file
        start_time: Start time in seconds
        duration: Duration in seconds
        output_path: Output PNG file path
    """
    cmd = [
        'sox',
        '-V1',
        flac_path,
        '-n',
        'trim', str(start_time), str(duration),
        'remix', '1',
        'rate', '8k',
        'spectrogram',
        '-x', '1000',
        '-y', '257',
        '-z', '90',
        '-m',
        '-r',
        '-o', output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sox command failed: {result.stderr}")


def load_spectrogram_pixels(png_path: str) -> np.ndarray:
    """
    Load PNG spectrogram and convert to pixel values.

    Args:
        png_path: Path to PNG file

    Returns:
        Pixel array (uint8) with shape (height, width)
    """
    img = Image.open(png_path)
    if img.mode == 'P':  # Palette mode
        # Convert to grayscale using palette
        img = img.convert('L')
    return np.array(img, dtype=np.uint8)


def compute_pixel_statistics(sox_pixels: np.ndarray, tf_pixels: np.ndarray) -> Dict[str, float]:
    """
    Compute detailed pixel comparison statistics.

    Args:
        sox_pixels: Sox-generated pixel array
        tf_pixels: TensorFlow-generated pixel array

    Returns:
        Dictionary of statistics
    """
    # Ensure same shape
    assert sox_pixels.shape == tf_pixels.shape, f"Shape mismatch: {sox_pixels.shape} vs {tf_pixels.shape}"

    # Calculate differences
    diff = sox_pixels.astype(np.int16) - tf_pixels.astype(np.int16)
    abs_diff = np.abs(diff)

    # Basic statistics
    stats = {
        'accuracy': np.mean(sox_pixels == tf_pixels) * 100,
        'std_dev': np.std(diff),
        'max_diff': np.max(abs_diff),
        'min_diff': np.min(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'pixels_within_1': np.sum(abs_diff <= 1),
        'pixels_within_2': np.sum(abs_diff <= 2),
        'pixels_within_5': np.sum(abs_diff <= 5),
        'pixels_greater_5': np.sum(abs_diff > 5),
        'total_pixels': sox_pixels.size
    }

    # Convert counts to percentages
    total = stats['total_pixels']
    stats['pct_within_1'] = (stats['pixels_within_1'] / total) * 100
    stats['pct_within_2'] = (stats['pixels_within_2'] / total) * 100
    stats['pct_within_5'] = (stats['pixels_within_5'] / total) * 100
    stats['pct_greater_5'] = (stats['pixels_greater_5'] / total) * 100

    return stats


def compute_brightness_percentile_stats(sox_pixels: np.ndarray, tf_pixels: np.ndarray) -> List[Dict[str, float]]:
    """
    Compute statistics for each 10% brightness percentile.

    Args:
        sox_pixels: Sox-generated pixel array
        tf_pixels: TensorFlow-generated pixel array

    Returns:
        List of statistics for each 10% percentile (darkest to brightest)
    """
    # Flatten arrays for percentile calculation
    sox_flat = sox_pixels.flatten()
    tf_flat = tf_pixels.flatten()

    percentile_stats = []

    for p in range(0, 100, 10):  # 0-10%, 10-20%, ..., 90-100%
        # Calculate percentile thresholds for sox pixels (reference)
        p_low = np.percentile(sox_flat, p)
        p_high = np.percentile(sox_flat, p + 10)

        # Create mask for pixels in this brightness range
        if p == 90:  # Last percentile includes 100%
            mask = (sox_flat >= p_low)
        else:
            mask = (sox_flat >= p_low) & (sox_flat < p_high)

        if np.sum(mask) == 0:
            # Skip if no pixels in this range
            continue

        # Extract pixels in this range
        sox_subset = sox_flat[mask]
        tf_subset = tf_flat[mask]

        # Compute statistics for this subset
        stats = compute_pixel_statistics(
            sox_subset.reshape(-1, 1),
            tf_subset.reshape(-1, 1)
        )
        stats['percentile_range'] = f"{p}-{p+10}%"
        stats['brightness_low'] = p_low
        stats['brightness_high'] = p_high

        percentile_stats.append(stats)

    return percentile_stats


def compare_spectrograms_for_segment(
    flac_path: str,
    start_time: float,
    segment_idx: int,
    base_filename: str,
    output_dir: Path
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Generate and compare spectrograms for a single audio segment.

    Args:
        flac_path: Path to audio file
        start_time: Segment start time in seconds
        segment_idx: Segment index number
        base_filename: Base filename (e.g. "20230522_000000")
        output_dir: Output directory for spectrograms

    Returns:
        Tuple of (overall_stats, percentile_stats)
    """
    duration = 12.0

    # Generate output filenames
    sox_png = output_dir / f"{base_filename}-{segment_idx}.sox.png"
    tf_png = output_dir / f"{base_filename}-{segment_idx}.tf.png"

    # Create sox spectrogram
    create_sox_spectrogram(flac_path, start_time, duration, str(sox_png))

    # Create TensorFlow spectrogram
    spectrogram_from_flac(
        flac_path=flac_path,
        start_time=start_time,
        duration=duration,
        shape=(257, 1000),
        dest=str(tf_png),
        db_range=90
    )

    # Load and compare pixels
    sox_pixels = load_spectrogram_pixels(str(sox_png))
    tf_pixels = load_spectrogram_pixels(str(tf_png))

    # Compute statistics
    overall_stats = compute_pixel_statistics(sox_pixels, tf_pixels)
    percentile_stats = compute_brightness_percentile_stats(sox_pixels, tf_pixels)

    # Add metadata
    overall_stats['audio_file'] = base_filename
    overall_stats['segment_idx'] = segment_idx
    overall_stats['start_time'] = start_time

    return overall_stats, percentile_stats


def main():
    """Main comparison function."""
    random.seed(42)  # For reproducible results
    audio_files = [
        AUDIO_DIR / "20230522_000000.flac",
        AUDIO_DIR / "20230526_000000.flac"
    ]

    # Create spectrograms output directory
    output_dir = SCRIPT_DIR / "spectrograms"
    output_dir.mkdir(exist_ok=True)
    # Create results directory relative to script location
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Results storage
    all_overall_stats = []
    all_percentile_stats = []

    for flac_path in audio_files:
        print(f"Processing {flac_path}...")

        # Get base filename
        base_filename = Path(flac_path).stem

        # Get audio duration and generate random segments
        duration = get_audio_duration(flac_path)
        print(f"Audio duration: {duration:.1f}s")

        start_times = generate_random_segments(duration, n_samples=50)

        # Process each segment
        for i, start_time in enumerate(start_times):
            print(f"  Segment {i+1}/50: {start_time:.3f}s")

            try:
                overall_stats, percentile_stats = compare_spectrograms_for_segment(
                    flac_path, start_time, i+1, base_filename, output_dir
                )

                all_overall_stats.append(overall_stats)
                all_percentile_stats.extend(percentile_stats)

            except Exception as e:
                print(f"    Error processing segment {i+1}: {e}")
                continue

    # Write overall results CSV
    overall_csv = results_dir / "overall_comparison.csv"
    with open(overall_csv, 'w', newline='') as f:
        if all_overall_stats:
            writer = csv.DictWriter(f, fieldnames=all_overall_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_overall_stats)

    # Write percentile results CSV
    percentile_csv = results_dir / "percentile_comparison.csv"
    with open(percentile_csv, 'w', newline='') as f:
        if all_percentile_stats:
            writer = csv.DictWriter(f, fieldnames=all_percentile_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_percentile_stats)

    # Generate summary statistics
    if all_overall_stats:
        overall_accuracy = np.mean([s['accuracy'] for s in all_overall_stats])
        overall_std = np.mean([s['std_dev'] for s in all_overall_stats])
        overall_max_diff = np.max([s['max_diff'] for s in all_overall_stats])
        overall_within_5 = np.mean([s['pct_within_5'] for s in all_overall_stats])

        print(f"\nOverall Results:")
        print(f"  Average accuracy: {overall_accuracy:.3f}%")
        print(f"  Average std dev: {overall_std:.3f}")
        print(f"  Maximum difference: {overall_max_diff}")
        print(f"  Pixels within ±5: {overall_within_5:.3f}%")

        # Write summary markdown
        summary_md = results_dir / "summary.md"
        with open(summary_md, 'w') as f:
            f.write("# Sox vs TensorFlow Spectrogram Comparison\n\n")
            f.write("## Overview\n\n")
            f.write(f"Compared {len(all_overall_stats)} 12-second audio segments from 2 FLAC files.\n")
            f.write("Generated spectrograms using both sox binary and tf_sox_spectrogram.\n\n")
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Average Accuracy**: {overall_accuracy:.3f}% (pixels exactly matching)\n")
            f.write(f"- **Average Standard Deviation**: {overall_std:.3f}\n")
            f.write(f"- **Maximum Difference**: {overall_max_diff} pixel values\n")
            f.write(f"- **Pixels within ±5**: {overall_within_5:.3f}%\n\n")
            f.write("## Files Generated\n\n")
            f.write("- `overall_comparison.csv`: Per-segment comparison statistics\n")
            f.write("- `percentile_comparison.csv`: Statistics by brightness percentile\n")
            f.write("- `spectrograms/`: PNG files for visual comparison\n")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()