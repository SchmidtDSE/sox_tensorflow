"""
Unified Audio Processing Interface

Routes to backend-specific implementations (TensorFlow, PyTorch, or librosa)

License: BSD 3-Clause
"""

from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict, Any
import pandas as pd
from cocina.config_handler import ConfigHandler

ch = ConfigHandler(__file__)

# Export key classes
def load_audio(
    src: Union[str, Path],
    sample_rate: Optional[int] = None,
    file_type: Optional[str] = None,
    backbone: Optional[str] = None
):
    """
    Load audio file.

    Args:
        src: Path to audio file (FLAC, WAV, etc.)
        sample_rate: Target sample rate in Hz. If None, uses original sample rate
        file_type: Audio file type. If None, auto-detected from extension
        backbone: Backend to use ('tensorflow', 'pytorch', or 'librosa')

    Returns:
        AudioData containing tensor/array, sample_rate, duration, and channels

    Raises:
        IOError: If file cannot be loaded
        ValueError: If invalid file type, sample rate, or backbone specified
    """
    backend = _get_backend(backbone)
    return backend.load_audio(src, sample_rate, file_type)


def spectrograms(
    src: Union[str, Path],
    dest_folder: str,
    spectrogram_shape: Tuple[int, int],
    basename: Optional[str] = None,
    directory: Optional[str] = None,
    segment_duration: Optional[float] = None,
    segment_overlap: Optional[float] = None,
    sample_rate: Optional[int] = None,
    spectrogram_type: Optional[str] = None,
    create_parents: bool = True,
    overwrite: bool = True,
    backbone: Optional[str] = None,
    output_format: Optional[str] = None,
    tfrecord_compression: Optional[str] = None,
    tfrecord_chunk_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Union[List[str], List[Any]]:
    """
    Generate multiple spectrograms from audio source with segmentation.

    Args:
        src: Audio source - file path
        dest_folder: Folder to save spectrogram files
        spectrogram_shape: Output shape for each spectrogram (height, width)
        basename: Base name for output files (auto-determined if None)
        directory: Directory name within dest_folder (optional)
        segment_duration: Duration of each segment in seconds
        segment_overlap: Overlap between segments in seconds
        sample_rate: Audio sample rate in Hz
        spectrogram_type: Type of spectrogram ('mel', 'sox', or None)
        create_parents: Create parent directories if needed
        overwrite: Overwrite existing files if they exist
        backbone: Backend to use ('tensorflow', 'pytorch', or 'librosa')
        output_format: Output format - 'png' or 'tfrecord' (default from config)
        tfrecord_compression: TFRecord compression - 'GZIP', 'ZLIB', or None
        tfrecord_chunk_size: Split TFRecord into chunks of this size (None = single file)
        metadata: Additional metadata to store in TFRecord (e.g., audio_uri, start_time)

    Returns:
        If output_format='png': List of spectrogram file paths (or tensors/arrays)
        If output_format='tfrecord': List of TFRecord file paths

    Raises:
        ValueError: If invalid parameters or backbone specified
        IOError: If audio loading or file operations fail
        FileExistsError: If files exist and overwrite=False
    """
    backend = _get_backend(backbone)
    from cocina.printer import Printer
    Printer().callout(backend)
    return backend.spectrograms(
        src=src,
        dest_folder=dest_folder,
        spectrogram_shape=spectrogram_shape,
        basename=basename,
        directory=directory,
        segment_duration=segment_duration,
        segment_overlap=segment_overlap,
        sample_rate=sample_rate,
        spectrogram_type=spectrogram_type,
        create_parents=create_parents,
        overwrite=overwrite,
        output_format=output_format,
        tfrecord_compression=tfrecord_compression,
        tfrecord_chunk_size=tfrecord_chunk_size,
        metadata=metadata
    )


def inference(
    model: str,
    files: Union[List[str], pd.DataFrame],
    spectrogram_shape: Optional[Tuple[int, int]] = None,
    confidence_threshold: float = 0.1,
    top_k: int = 5,
    class_names: Optional[List[str]] = None,
    return_format: str = 'dict',
    backbone: Optional[str] = None
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Run model inference on spectrogram files.

    Args:
        model: Path to model file
        files: List of spectrogram file paths or DataFrame with file paths
        spectrogram_shape: Expected spectrogram shape (height, width)
        confidence_threshold: Minimum confidence for predictions
        top_k: Number of top predictions to return per file
        class_names: List of class names (if None, uses indices)
        return_format: Output format ('dict' or 'dataframe')
        backbone: Backend to use ('tensorflow', 'pytorch', or 'librosa')

    Returns:
        Dictionary or DataFrame containing inference results

    Raises:
        FileNotFoundError: If model file not found
        ValueError: If invalid parameters or backbone specified
        IOError: If inference fails
    """
    backend = _get_backend(backbone)
    return backend.inference(
        model=model,
        files=files,
        spectrogram_shape=spectrogram_shape,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
        class_names=class_names,
        return_format=return_format
    )


#
# INTERNAL
#
def _load_backbone_modules():
    backbones = []
    modules = {}
    try:
        from soundhub_utils.audio import pytorch_audio
        backbones.append('pytorch')
        modules['pytorch'] = pytorch_audio
    except ImportError:
        pass
    try:
        from soundhub_utils.audio import librosa_audio
        backbones.append('librosa')
        modules['librosa'] = librosa_audio
    except ImportError:
        pass
    try:
        from soundhub_utils.audio import tensorflow_audio
        backbones.append('tensorflow')
        modules['tensorflow'] = tensorflow_audio
    except ImportError as e:
        pass
    default = ch.get('DEFAULT_AUDIO_BACKBONE', backbones[0]) 
    return backbones, modules, default


def _get_backend(backbone: Optional[str] = None):
    """
    Get the audio backend module.

    Args:
        backbone: Backend to use ('tensorflow', 'pytorch', or 'librosa').
                 If None, uses DEFAULT_BACKBONE.

    Returns:
        Backend module

    Raises:
        ValueError: If backbone is not available
    """
    if backbone is None:
        backbone = DEFAULT_BACKBONE

    if backbone not in AVAILABLE_BACKBONES:
        available_str = ', '.join(AVAILABLE_BACKBONES) if AVAILABLE_BACKBONES else 'none'
        source_map = {
            'tensorflow': 'tensorflow (conda or pip)',
            'pytorch': 'pytorch (conda or pip)',
            'librosa': 'librosa (conda or pip)'
        }
        install_msg = source_map.get(backbone, backbone)
        raise ValueError(
            f"Selected backbone '{backbone}' is not available. "
            f"Available backbones: {available_str}. "
            f"To use '{backbone}', install {install_msg}"
        )

    return BACKBONE_MODULES[backbone]


#
# Detect available backends
#
AVAILABLE_BACKBONES, BACKBONE_MODULES, DEFAULT_BACKBONE = _load_backbone_modules()
