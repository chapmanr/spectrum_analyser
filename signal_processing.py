"""Signal processing functions for spectrum analysis.

This module contains core signal processing functions including:
- Signal loading from various file formats
- FFT computation with windowing and zero-padding
- Peak detection
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy.signal import savgol_filter


def load_signal(file_path: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load signal from file.
    
    Args:
        file_path: Path to the signal file (.raw, .npy, .wav)
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (signal data as numpy array, sample rate)
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.npy':
        data = np.load(file_path)
        return np.asarray(data, dtype=np.float32), sample_rate

    if ext == '.raw':
        # raw float32 little-endian
        data = np.fromfile(file_path, dtype='<f4')
        return data.astype(np.float32), sample_rate

    # try for wav/other audio formats
    try:
        import soundfile as sf
        data, sr = sf.read(file_path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception as e:
        print(f"Error loading file: {e}")
        return np.array([]), sample_rate


def compute_fft(
    data: np.ndarray,
    sample_rate: int,
    window_samples: Optional[int] = None,
    nfft: Optional[int] = None,
    zp_factor: float = 1.0,
    use_db: bool = True,
    window_type: str = 'hanning',
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT with windowing and zero-padding.
    
    Args:
        data: Input signal data
        sample_rate: Sample rate in Hz
        window_samples: Number of samples to use (None = all)
        nfft: FFT size (None = auto based on zp_factor)
        zp_factor: Zero-padding factor
        use_db: Convert magnitude to dB scale
        window_type: Window function type
        
    Returns:
        Tuple of (frequencies, magnitudes)
    """
    # choose segment length
    N = data.size if window_samples is None else min(window_samples, data.size)
    if N <= 0:
        return np.array([]), np.array([])

    seg = data[:N].astype(np.float64)

    # apply window function
    if window_type == 'hanning':
        win = np.hanning(N)
    elif window_type == 'hamming':
        win = np.hamming(N)
    elif window_type == 'blackman':
        win = np.blackman(N)
    elif window_type == 'bartlett':
        win = np.bartlett(N)
    elif window_type == 'rectangular':
        win = np.ones(N)
    elif window_type == 'kaiser':
        win = np.kaiser(N, beta=8.6)
    elif window_type == 'gaussian':
        try:
            from scipy.signal.windows import gaussian
            win = gaussian(N, std=N/6)
        except ImportError:
            win = np.hanning(N)
    elif window_type == 'tukey':
        try:
            from scipy.signal import tukey
            win = tukey(N, alpha=0.5)
        except ImportError:
            win = np.hanning(N)
    elif window_type == 'flattop':
        try:
            from scipy.signal import flattop
            win = flattop(N)
        except ImportError:
            win = np.hanning(N)
    else:
        win = np.hanning(N)
    
    seg_win = seg * win

    # determine nfft
    if nfft is None:
        target = int(max(1, int(np.ceil(N * float(zp_factor)))))
        nfft = 1 << (int(np.ceil(np.log2(target))) if target > 0 else 0)
    nfft = max(1, int(nfft))

    Y = np.fft.rfft(seg_win, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    mag = np.abs(Y)
    
    if use_db:
        eps = 1e-20
        mag = 20.0 * np.log10(mag + eps)
    
    return freqs, mag


def detect_peaks(y: np.ndarray, threshold: float = 10.0, min_distance: int = 5, use_db: bool = True) -> np.ndarray:
    """Simple peak detection: find local maxima that are threshold above neighbors.
    
    Args:
        y: Input magnitude array
        threshold: Threshold above mean for peak detection
        min_distance: Minimum distance between peaks in samples
        use_db: Whether data is in dB scale
        
    Returns:
        Array of peak indices
    """
    if y.size < 3:
        return np.array([], dtype=int)
    peaks = []
    for i in range(1, y.size - 1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            if use_db:
                if y[i] > (np.mean(y) + threshold):
                    peaks.append(i)
            else:
                if y[i] > (np.mean(y) * threshold):
                    peaks.append(i)
    if not peaks:
        return np.array([], dtype=int)
    peaks = np.array(peaks, dtype=int)
    if min_distance > 1 and peaks.size > 1:
        filtered = [peaks[0]]
        for pk in peaks[1:]:
            if pk - filtered[-1] >= min_distance:
                filtered.append(pk)
        peaks = np.array(filtered, dtype=int)
    return peaks


def generate_sine_with_noise(
    freq: float,
    duration: float,
    sample_rate: int,
    amplitude: float,
    noise_level: float,
    harmonics: int = 0,
    harmonic_amps: Optional[list] = None,
    sine_stop_frac: float = 0.85,
    num_segments: int = 1,
    freq_drift: float = 0.0,
) -> np.ndarray:
    """Generate a primary sine plus optional integer harmonics.

    Args:
        freq: Base frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Peak amplitude of primary sine
        noise_level: Standard deviation of additive Gaussian noise
        harmonics: Number of integer harmonics to add (1 => 2nd harmonic, etc.)
        harmonic_amps: List of absolute amplitudes for each harmonic (if fewer than harmonics, missing default to halving)
        sine_stop_frac: Fraction of signal after which sine stops (remaining is noise only)
        num_segments: Divide signal into equal parts with different frequencies
        freq_drift: Frequency increase (in Hz) per segment
        
    Returns:
        Generated signal as numpy array
    """
    import math
    
    n_samples = int(math.floor(sample_rate * duration))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    
    # create mask to zero the sine (and harmonics) after sine_stop_frac of samples
    if sine_stop_frac <= 0.0:
        # no sine at all
        mask = np.zeros(n_samples)
    elif sine_stop_frac >= 1.0:
        mask = np.ones(n_samples)
    else:
        stop_sample = int(math.floor(n_samples * float(sine_stop_frac)))
        mask = np.ones(n_samples, dtype=np.float64)
        if stop_sample < n_samples:
            mask[stop_sample:] = 0.0

    # Initialize data array
    data = np.zeros(n_samples, dtype=np.float64)
    
    # Process each segment with frequency drift
    if num_segments <= 1:
        # No segmentation, use original behavior
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        data = amplitude * (mask * np.sin(2.0 * math.pi * freq * t))
        
        if harmonics and harmonics > 0:
            provided = list(harmonic_amps) if harmonic_amps else []
            for i in range(1, harmonics + 1):
                h_freq = freq * (i + 1)
                if i - 1 < len(provided):
                    h_amp = float(provided[i - 1])
                else:
                    h_amp = amplitude * (0.5 ** i)
                data = data + h_amp * (mask * np.sin(2.0 * math.pi * h_freq * t))
    else:
        # Segment the signal with frequency drift
        samples_per_segment = n_samples // num_segments
        
        for seg_idx in range(num_segments):
            start_idx = seg_idx * samples_per_segment
            if seg_idx == num_segments - 1:
                # Last segment gets any remaining samples
                end_idx = n_samples
            else:
                end_idx = (seg_idx + 1) * samples_per_segment
            
            seg_len = end_idx - start_idx
            # Time array for this segment starts from 0
            t_seg = np.arange(seg_len, dtype=np.float64) / sample_rate
            
            # Calculate frequency for this segment
            current_freq = freq + seg_idx * freq_drift
            
            # Generate sine for this segment
            seg_mask = mask[start_idx:end_idx]
            seg_data = amplitude * (seg_mask * np.sin(2.0 * math.pi * current_freq * t_seg))
            
            # Add harmonics for this segment
            if harmonics and harmonics > 0:
                provided = list(harmonic_amps) if harmonic_amps else []
                for i in range(1, harmonics + 1):
                    h_freq = current_freq * (i + 1)
                    if i - 1 < len(provided):
                        h_amp = float(provided[i - 1])
                    else:
                        h_amp = amplitude * (0.5 ** i)
                    seg_data = seg_data + h_amp * (seg_mask * np.sin(2.0 * math.pi * h_freq * t_seg))
            
            data[start_idx:end_idx] = seg_data

    noise = noise_level * np.random.randn(n_samples)
    data = data + noise
    return data.astype(np.float32)
