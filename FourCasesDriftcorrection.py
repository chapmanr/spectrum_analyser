"""
DFT Analysis: Signal with Drift, Noise, Interference, Apodization, and Zero-Padding
Converted from Mathematica notebook FourCasesDriftcorrection.nb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Input Parameters
# ============================================================
fs = 1_000_000      # Sampling frequency: 1 MS/s
duration = 1.0      # Total acquisition time in seconds

# Signal Parameters
f0 = 50001.35742    # Main signal frequency in Hz
amp = 1.0           # Main signal amplitude
drift_rate = 10     # Frequency drift in Hz/s
noise_level = 100    # White noise standard deviation

# Window Parameters
use_window = True         # Enable/disable windowing
window_type = "Hann"      # Options: "Hann", "Hamming", "Blackman", "None"

# Zero-Padding Parameters
zero_pad_factor = 8       # Zero-padding factor for spectral interpolation

# Interference Parameters
include_interference = True
f_interference = 50020    # Interference frequency in Hz
amp_interference = 0.003  # Interference amplitude

# ============================================================
# Generate Time Vector
# ============================================================
n_points = round(fs * duration)
t_vec = np.arange(n_points) / fs   # 0 to duration - 1/fs

# ============================================================
# Generate Main Signal with Drift
# ============================================================
phase = 2 * np.pi * (f0 * t_vec + drift_rate * t_vec**2 / 2)
main_signal = amp * np.cos(phase)

# ============================================================
# Generate White Noise
# ============================================================
rng = np.random.default_rng(seed=42)   # seed for reproducibility; remove for random
noise = rng.normal(0, noise_level, n_points)

# ============================================================
# Generate Interference Signal
# ============================================================
if include_interference:
    interference_signal = amp_interference * np.cos(2 * np.pi * f_interference * t_vec)
else:
    interference_signal = np.zeros(n_points)

# ============================================================
# Combined Signal
# ============================================================
signal = main_signal + interference_signal + noise

# ============================================================
# Window Function
# ============================================================
def make_window(n, window_type, use_window):
    if not use_window or window_type == "None":
        return np.ones(n)
    k = np.arange(n)
    if window_type == "Hann":
        return 0.5 * (1 - np.cos(2 * np.pi * k / (n - 1)))
    elif window_type == "Hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))
    elif window_type == "Blackman":
        return (0.42
                - 0.5  * np.cos(2 * np.pi * k / (n - 1))
                + 0.08 * np.cos(4 * np.pi * k / (n - 1)))
    else:
        return np.ones(n)

window = make_window(n_points, window_type, use_window)
window_norm = np.mean(window)   # Coherent gain

# ============================================================
# Segment Window (for Cases 2 and 4)
# ============================================================
n_segments = 10
segment_length = round(n_points / n_segments)
segment_duration = duration / n_segments

segment_window = make_window(segment_length, window_type, use_window)
segment_window_norm = np.mean(segment_window)

# ============================================================
# CASE 1: Single 1-second DFT with Apodization and Zero-Padding
# ============================================================
n_points_padded = n_points * zero_pad_factor
signal_windowed = signal * window
signal_padded = np.concatenate([signal_windowed, np.zeros(n_points_padded - n_points)])

fft1 = np.fft.fft(signal_padded)
freqs1 = np.arange(n_points_padded) * fs / n_points_padded
magnitude1 = np.abs(fft1) / (n_points * window_norm)

# Focus on region around main signal
freq_range = (49950, 50050)
indices1 = np.where((freqs1 >= freq_range[0]) & (freqs1 <= freq_range[1]))[0]

# ============================================================
# CASE 2: Average of 10 x 100ms DFTs with Apodization and Zero-Padding
# ============================================================
segment_length_padded = segment_length * zero_pad_factor

magnitudes_case2 = []
for i in range(n_segments):
    seg_start = i * segment_length
    seg_end = (i + 1) * segment_length
    seg = signal[seg_start:seg_end]
    seg_windowed = seg * segment_window
    seg_padded = np.concatenate([seg_windowed, np.zeros(segment_length_padded - segment_length)])
    fft_seg = np.fft.fft(seg_padded)
    magnitudes_case2.append(np.abs(fft_seg) / (n_points * segment_window_norm))

magnitude_avg = np.mean(magnitudes_case2, axis=0)
freqs2 = np.arange(segment_length_padded) * fs / segment_length_padded
indices2 = np.where((freqs2 >= freq_range[0]) & (freqs2 <= freq_range[1]))[0]

# ============================================================
# DRIFT ESTIMATION (used by Cases 3 and 4)
# Estimate drift by tracking the spectral peak in overlapping segments.
# Longer segments give finer frequency bins for better peak resolution.
# ============================================================
drift_segment_length = round(0.2 * fs)    # 200 ms segments -> 5 Hz bins
drift_segment_step   = round(0.02 * fs)   # 20 ms step between segments
n_drift_segments = (n_points - drift_segment_length) // drift_segment_step + 1
drift_segment_duration = drift_segment_length / fs

estimated_freqs = []
for i in range(n_drift_segments):
    seg_start = i * drift_segment_step
    seg_end = seg_start + drift_segment_length
    if seg_end > n_points:
        break
    seg = signal[seg_start:seg_end]
    fft_seg = np.fft.fft(seg)
    mag_seg = np.abs(fft_seg)
    freqs_seg = np.arange(drift_segment_length) * fs / drift_segment_length

    # Find peak in expected region
    search_mask = (freqs_seg >= 49995) & (freqs_seg <= 50020)
    search_idx = np.where(search_mask)[0]
    if len(search_idx) == 0:
        estimated_freqs.append(50000.0)   # default if no peak found
    else:
        peak_pos = search_idx[np.argmax(mag_seg[search_idx])]
        estimated_freqs.append(freqs_seg[peak_pos])

estimated_freqs = np.array(estimated_freqs)
n_actual_segments = len(estimated_freqs)

# Center time of each drift segment
segment_times_overlap = (np.arange(n_actual_segments) * drift_segment_step / fs
                          + drift_segment_duration / 2)

# Fit linear drift: frequency(t) = f0_est + drift_rate_est * t
slope, intercept, *_ = stats.linregress(segment_times_overlap, estimated_freqs)
extracted_f0 = intercept
extracted_drift_rate = slope

print(f"Number of drift segments: {n_actual_segments}")
print(f"First 10 estimated frequencies: {estimated_freqs[:10]}")
print(f"Extracted F0:         {extracted_f0:.6f} Hz")
print(f"Extracted drift rate: {extracted_drift_rate:.6f} Hz/s")

# ============================================================
# CASE 3: Drift-Corrected Full DFT
# Method: Demodulate -> low-pass filter -> remodulate on full 1-second signal
# ============================================================

# Step 1: Demodulate to baseband using estimated drift
estimated_phase = 2 * np.pi * (extracted_f0 * t_vec + extracted_drift_rate * t_vec**2 / 2)
demodulated = signal * np.exp(-1j * estimated_phase)

# Step 2: Low-pass filter to isolate baseband (±5000 Hz cutoff)
fft_demod = np.fft.fft(demodulated)
cutoff_bin = round(5000 * n_points / fs)
fft_demod[cutoff_bin : n_points - cutoff_bin] = 0
baseband_complex = np.fft.ifft(fft_demod)

# Step 3: Remodulate to fixed frequency
remod_phase = 2 * np.pi * extracted_f0 * t_vec
corrected_signal_real = 2 * np.real(baseband_complex * np.exp(1j * remod_phase))

# Step 4: Apply window
corrected_signal_windowed = corrected_signal_real * window

# Step 5: Zero-pad and FFT
corrected_signal_padded = np.concatenate([corrected_signal_windowed,
                                           np.zeros(n_points_padded - n_points)])
fft3 = np.fft.fft(corrected_signal_padded)
magnitude3 = np.abs(fft3) / (n_points * window_norm)

# ============================================================
# CASE 4: Drift-Corrected Segments Averaged
# Apply drift correction to each 100 ms segment, then average
# ============================================================
magnitudes_case4 = []
for i in range(n_segments):
    seg_start = i * segment_length
    seg_end = (i + 1) * segment_length
    seg = signal[seg_start:seg_end]
    seg_t = t_vec[seg_start:seg_end]

    # Demodulate
    seg_phase = 2 * np.pi * (extracted_f0 * seg_t + extracted_drift_rate * seg_t**2 / 2)
    seg_demod = seg * np.exp(-1j * seg_phase)

    # Low-pass filter
    seg_fft_demod = np.fft.fft(seg_demod)
    seg_cutoff_bin = round(5000 * segment_length / fs)
    seg_fft_demod[seg_cutoff_bin : segment_length - seg_cutoff_bin] = 0
    seg_baseband = np.fft.ifft(seg_fft_demod)

    # Remodulate
    seg_remod_phase = 2 * np.pi * extracted_f0 * seg_t
    seg_corrected = 2 * np.real(seg_baseband * np.exp(1j * seg_remod_phase))

    # Window and pad
    seg_windowed = seg_corrected * segment_window
    seg_padded = np.concatenate([seg_windowed, np.zeros(segment_length_padded - segment_length)])

    # FFT
    fft_seg = np.fft.fft(seg_padded)
    magnitudes_case4.append(np.abs(fft_seg) / (n_points * segment_window_norm))

magnitude_avg_case4 = np.mean(magnitudes_case4, axis=0)
freqs4 = freqs2.copy()
indices4 = indices2.copy()

# ============================================================
# Peak Measurements (centroid and area)
# ============================================================
peak_width_bins = 5 * zero_pad_factor

def measure_peak(freqs, magnitude, search_indices, peak_width_bins, bin_size):
    """Return (peak_idx, peak_mag, peak_freq, peak_area, centroid_freq)."""
    local_peak = np.argmax(magnitude[search_indices])
    peak_idx = search_indices[local_peak]
    peak_mag  = magnitude[peak_idx]
    peak_freq = freqs[peak_idx]
    a_start = max(0, peak_idx - peak_width_bins)
    a_end   = min(len(freqs), peak_idx + peak_width_bins + 1)
    ai = np.arange(a_start, a_end)
    peak_area = np.sum(magnitude[ai]) * bin_size
    centroid  = np.sum(freqs[ai] * magnitude[ai]) / np.sum(magnitude[ai])
    return peak_idx, peak_mag, peak_freq, peak_area, centroid

peak_idx1, peak_mag1, peak_freq1, peak_area1, centroid1 = measure_peak(
    freqs1, magnitude1, indices1, peak_width_bins, fs / n_points_padded)

peak_idx2, peak_mag2, peak_freq2, peak_area2, centroid2 = measure_peak(
    freqs2, magnitude_avg, indices2, peak_width_bins, fs / segment_length_padded)

peak_idx3, peak_mag3, peak_freq3, peak_area3, centroid3 = measure_peak(
    freqs1, magnitude3, indices1, peak_width_bins, fs / n_points_padded)

peak_idx4, peak_mag4, peak_freq4, peak_area4, centroid4 = measure_peak(
    freqs4, magnitude_avg_case4, indices4, peak_width_bins, fs / segment_length_padded)

# ============================================================
# Print Summary Tables
# ============================================================
bin1 = fs / n_points_padded
bin2 = fs / segment_length_padded

print("\n=== Peak Comparison ===")
hdr = f"{'':28s}  {'Peak Mag':>10s}  {'Peak Freq (Hz)':>16s}  {'Centroid (Hz)':>16s}  {'Area':>10s}  {'Ratio':>7s}"
print(hdr)
print("-" * len(hdr))
print(f"{'Case 1 (smeared):':28s}  {peak_mag1:10.5f}  {peak_freq1:16.6f}  {centroid1:16.6f}  {peak_area1:10.5f}  {'—':>7s}")
print(f"{'Case 2 (averaged):':28s}  {peak_mag2:10.5f}  {peak_freq2:16.6f}  {centroid2:16.6f}  {peak_area2:10.5f}  {peak_mag2/peak_mag1:7.3f}")
print(f"{'Case 3 (full corrected):':28s}  {peak_mag3:10.5f}  {peak_freq3:16.6f}  {centroid3:16.6f}  {peak_area3:10.5f}  {peak_mag3/peak_mag1:7.3f}")
print(f"{'Case 4 (seg corrected):':28s}  {peak_mag4:10.5f}  {peak_freq4:16.6f}  {centroid4:16.6f}  {peak_area4:10.5f}  {peak_mag4/peak_mag1:7.3f}")

print("\n=== Frequency Accuracy ===")
print(f"{'':10s}  {'Peak Freq Error (Hz)':>22s}  {'Centroid Error (Hz)':>22s}")
print(f"{'Case 1:':10s}  {abs(peak_freq1 - f0):22.6f}  {abs(centroid1 - f0):22.6f}")
print(f"{'Case 2:':10s}  {abs(peak_freq2 - f0):22.6f}  {abs(centroid2 - f0):22.6f}")
print(f"{'Case 3:':10s}  {abs(peak_freq3 - extracted_f0):22.6f}  {abs(centroid3 - extracted_f0):22.6f}")
print(f"{'Case 4:':10s}  {abs(peak_freq4 - extracted_f0):22.6f}  {abs(centroid4 - extracted_f0):22.6f}")

print("\n=== Drift Estimation ===")
print(f"  Actual drift rate:      {drift_rate:.4f} Hz/s")
print(f"  Estimated drift rate:   {extracted_drift_rate:.4f} Hz/s")
print(f"  Drift estimation error: {abs(extracted_drift_rate - drift_rate):.4f} Hz/s")
print(f"  Actual f0:              {f0:.6f} Hz")
print(f"  Estimated f0:           {extracted_f0:.6f} Hz")
print(f"  f0 estimation error:    {abs(extracted_f0 - f0):.6f} Hz")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(12, 16))
fig.suptitle('DFT Analysis: Signal with Frequency Drift', fontsize=14, fontweight='bold')

plot_configs = [
    (freqs1, magnitude1, indices1, bin1, peak_mag1, peak_freq1, 'blue',
     'Case 1: Single 1-second DFT'),
    (freqs2, magnitude_avg, indices2, bin2, peak_mag2, peak_freq2, 'red',
     'Case 2: Average of 10×100 ms DFTs'),
    (freqs1, magnitude3, indices1, bin1, peak_mag3, peak_freq3, 'green',
     'Case 3: Drift-Corrected Full DFT'),
    (freqs4, magnitude_avg_case4, indices4, bin2, peak_mag4, peak_freq4, 'purple',
     'Case 4: Drift-Corrected Segments Averaged'),
]

for ax, (freqs, mag, idx, bin_sz, pmag, pfreq, color, title) in zip(axes, plot_configs):
    ax.plot(freqs[idx], mag[idx], color=color, linewidth=0.8)
    ax.set_title(f'{title} ({bin_sz:.4f} Hz bins)\nPeak: {pmag:.5f} at {pfreq:.6f} Hz',
                 fontsize=11)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('drift_correction_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to drift_correction_analysis.png")
