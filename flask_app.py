"""Flask-based Spectrum Analyzer with FFT Analysis and Visualization

This application provides an interactive web GUI for analyzing sine wave files using FFT.
Users can upload files (.raw, .npy, .wav) and adjust all analysis parameters via the interface.

Usage:
    python flask_app.py
"""
from __future__ import annotations

from flask import Flask, render_template, request, jsonify, session, send_file
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy.signal import savgol_filter
import io
import json
import plotly.graph_objects as go
import plotly
import base64
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_signal(file_path: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load signal from file."""
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
    """Compute FFT with windowing and zero-padding."""
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
    """Simple peak detection: find local maxima that are threshold above neighbors."""
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


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'signal_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['signal_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file with unique name
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
    file.save(file_path)
    
    # Also save JSON file if provided
    json_path = None
    if 'json_file' in request.files and request.files['json_file'].filename:
        json_file = request.files['json_file']
        json_filename = secure_filename(json_file.filename)
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{json_filename}")
        json_file.save(json_path)
    
    # Store file paths in session
    session['signal_file'] = file_path
    session['signal_filename'] = filename
    if json_path:
        session['json_file'] = json_path
    
    return jsonify({
        'success': True,
        'filename': filename,
        'file_id': unique_id
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Perform FFT analysis on uploaded file."""
    if 'signal_file' not in session:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Get parameters from request
    params = request.json
    sample_rate = int(params.get('sample_rate', 2000000))
    window_type = params.get('window_type', 'hanning')
    use_db = params.get('use_db', True)
    zp_factor = float(params.get('zp_factor', 4.0))
    use_custom_nfft = params.get('use_custom_nfft', False)
    nfft = int(params.get('nfft', 262144)) if use_custom_nfft else None
    num_segments = int(params.get('num_segments', 1))
    window_size_sec = float(params.get('window_size_sec', 0.1))
    step_size_sec = float(params.get('step_size_sec', 0.01))
    use_stft = params.get('use_stft', False)
    center_freq = float(params.get('center_freq', 15000))
    freq_band = float(params.get('freq_band', 1000))
    peak_threshold = float(params.get('peak_threshold', 10.0))
    peak_min_distance = int(params.get('peak_min_distance', 8))
    smooth_window = int(params.get('smooth_window', 11))
    smooth_polyorder = int(params.get('smooth_polyorder', 3))
    show_full_signal = params.get('show_full_signal', False)
    start_time = float(params.get('start_time', 0.0))
    num_display_points = int(params.get('num_display_points', 10000))
    
    # Load signal
    file_path = session['signal_file']
    data, sr = load_signal(file_path, sample_rate)
    
    if data.size == 0:
        return jsonify({'error': 'Failed to load signal data'}), 400
    
    duration = data.size / sr
    
    # Prepare signal info
    signal_info = {
        'sample_rate': sr,
        'total_samples': int(data.size),
        'duration': float(duration),
        'min_value': float(np.min(data)),
        'max_value': float(np.max(data)),
        'mean_value': float(np.mean(data))
    }
    
    # Load JSON parameters if available
    json_params = None
    if 'json_file' in session:
        try:
            with open(session['json_file'], 'r') as f:
                json_params = json.load(f)
        except Exception:
            pass
    
    # Create time-domain plot
    if show_full_signal:
        max_display_points = 50000
        if data.size > max_display_points:
            downsample_factor = data.size // max_display_points
            time_display = np.arange(0, data.size, downsample_factor) / sr
            data_display = data[::downsample_factor]
        else:
            time_display = np.arange(data.size) / sr
            data_display = data
    else:
        start_idx = int(start_time * sr)
        end_idx = min(start_idx + num_display_points, data.size)
        data_display = data[start_idx:end_idx]
        time_display = np.arange(start_idx, end_idx) / sr
    
    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(
        x=time_display.tolist(),
        y=data_display.tolist(),
        mode='lines',
        name='Amplitude',
        line=dict(width=1, color='steelblue')
    ))
    fig_signal.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=400,
        hovermode='x unified'
    )
    
    # Perform FFT analysis
    segments = int(max(1, num_segments))
    total = data.size
    
    result = {
        'signal_info': signal_info,
        'json_params': json_params,
        'time_plot': json.loads(plotly.io.to_json(fig_signal)),
        'analysis_params': params
    }
    
    if segments <= 1:
        # Single segment analysis
        freqs, mag = compute_fft(data, sr, window_samples=None, nfft=nfft, 
                                 zp_factor=zp_factor, use_db=use_db, window_type=window_type)
        
        if freqs.size == 0:
            return jsonify({'error': 'No data to analyze'}), 400
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs.tolist(), y=mag.tolist(), mode='lines', name='FFT Magnitude'))
        y_label = "Magnitude (dB)" if use_db else "Magnitude"
        fig.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            height=500,
            hovermode='x unified'
        )
        
        result['spectrum_plot'] = json.loads(plotly.io.to_json(fig))
        result['single_segment'] = True
        
    else:
        # Multi-segment analysis
        window_samples = int(window_size_sec * sr)
        step_samples = int(step_size_sec * sr)
        
        if window_samples <= 0 or window_samples > total:
            return jsonify({'error': f'Invalid window size'}), 400
        
        if use_stft:
            num_windows = (total - window_samples) // step_samples + 1
            if num_windows <= 0:
                return jsonify({'error': 'Signal too short'}), 400
        else:
            num_windows = (total - window_samples) // step_samples + 1
            if num_windows <= 0:
                return jsonify({'error': 'Signal too short'}), 400
        
        all_freqs = []
        all_mag = []
        
        for i in range(num_windows):
            if use_stft:
                start = i * step_samples
                end = start + window_samples
                if end > total:
                    break
            else:
                start = 0
                end = window_samples + i * step_samples
                if end > total:
                    end = total
            
            seg = data[start:end]
            if seg.size == 0:
                continue
            freqs_i, mag_i = compute_fft(seg, sr, window_samples=None, nfft=nfft,
                                        zp_factor=zp_factor, use_db=use_db, window_type=window_type)
            all_freqs.append(freqs_i)
            all_mag.append(mag_i)
        
        if not all_freqs:
            return jsonify({'error': 'No segments yielded data'}), 400
        
        # Frequency window
        fmin = center_freq - freq_band
        fmax = center_freq + freq_band
        
        # Plot multi-segment FFT
        fig1 = go.Figure()
        total_peaks_detected = 0
        all_segment_data = []
        
        for i, (freqs, mag) in enumerate(zip(all_freqs, all_mag)):
            if freqs.size == 0:
                continue
            idx = np.where((freqs >= fmin) & (freqs <= fmax))
            if idx[0].size == 0:
                continue
            
            freqs_win = freqs[idx]
            mag_win = mag[idx]
            
            fig1.add_trace(go.Scatter(
                x=freqs_win.tolist(),
                y=mag_win.tolist(),
                mode='lines',
                name=f'Window {i+1}',
                line=dict(width=1)
            ))
            
            # Peak detection
            detected_peaks = []
            if mag_win.size >= 3:
                mag_smoothed = mag_win.copy()
                if mag_smoothed.size >= smooth_window:
                    mag_smoothed = savgol_filter(mag_smoothed, smooth_window, smooth_polyorder)
                    mag_smoothed = savgol_filter(mag_smoothed, smooth_window, smooth_polyorder)
                
                peak_idx = detect_peaks(mag_smoothed, threshold=peak_threshold, 
                                       min_distance=peak_min_distance, use_db=use_db)
                
                if peak_idx.size > 0:
                    total_peaks_detected += peak_idx.size
                    for pk in peak_idx:
                        if pk < freqs_win.size:
                            pk_freq = freqs_win[pk]
                            pk_mag = mag_win[pk]
                            detected_peaks.append((float(pk_freq), float(pk_mag)))
                            
                            y_min = np.min(mag_win) if mag_win.size > 0 else 0
                            
                            fig1.add_trace(go.Scatter(
                                x=[pk_freq, pk_freq],
                                y=[y_min, pk_mag],
                                mode='lines',
                                line=dict(color='red', width=2, dash='dash'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
            
            all_segment_data.append((i + 1, freqs_win, mag_win, detected_peaks))
        
        y_label = "Magnitude (dB)" if use_db else "Magnitude"
        fig1.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            height=600,
            hovermode='x unified'
        )
        
        # Peak magnitude plot
        seg_nums = []
        mags = []
        peak_window = 5.0
        
        for seg_num, freqs_win, mag_win, detected_peaks in all_segment_data:
            if freqs_win.size == 0 or mag_win.size == 0:
                continue
            
            best_mag = None
            
            if detected_peaks:
                peaks_near_center = [(pk_freq, pk_mag) for pk_freq, pk_mag in detected_peaks 
                                    if abs(pk_freq - center_freq) <= peak_window]
                if peaks_near_center:
                    peaks_near_center.sort(key=lambda x: abs(x[0] - center_freq))
                    best_mag = peaks_near_center[0][1]
            
            if best_mag is None:
                closest_idx = np.argmin(np.abs(freqs_win - center_freq))
                best_mag = mag_win[closest_idx]
            
            seg_nums.append(seg_num)
            mags.append(float(best_mag))
        
        fig2 = None
        mean_mag = None
        rmsd = None
        
        if seg_nums:
            mags_array = np.array(mags)
            mean_mag = float(np.mean(mags_array))
            rmsd = float(np.sqrt(np.mean((mags_array - mean_mag) ** 2)))
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=seg_nums,
                y=mags,
                mode='markers+lines',
                marker=dict(size=10, color='blue'),
                line=dict(width=2),
                name='Peak Magnitude'
            ))
            
            fig2.add_trace(go.Scatter(
                x=[min(seg_nums), max(seg_nums)],
                y=[mean_mag, mean_mag],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Mean'
            ))
            
            fig2.update_layout(
                xaxis_title="Segment Number",
                yaxis_title=y_label,
                height=400,
                hovermode='x unified'
            )
        
        result['multi_segment_plot'] = json.loads(plotly.io.to_json(fig1))
        result['peak_magnitude_plot'] = json.loads(plotly.io.to_json(fig2)) if fig2 else None
        result['total_peaks_detected'] = total_peaks_detected
        result['num_windows'] = len(all_freqs)
        result['mean_mag'] = mean_mag
        result['rmsd'] = rmsd
        result['single_segment'] = False
        result['mode_label'] = 'STFT' if use_stft else 'STORI'
    
    return jsonify(result)


@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download comprehensive HTML report."""
    # Implementation would mirror the Streamlit version
    # For brevity, returning a simple message
    return jsonify({'error': 'Not implemented yet'}), 501


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
