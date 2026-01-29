"""Flask-based Spectrum Analyzer with FFT Analysis and Visualization

This application provides an interactive web GUI for analyzing sine wave files using FFT.
Users can upload files (.raw, .npy, .wav) and adjust all analysis parameters via the interface.

Usage:
    python flask_app.py
"""
from __future__ import annotations

from flask import Flask, render_template, request, jsonify, session, send_file
import numpy as np
from scipy.signal import savgol_filter
import io
import json
import plotly.graph_objects as go
import plotly
import base64
import os
import uuid
from werkzeug.utils import secure_filename
from signal_processing import load_signal, compute_fft, detect_peaks

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


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
    
    # Calculate number of segments from window and step size
    window_samples = int(window_size_sec * sr)
    step_samples = int(step_size_sec * sr)
    if window_samples > 0 and step_samples > 0 and data.size > window_samples:
        num_segments = (data.size - window_samples) // step_samples + 1
    else:
        num_segments = 1
    
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
        xaxis=dict(
            title="Time (seconds)",
            fixedrange=False
        ),
        yaxis=dict(
            title="Amplitude",
            fixedrange=False
        ),
        height=400,
        hovermode='closest',
        dragmode='zoom'
    )
    
    # Perform FFT analysis
    segments = int(max(1, num_segments))
    total = data.size
    
    result = {
        'signal_info': signal_info,
        'json_params': json_params,
        'time_plot': json.loads(plotly.io.to_json(fig_signal)),
        'analysis_params': params,
        'num_segments': num_segments
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
            xaxis=dict(
                title="Frequency (Hz)",
                fixedrange=False
            ),
            yaxis=dict(
                title=y_label,
                fixedrange=False
            ),
            height=500,
            hovermode='closest',
            dragmode='zoom'
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
        
        # Plot multi-segment FFT with layout configured upfront
        y_label = "Magnitude (dB)" if use_db else "Magnitude"
        fig1 = go.Figure(layout=dict(
            xaxis=dict(title="Frequency (Hz)", fixedrange=False),
            yaxis=dict(title=y_label, fixedrange=False),
            height=600,
            hovermode='closest',
            dragmode='zoom',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
        ))
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
                xaxis=dict(
                    title="Segment Number",
                    fixedrange=False
                ),
                yaxis=dict(
                    title=y_label,
                    fixedrange=False
                ),
                height=400,
                hovermode='closest',
                dragmode='zoom'
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


@app.route('/generate_sine', methods=['POST'])
def generate_sine():
    """Generate sine wave with noise and harmonics."""
    from signal_processing import generate_sine_with_noise
    
    # Clean up old generated file if exists
    if 'generated_signal_file' in session:
        old_file = session['generated_signal_file']
        if old_file and os.path.exists(old_file):
            try:
                os.remove(old_file)
            except Exception:
                pass
    
    params = request.json
    
    # Extract parameters
    freq = float(params.get('freq', 15000.0))
    duration = float(params.get('duration', 1.0))
    sample_rate = int(params.get('sample_rate', 2000000))
    amplitude = float(params.get('amplitude', 0.5))
    noise_level = float(params.get('noise_level', 0.01))
    harmonics = int(params.get('harmonics', 0))
    harmonic_amps = params.get('harmonic_amps', None)
    sine_stop_frac = float(params.get('sine_stop_frac', 0.85))
    num_segments = int(params.get('num_segments', 1))
    freq_drift = float(params.get('freq_drift', 0.0))
    make_nonnegative = params.get('make_nonnegative', True)
    
    # Generate signal
    data = generate_sine_with_noise(
        freq=freq,
        duration=duration,
        sample_rate=sample_rate,
        amplitude=amplitude,
        noise_level=noise_level,
        harmonics=harmonics,
        harmonic_amps=harmonic_amps,
        sine_stop_frac=sine_stop_frac,
        num_segments=num_segments,
        freq_drift=freq_drift,
    )
    
    if data.size == 0:
        return jsonify({'error': 'Failed to generate signal'}), 400
    
    # Make non-negative if requested
    if make_nonnegative:
        minv = float(np.min(data))
        if minv < 0.0:
            data = (data - minv).astype(np.float32)
    
    # Calculate statistics
    stats = {
        'total_samples': int(data.size),
        'min_value': float(np.min(data)),
        'max_value': float(np.max(data)),
        'mean_value': float(np.mean(data)),
        'std_value': float(np.std(data)),
        'duration': duration,
        'sample_rate': sample_rate,
    }
    
    # Create time-domain plot
    max_display_points = 50000
    if data.size > max_display_points:
        downsample_factor = data.size // max_display_points
        time_display = np.arange(0, data.size, downsample_factor) / sample_rate
        data_display = data[::downsample_factor]
    else:
        time_display = np.arange(data.size) / sample_rate
        data_display = data
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_display.tolist(),
        y=data_display.tolist(),
        mode='lines',
        name='Amplitude',
        line=dict(width=1, color='steelblue')
    ))
    fig.update_layout(
        xaxis=dict(
            title="Time (seconds)",
            fixedrange=False
        ),
        yaxis=dict(
            title="Amplitude",
            fixedrange=False
        ),
        height=400,
        hovermode='closest',
        dragmode='zoom'
    )
    
    # Store in temporary file for download
    unique_id = str(uuid.uuid4())
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f'generated_{unique_id}.npy')
    np.save(temp_file, data)
    
    # Store config and file path in session
    session['generated_signal'] = unique_id
    session['generated_signal_file'] = temp_file
    session['generated_signal_config'] = params
    
    result = {
        'success': True,
        'stats': stats,
        'plot': json.loads(plotly.io.to_json(fig)),
        'signal_id': unique_id
    }
    
    return jsonify(result)


@app.route('/download_generated/<file_format>', methods=['GET'])
def download_generated(file_format):
    """Download generated signal in specified format."""
    if 'generated_signal' not in session:
        return jsonify({'error': 'No signal generated'}), 400
    
    signal_file = session.get('generated_signal_file')
    config = session.get('generated_signal_config', {})
    
    if not signal_file or not os.path.exists(signal_file):
        return jsonify({'error': 'Signal data not found'}), 400
    
    # Load the numpy array
    data = np.load(signal_file)
    
    # Generate filename
    freq = config.get('freq', 15000)
    duration = config.get('duration', 1.0)
    base_filename = f"sine_{freq:.0f}Hz_{duration:.2f}s"
    
    if file_format == 'npy':
        buffer = io.BytesIO()
        np.save(buffer, data)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f"{base_filename}.npy"
        )
    elif file_format == 'raw':
        buffer = io.BytesIO()
        buffer.write(data.astype('<f4').tobytes())
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f"{base_filename}.raw"
        )
    elif file_format == 'json':
        json_data = {
            'signal_parameters': {
                'base_frequency_hz': config.get('freq'),
                'duration_seconds': config.get('duration'),
                'sample_rate_hz': config.get('sample_rate'),
                'amplitude': config.get('amplitude'),
                'noise_level_sigma': config.get('noise_level'),
            },
            'harmonics': {
                'number_of_harmonics': config.get('harmonics'),
                'harmonic_amplitudes': config.get('harmonic_amps'),
            },
            'segmentation': {
                'number_of_segments': config.get('num_segments'),
                'frequency_drift_hz_per_segment': config.get('freq_drift'),
            },
            'processing': {
                'sine_stop_fraction': config.get('sine_stop_frac', 0.85),
                'make_nonnegative': config.get('make_nonnegative', True),
            },
            'output': {
                'total_samples': int(data.size),
                'data_type': str(data.dtype),
                'min_value': float(np.min(data)),
                'max_value': float(np.max(data)),
                'mean_value': float(np.mean(data)),
            },
        }
        
        buffer = io.BytesIO()
        buffer.write(json.dumps(json_data, indent=2).encode('utf-8'))
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/json',
            as_attachment=True,
            download_name=f"{base_filename}.json"
        )
    else:
        return jsonify({'error': 'Invalid format'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
