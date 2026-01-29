"""Streamlit-based Sine Wave Generator

Interactive GUI for generating sine waves with noise, harmonics, and frequency drift.
Users can configure all parameters and download the generated data as .npy files.

Usage:
    streamlit run sine_wave_generator_streamlit.py
"""
from __future__ import annotations

import streamlit as st
import numpy as np
import math
import io
import json
from typing import Optional


def generate_sine_with_noise(
    freq: float,
    duration: float,
    sample_rate: int,
    amplitude: float,
    noise_level: float,
    harmonics: int = 0,
    harmonic_amps: list[float] | None = None,
    sine_stop_frac: float = 0.85,
    num_segments: int = 1,
    freq_drift: float = 0.0,
) -> np.ndarray:
    """Generate a primary sine plus optional integer harmonics.

    - `harmonics` specifies how many integer harmonics to add (1 => 2nd harmonic, etc.).
    - `harmonic_amps` if provided should be a list of absolute amplitudes for each harmonic; if
      fewer amplitudes are provided than `harmonics`, missing amplitudes default to halving each step.
    - `num_segments` divides the signal into equal parts, each with a different frequency.
    - `freq_drift` specifies the frequency increase (in Hz) per segment.
    """
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


def main():
    st.set_page_config(page_title="Sine Wave Generator", layout="wide")
    
    st.title("🌊 Sine Wave Generator with Noise")
    st.markdown("Generate custom sine waves with harmonics, noise, and frequency drift.")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Generation Parameters")
    
    st.sidebar.subheader("Basic Settings")
    
    # Frequency
    freq = st.sidebar.number_input(
        "Frequency (Hz)",
        min_value=0.1,
        max_value=10000000.0,
        value=15000.0,
        step=100.0,
        help="Base frequency of the sine wave"
    )
    
    # Duration
    duration = st.sidebar.number_input(
        "Duration (seconds)",
        min_value=0.001,
        max_value=60.0,
        value=1.0,
        step=0.1,
        help="Length of the generated signal"
    )
    
    # Sample rate
    sample_rate = st.sidebar.number_input(
        "Sample Rate (Hz)",
        min_value=1000,
        max_value=10000000,
        value=2000000,
        step=10000,
        help="Sampling rate in samples per second"
    )
    
    # Amplitude
    amplitude = st.sidebar.slider(
        "Amplitude",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.01,
        help="Peak amplitude of the sine wave"
    )
    
    # Noise level
    noise_level = st.sidebar.slider(
        "Noise Level (σ)",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Standard deviation of additive Gaussian noise"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Harmonics")
    
    # Number of harmonics
    harmonics = st.sidebar.number_input(
        "Number of Harmonics",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of integer harmonics to add (e.g., 3 adds 2nd, 3rd, 4th harmonics)"
    )
    
    # Harmonic amplitudes
    harmonic_amps = None
    if harmonics > 0:
        use_custom_harmonic_amps = st.sidebar.checkbox(
            "Custom Harmonic Amplitudes",
            value=False,
            help="Specify custom amplitudes for each harmonic"
        )
        
        if use_custom_harmonic_amps:
            st.sidebar.markdown("**Harmonic Amplitudes:**")
            harmonic_amps = []
            for i in range(harmonics):
                amp = st.sidebar.number_input(
                    f"Harmonic {i+2} amplitude",
                    min_value=0.0,
                    max_value=2.0,
                    value=amplitude * (0.5 ** (i+1)),
                    step=0.01,
                    key=f"harm_amp_{i}"
                )
                harmonic_amps.append(amp)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Signal Segmentation")
    
    # Number of segments
    num_segments = st.sidebar.number_input(
        "Number of Segments",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Divide signal into segments with frequency drift"
    )
    
    # Frequency drift
    freq_drift = st.sidebar.number_input(
        "Frequency Drift (Hz/segment)",
        min_value=-1000.0,
        max_value=1000.0,
        value=1.0,
        step=0.1,
        help="Frequency increase per segment"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Settings")
    
    # Sine stop fraction
    sine_stop_frac = st.sidebar.slider(
        "Sine Stop Fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        help="Fraction of signal after which sine stops (remaining is noise only)"
    )
    
    # Make non-negative
    make_nonnegative = st.sidebar.checkbox(
        "Make Non-Negative",
        value=True,
        help="Shift signal so minimum value is 0"
    )
    
    # Generate button
    st.sidebar.markdown("---")
    generate_button = st.sidebar.button("🎵 Generate Signal", type="primary")
    
    # Main content area
    if not generate_button and 'generated_data' not in st.session_state:
        st.info("👈 Configure parameters in the sidebar and click **Generate Signal** to create a waveform.")
        
        # Display preview of what will be generated
        st.markdown("### Configuration Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Signal Properties:**")
            st.markdown(f"- Base Frequency: {freq:,.1f} Hz")
            st.markdown(f"- Duration: {duration:.3f} seconds")
            st.markdown(f"- Sample Rate: {sample_rate:,} Hz")
            st.markdown(f"- Total Samples: {int(sample_rate * duration):,}")
            
        with col2:
            st.markdown("**Generation Details:**")
            st.markdown(f"- Amplitude: {amplitude:.2f}")
            st.markdown(f"- Noise Level: {noise_level:.3f}")
            st.markdown(f"- Harmonics: {harmonics}")
            st.markdown(f"- Segments: {num_segments}")
            
        if num_segments > 1:
            st.markdown("**Frequency Drift:**")
            final_freq = freq + (num_segments - 1) * freq_drift
            st.markdown(f"- Frequency range: {freq:.1f} Hz → {final_freq:.1f} Hz")
            st.markdown(f"- Drift per segment: {freq_drift:.1f} Hz")
        
        return
    
    # Generate or retrieve from session state
    if generate_button:
        with st.spinner("Generating signal..."):
            # Calculate memory requirement
            n_samples = int(sample_rate * duration)
            approx_mem_mb = (n_samples * 4) / (1024 * 1024)
            
            if approx_mem_mb > 1000:
                st.warning(f"⚠️ Large signal: ~{approx_mem_mb:.1f} MB of memory required")
            
            # Generate the signal
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
            
            # Make non-negative if requested
            if make_nonnegative:
                minv = float(np.min(data)) if data.size else 0.0
                if minv < 0.0:
                    data = (data - minv).astype(np.float32)
            
            # Store in session state
            st.session_state['generated_data'] = data
            st.session_state['config'] = {
                'freq': freq,
                'duration': duration,
                'sample_rate': sample_rate,
                'amplitude': amplitude,
                'noise_level': noise_level,
                'harmonics': harmonics,
                'harmonic_amps': harmonic_amps if harmonic_amps else None,
                'num_segments': num_segments,
                'freq_drift': freq_drift,
                'sine_stop_frac': sine_stop_frac,
                'make_nonnegative': make_nonnegative,
                'total_samples': data.size,
                'data_type': str(data.dtype),
            }
    
    # Display results
    if 'generated_data' in st.session_state:
        data = st.session_state['generated_data']
        config = st.session_state['config']
        
        st.success(f"✅ Generated {data.size:,} samples ({data.dtype})")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{data.size:,}")
        with col2:
            st.metric("Min Value", f"{np.min(data):.6f}")
        with col3:
            st.metric("Max Value", f"{np.max(data):.6f}")
        with col4:
            st.metric("Mean Value", f"{np.mean(data):.6f}")
        
        # Plot the waveform
        st.subheader("📊 Waveform")
        
        import plotly.graph_objects as go
        
        # Downsample for display if needed
        max_display_points = 50000
        if data.size > max_display_points:
            downsample_factor = data.size // max_display_points
            time_display = np.arange(0, data.size, downsample_factor) / config['sample_rate']
            data_display = data[::downsample_factor]
            st.info(f"Waveform downsampled by factor of {downsample_factor} for display ({data.size:,} → {data_display.size:,} points)")
        else:
            time_display = np.arange(data.size) / config['sample_rate']
            data_display = data
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_display,
            y=data_display,
            mode='lines',
            name='Amplitude',
            line=dict(width=1, color='steelblue')
        ))
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=400,
            hovermode='x unified',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download section
        st.subheader("💾 Download Generated Data")
        
        # Generate base filename
        base_filename = f"sine_{config['freq']:.0f}Hz_{config['duration']:.2f}s"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create numpy file in memory
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, data)
            npy_buffer.seek(0)
            
            st.download_button(
                label="📥 Download as .npy",
                data=npy_buffer,
                file_name=f"{base_filename}.npy",
                mime="application/octet-stream",
                help="Download as NumPy array file"
            )
        
        with col2:
            # Create raw binary file in memory
            raw_buffer = io.BytesIO()
            raw_buffer.write(data.astype('<f4').tobytes())
            raw_buffer.seek(0)
            
            st.download_button(
                label="📥 Download as .raw",
                data=raw_buffer,
                file_name=f"{base_filename}.raw",
                mime="application/octet-stream",
                help="Download as raw float32 binary file"
            )
        
        with col3:
            # Create JSON parameter file
            json_data = {
                'signal_parameters': {
                    'base_frequency_hz': config['freq'],
                    'duration_seconds': config['duration'],
                    'sample_rate_hz': config['sample_rate'],
                    'amplitude': config['amplitude'],
                    'noise_level_sigma': config['noise_level'],
                },
                'harmonics': {
                    'number_of_harmonics': config['harmonics'],
                    'harmonic_amplitudes': config.get('harmonic_amps'),
                },
                'segmentation': {
                    'number_of_segments': config['num_segments'],
                    'frequency_drift_hz_per_segment': config['freq_drift'],
                },
                'processing': {
                    'sine_stop_fraction': config.get('sine_stop_frac', 0.85),
                    'make_nonnegative': config.get('make_nonnegative', True),
                },
                'output': {
                    'total_samples': config.get('total_samples', data.size),
                    'data_type': config.get('data_type', str(data.dtype)),
                    'min_value': float(np.min(data)),
                    'max_value': float(np.max(data)),
                    'mean_value': float(np.mean(data)),
                },
            }
            
            # Calculate frequency range if multi-segment
            if config['num_segments'] > 1:
                final_freq = config['freq'] + (config['num_segments'] - 1) * config['freq_drift']
                json_data['segmentation']['frequency_range_hz'] = [config['freq'], final_freq]
            
            json_buffer = io.BytesIO()
            json_buffer.write(json.dumps(json_data, indent=2).encode('utf-8'))
            json_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Parameters (.json)",
                data=json_buffer,
                file_name=f"{base_filename}.json",
                mime="application/json",
                help="Download generation parameters as JSON file"
            )
        
        # Configuration summary
        with st.expander("ℹ️ Generation Details"):
            st.markdown(f"""
            **Signal Parameters:**
            - Base Frequency: {config['freq']:.2f} Hz
            - Duration: {config['duration']:.3f} seconds
            - Sample Rate: {config['sample_rate']:,} Hz
            - Amplitude: {config['amplitude']:.2f}
            - Noise Level (σ): {config['noise_level']:.3f}
            
            **Harmonics:**
            - Number of Harmonics: {config['harmonics']}
            
            **Segmentation:**
            - Number of Segments: {config['num_segments']}
            - Frequency Drift: {config['freq_drift']:.2f} Hz per segment
            """)
            
            if config['num_segments'] > 1:
                final_freq = config['freq'] + (config['num_segments'] - 1) * config['freq_drift']
                st.markdown(f"- **Frequency Range:** {config['freq']:.2f} Hz → {final_freq:.2f} Hz")


if __name__ == '__main__':
    main()
