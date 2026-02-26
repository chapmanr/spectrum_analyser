# Flask Spectrum Analyzer

A Flask-based web application for analyzing sine wave files using FFT with interactive visualizations.

## Features

- 📂 **File Upload**: Support for .npy, .raw, and .wav files
- 📊 **FFT Analysis**: Multiple window functions (Hanning, Hamming, Blackman, etc.)
- 🔍 **Peak Detection**: Automatic peak detection with smoothing
- 📈 **Multi-Segment Analysis**: STFT (sliding windows) and STORI (expanding windows) modes
- 📉 **Interactive Plots**: Powered by Plotly.js
- ⚙️ **Customizable Parameters**: Full control over all analysis settings

## Installation

1. Install dependencies:
```bash
pip install -r requirements_flask.txt
```

2. Run the application:
```bash
python flask_app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### 1. Upload Signal File
- Click on "Signal File" in the sidebar
- Select your .npy, .raw, or .wav file
- Optionally upload a JSON parameter file
- Set the sample rate (for .npy and .raw files)
- Click "Upload & Load"

### 2. Configure Analysis Parameters

**FFT Settings:**
- Window Function: Choose from 9 different window types
- Display in dB: Toggle between dB and absolute magnitude
- Zero-Pad Factor: Improve frequency resolution (1-16x)
- Custom NFFT: Optionally override automatic FFT size

**Segmentation:**
- Number of Segments: 1 for single analysis, >1 for multi-segment
- Window Size: Length of each analysis window in seconds
- Step Size: How much to advance between windows
- STFT Mode: Toggle between STFT and STORI modes

**Display Settings:**
- Center Frequency: Focus frequency for zoomed view
- Frequency Band: ±Hz around center frequency

**Peak Detection:**
- Peak Threshold: Minimum threshold for peak detection
- Min Peak Distance: Minimum separation between peaks

**Smoothing:**
- Smoothing Window: Savitzky-Golay window size
- Polynomial Order: Order of smoothing polynomial

### 3. Analyze Signal
- Click "Analyze Signal" button
- View results including:
  - Time-domain signal plot
  - Frequency spectrum (single or multi-segment)
  - Peak magnitude analysis
  - Statistics and metrics

## Differences from Streamlit Version

The Flask version provides the same core functionality as the Streamlit version but with a traditional web application architecture:

### Streamlit Version
- Single-page reactive application
- Auto-rerun on parameter changes
- Built-in widgets and state management
- Run with: `streamlit run sl_main.py`

### Flask Version
- Client-server architecture
- Explicit form submission
- Custom HTML/CSS/JavaScript frontend
- Run with: `python flask_app.py`

## File Structure

```
sl_truemass_data_explorer/
├── flask_app.py              # Flask backend application
├── templates/
│   └── index.html            # Frontend HTML template
├── temp_uploads/             # Temporary file storage (auto-created)
├── requirements_flask.txt    # Python dependencies
├── sl_main.py               # Original Streamlit version
└── requirements.txt          # Streamlit dependencies
```

## API Endpoints

### POST /upload
Upload signal and optional JSON parameter files.

**Form Data:**
- `signal_file`: Signal file (.npy, .raw, .wav)
- `json_file`: Optional JSON parameter file

**Response:**
```json
{
  "success": true,
  "filename": "signal.npy",
  "file_id": "unique-id"
}
```

### POST /analyze
Perform FFT analysis on uploaded file.

**Request Body:** JSON with all analysis parameters

**Response:** JSON with:
- `signal_info`: Signal metadata
- `time_plot`: Plotly JSON for time-domain plot
- `spectrum_plot` or `multi_segment_plot`: Frequency domain plots
- `peak_magnitude_plot`: Peak analysis plot (multi-segment only)
- Statistics and metrics

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- IE: ❌ Not supported

## Performance Notes

- Files up to 500MB supported
- Large files may take time to process
- Multi-segment analysis is computationally intensive
- Consider downsampling very large signals for display

## Troubleshooting

**File upload fails:**
- Check file size (max 500MB)
- Ensure file format is supported (.npy, .raw, .wav)

**Analysis takes too long:**
- Reduce number of segments
- Increase step size
- Reduce NFFT size

**Plots not displaying:**
- Check browser console for errors
- Ensure JavaScript is enabled
- Try refreshing the page

## License

Same as the original Streamlit version.
