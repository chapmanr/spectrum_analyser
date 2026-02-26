import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


wave_duration = 3
sample_rate = 100
freq = 2
q = 5
noise_amplitude = 0.3
rng = np.random.default_rng(seed=42)


samples = wave_duration*sample_rate
samples_decimated = int(samples/q)


x = np.linspace(0, wave_duration, samples, endpoint=False)
xnew = np.linspace(0, wave_duration, samples_decimated, endpoint=False)

y = np.cos(x*np.pi*freq*2)

# Noise before decimation: added at full sample rate, then decimated
# The decimate() low-pass filter will attenuate high-frequency noise components
noise_pre = rng.normal(0, noise_amplitude, size=samples)
y_noisy_pre = y + noise_pre
ydem_from_noisy = signal.decimate(y_noisy_pre, q)

# Clean decimation for reference
ydem = signal.decimate(y, q)

# Noise after decimation: added at the reduced sample rate, no filtering applied
noise_post = rng.normal(0, noise_amplitude, size=samples_decimated)
ydem_noisy_post = ydem + noise_post


fig, axes = plt.subplots(5, 1, figsize=(10, 12))

axes[0].plot(x, y, '.-')
axes[0].set_title('Original signal (no noise)')
axes[0].set_ylabel('Amplitude')

axes[1].plot(x, y_noisy_pre, '.-', alpha=0.6)
axes[1].set_title(f'Signal + white noise BEFORE decimation (amplitude={noise_amplitude})')
axes[1].set_ylabel('Amplitude')

axes[2].plot(xnew, ydem_from_noisy, 'o-')
axes[2].set_title('After decimating the noisy signal (anti-alias filter attenuates noise)')
axes[2].set_ylabel('Amplitude')

# FFT of decimated noisy signal with Hann window and zero padding (4x)
hann_window_dec = np.hanning(len(ydem_from_noisy))
y_dec_windowed = ydem_from_noisy * hann_window_dec

# Zero padding to 4 times the original length
n_fft_dec = len(y_dec_windowed) * 4
y_dec_fft = np.fft.fft(y_dec_windowed, n_fft_dec)
freqs_dec = np.fft.fftfreq(n_fft_dec, 1/(sample_rate/q))  # Use decimated sample rate

# Only plot positive frequencies
positive_freqs_dec = freqs_dec[:n_fft_dec//2]
magnitude_dec = np.abs(y_dec_fft[:n_fft_dec//2])

axes[3].plot(positive_freqs_dec, magnitude_dec)
axes[3].set_title('FFT of decimated noisy signal (Hann window, 4x zero padding)')
axes[3].set_ylabel('Magnitude')

# FFT of undecimated noisy signal with Hann window and zero padding (4x)
hann_window_noisy = np.hanning(len(y_noisy_pre))
y_noisy_windowed = y_noisy_pre * hann_window_noisy

# Zero padding to 4 times the original length
n_fft_noisy = len(y_noisy_windowed) * 4
y_noisy_fft = np.fft.fft(y_noisy_windowed, n_fft_noisy)
freqs_noisy = np.fft.fftfreq(n_fft_noisy, 1/sample_rate)

# Only plot positive frequencies
positive_freqs_noisy = freqs_noisy[:n_fft_noisy//2]
magnitude_noisy = np.abs(y_noisy_fft[:n_fft_noisy//2])

axes[4].plot(positive_freqs_noisy, magnitude_noisy)
axes[4].set_title('FFT of undecimated noisy signal (Hann window, 4x zero padding)')
axes[4].set_ylabel('Magnitude')
axes[4].set_xlabel('Frequency, Hz')

plt.tight_layout()
plt.show()
