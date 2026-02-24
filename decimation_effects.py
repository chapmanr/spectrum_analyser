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


fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(x, y, '.-')
axes[0].set_title('Original signal (no noise)')
axes[0].set_ylabel('Amplitude')

axes[1].plot(x, y_noisy_pre, '.-', alpha=0.6)
axes[1].set_title(f'Signal + white noise BEFORE decimation (amplitude={noise_amplitude})')
axes[1].set_ylabel('Amplitude')

axes[2].plot(xnew, ydem_from_noisy, 'o-')
axes[2].set_title('After decimating the noisy signal (anti-alias filter attenuates noise)')
axes[2].set_ylabel('Amplitude')

axes[3].plot(xnew, ydem_noisy_post, 'o-')
axes[3].set_title(f'Clean decimation + white noise AFTER decimation (amplitude={noise_amplitude})')
axes[3].set_ylabel('Amplitude')
axes[3].set_xlabel('Time, Seconds')

plt.tight_layout()
plt.show()
