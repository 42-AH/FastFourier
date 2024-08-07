import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
import random

# Fast Fourier Transform (FFT) function
def fft(x):
    N = len(x)
    if N <= 1:
        return x
    if N % 2 != 0:
        x.append(0)
        N += 1
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Sampling frequency
fs = 100
# Create time points
t = np.linspace(0, 10, fs * 10, endpoint=False)

# Function to make a heartbeat wave
def heartbeat_wave(t, frequency, duration):
    t_cycle = np.linspace(0, duration, int(fs * duration), endpoint=False)
    heartbeat_cycle = np.exp(-0.5 * ((t_cycle - duration / 2) / 0.1) ** 2) - np.exp(-0.5 * ((t_cycle - duration) / 0.3) ** 2)
    num_cycles = int(len(t) / len(t_cycle))
    heartbeat_signal = np.tile(heartbeat_cycle, num_cycles + 1)[:len(t)]
    heartbeat_signal += 0.05 * np.random.normal(size=len(t))
    return heartbeat_signal

# Random frequency for the heartbeat
f_heartbeat = random.uniform(0.8, 2.4)
heartbeat = heartbeat_wave(t, f_heartbeat, 1 / f_heartbeat)

# Do FFT
fft_result = fft(list(heartbeat))
fft_result = np.array(fft_result)

# Get frequencies and magnitudes
frequencies = np.fft.fftfreq(len(fft_result), 1 / fs)
magnitude = np.abs(fft_result)

# Filter positive frequencies
positive_freq_indices = np.where(frequencies > 0)
positive_frequencies = frequencies[positive_freq_indices]
positive_magnitude = magnitude[positive_freq_indices]

# Find the most significant frequency (peak)
peak_freq = positive_frequencies[np.argmax(positive_magnitude)]
heart_rate_bpm = peak_freq * 60

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t, heartbeat, 'k')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')
ax1.set_title('Heartbeat Signal')

ax2.plot(positive_frequencies, positive_magnitude, 'r')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Magnitude')
ax2.set_title('FFT of Heartbeat Signal')

fig.text(0.5, 0.02, f"Estimated heart rate: {heart_rate_bpm:.2f} BPM", ha="center", fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
