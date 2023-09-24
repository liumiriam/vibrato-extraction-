import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the audio file
y, sr = librosa.load('/Users/jieying/Downloads/5.wav', sr=44100)

# Extract F0 using the probabilistic YIN algorithm
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0 = np.where(np.isnan(f0), 0, f0)

#Find peaks
peaks, _ = find_peaks(f0)

# Find troughs (or valleys) in the F0 data
troughs, _ = find_peaks(-f0)

# Get the times for peaks and troughs
times = librosa.times_like(f0)
peak_times = times[peaks]
trough_times = times[troughs]

# Save F0 and their times to "peak.txt"
with open('/Users/jieying/Downloads/F0.txt', 'w') as f:
    for time, value in zip(times, f0):
        f.write(f"{time:.6f}\t{value:.6f}\n")

# Save peaks and their times to "peak.txt"
with open('/Users/jieying/Downloads/peak.txt', 'w') as f:
    for time, value in zip(peak_times, f0[peaks]):
        f.write(f"{time:.6f}\t{value:.6f}\n")

# Save troughs and their times to "troughs.txt"
with open('/Users/jieying/Downloads/troughs.txt', 'w') as f:
    for time, value in zip(trough_times, f0[troughs]):
        f.write(f"{time:.6f}\t{value:.6f}\n")

#start plotting
plt.figure(figsize=(10, 4))

# Plot the F0 values
plt.plot(times, f0, label='F0 (fundamental frequency)', color='#3498db')


# Highlight the peaks
plt.scatter(times[peaks], f0[peaks], color='#a6d785', s=20, marker='o', label='F0 Peaks')

# Highlight the troughs (valleys)
plt.scatter(times[troughs], f0[troughs], color='#e74c3c', s=20, marker='x', label='F0 Troughs (Valleys)')

# Setting the x and y axis labels
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend(loc='upper right')
plt.savefig('/Users/jieying/Downloads/5.pdf', format='pdf')
