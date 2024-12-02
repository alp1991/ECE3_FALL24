# Step 1: Import the necessary modules
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

# Step 2: Read the audio file
input_file = "piano_w_noise.wav"  # Replace with your .wav file path
output_file = "piano_filtered_where.wav"

sample_rate, audio_data = wavfile.read(input_file)

# Handle stereo audio by taking only one channel if needed
#if audio_data.ndim > 1:
#    audio_data = audio_data[:, 0]  # Use the first channel if stereo
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)  # Convert to mono by averaging channels

# Step 3: Apply FFT to the audio signal
data_length = len(audio_data)
audio_fft = fft(audio_data)  # the amplitudes of the frequencies in the transform
frequencies = fftfreq(data_length, 1 / sample_rate) # frequency bins

# Step 4: Filter out undesired frequencies
# Define a cutoff frequency (in Hz) beyond which frequencies will be filtered out
cutoff_frequency = 1200  # Adjust as needed
# Filtering with np.where()
filtered_fft = np.where(np.abs(frequencies) > cutoff_frequency, 0, audio_fft)

# Alternative filtering with np.array conditional indexing:
#filtered_fft = audio_fft.copy()
#filtered_fft[np.abs(frequencies) > cutoff_frequency] = 0

# Optional: Plot the frequency spectrum before and after filtering
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(frequencies[:data_length // 2], np.abs(audio_fft[:data_length // 2]))
plt.title("Original Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.plot(frequencies[:data_length // 2], np.abs(filtered_fft[:data_length // 2]))
plt.title("Filtered Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Step 5: Apply the inverse FFT to get the time-domain signal
filtered_audio = ifft(filtered_fft).real  # Take the real part of the IFFT
# The imaginary part is only needed for the phase information

# Step 6: Convert the filtered audio back to the original data type 

filtered_audio = np.int16(filtered_audio/ np.max(np.abs(filtered_audio))*32767)  # Convert back to int16 format

# Step 7: Save the filtered audio to a new .wav file
wavfile.write(output_file, sample_rate, filtered_audio)

print(f"Filtered audio saved as {output_file}")
