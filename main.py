import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load spectrum image of banana
image = cv2.imread('banana_spectrum.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Sum pixel intensities along the vertical axis (assuming horizontal spectrum)
intensity_profile = np.sum(gray_blur, axis=0)

# Normalize intensity values
intensity_profile = (intensity_profile - np.min(intensity_profile)) / (np.max(intensity_profile) - np.min(intensity_profile))

# Find spectral peaks (brightest points)
peaks, _ = find_peaks(intensity_profile, height=0.5, distance=10)

# Define known wavelengths for banana ripeness:
# Green (Unripe) ~ 540-580 nm
# Yellow (Ripe) ~ 580-620 nm
# Brown/Black (Overripe) ~ 650-700 nm (or no strong peaks)
known_unripe = [550]  # Green Chlorophyll peak
known_ripe = [590, 610]  # Yellow Carotenoid peaks
known_overripe = [670]  # Brown/black reflectance shift

# Convert pixel positions to wavelengths (Assume linear mapping: adjust factor based on calibration)
pixel_to_wavelength_ratio = 0.5  # Example conversion factor (adjust as needed)
wavelengths = np.array(peaks) * pixel_to_wavelength_ratio

# Determine ripeness based on detected wavelengths
ripeness_score = {"unripe": 0, "ripe": 0, "overripe": 0}

for wl in wavelengths:
    if any(abs(wl - k) < 10 for k in known_unripe):
        ripeness_score["unripe"] += 1
    if any(abs(wl - k) < 10 for k in known_ripe):
        ripeness_score["ripe"] += 1
    if any(abs(wl - k) < 10 for k in known_overripe):
        ripeness_score["overripe"] += 1

# Determine ripeness level
if ripeness_score["ripe"] > 1:
    ripeness_status = "Ripe (Yellow Banana)"
elif ripeness_score["unripe"] > 1:
    ripeness_status = "Unripe (Green Banana)"
elif ripeness_score["overripe"] > 0:
    ripeness_status = "Overripe (Brown/Black Banana)"
else:
    ripeness_status = "Unknown Ripeness"

# Plot the spectrum with detected peaks
plt.figure(figsize=(10, 5))
plt.plot(intensity_profile, label="Intensity Profile")
plt.scatter(peaks, intensity_profile[peaks], color='red', label="Detected Peaks")
plt.xlabel("Pixel Position")
plt.ylabel("Normalized Intensity")
plt.title(f"Banana Ripeness Detection: {ripeness_status}")
plt.legend()
plt.show()

print(f"Detected Ripeness: {ripeness_status}")
