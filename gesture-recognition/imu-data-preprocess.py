import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample, detrend
from sklearn.preprocessing import MinMaxScaler

# ============================
# Configuration Parameters
# ============================

# Paths for raw gesture data
GESTURE_PATHS = {
    "bounce": "gesture_data/bounceData",
    "chop": "gesture_data/chopData",
    "iBounce": "gesture_data/iBounceData"
}

# Uncomment for single gesture batch testing
# GESTURE_PATHS = {
#     "external": "gesture_data/external"
# }

# Output folder to save processed gesture data
OUTPUT_PATH = "preprocessed_data/"

# Number of timesteps per segment (frame/window)
WINDOW_SIZE = 100

# Amount of overlap between windows (e.g., 50% overlap)
OVERLAP = 0.5

# ============================
# Signal Processing Functions
# ============================

def low_pass_filter(data, cutoff=5, fs=50, order=4):
    """
    Apply a low-pass Butterworth filter to remove high-frequency noise.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Normalizer (shared scaler to scale features between -1 and 1)
scaler = MinMaxScaler(feature_range=(-1, 1))

def normalize(data):
    """
    Normalize data to range [-1, 1] using MinMaxScaler.
    """
    return scaler.fit_transform(data)

def segment_data(data, window_size, overlap):
    """
    Divide time-series into overlapping sliding windows.
    """
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        segment = data[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

# ============================
# Data Augmentation Functions
# ============================

def augment_data(data):
    """
    Apply basic data augmentation to increase sample variety.
    Includes noise, scaling, and temporal shift.
    """
    augmented = []

    # Original
    augmented.append(data)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.01, data.shape)
    augmented.append(data + noise)

    # Scale amplitude
    scaling_factor = np.random.uniform(0.9, 1.1)
    augmented.append(data * scaling_factor)

    # Temporal shift
    shift = np.random.randint(1, data.shape[0] // 10)
    shifted_data = np.roll(data, shift, axis=0)
    augmented.append(shifted_data)

    return augmented

# ============================
# Data Cleaning Functions
# ============================

def remove_outliers(data, threshold=3):
    """
    Clip values beyond a z-score threshold to reduce spurious spikes.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = (data - mean) / std
    return np.where(np.abs(z_scores) > threshold, mean, data)

def handle_missing_values(data):
    """
    Interpolate and fill missing values linearly.
    """
    df = pd.DataFrame(data)
    df = df.interpolate(method='linear', axis=0)
    df = df.bfill().ffill()
    return df.values

def remove_drift(data):
    """
    Remove slow signal drift using detrending.
    """
    return detrend(data, axis=0)

def resample_data(data, target_length):
    """
    Optional: Resample time-series to a fixed length.
    """
    return resample(data, target_length, axis=0)

def clean_data(data):
    """
    Apply a full data cleaning pipeline.
    """
    data = remove_outliers(data)
    data = handle_missing_values(data)
    data = remove_drift(data)
    return data

# ============================
# Main Preprocessing Pipeline
# ============================

# Ensure output directory exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def process_gesture_folder(gesture, path):
    """
    Load and preprocess all CSV gesture recordings from a folder.
    Applies cleaning, filtering, normalization, segmentation, and augmentation.
    """
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    all_segments = []

    for file in files:
        file_path = os.path.join(path, file)

        # Read CSV and drop timestamp column
        data = pd.read_csv(file_path)
        imu_data = data.iloc[:, 1:].values  # Keep only IMU values

        # Clean signal (outliers, missing, drift)
        imu_data = clean_data(imu_data)

        # Apply low-pass filtering on each channel
        imu_data = np.array([low_pass_filter(imu_data[:, i]) for i in range(imu_data.shape[1])]).T

        # Normalize values
        imu_data = normalize(imu_data)

        # Segment into windows
        segments = segment_data(imu_data, WINDOW_SIZE, OVERLAP)

        # Augment each segment and collect
        augmented_segments = []
        for segment in segments:
            augmented_segments.extend(augment_data(segment))

        all_segments.extend(augmented_segments)

    # Save all processed data as .npy file
    np.save(os.path.join(OUTPUT_PATH, f"{gesture}_preprocessed.npy"), np.array(all_segments))


# Process all gesture folders
for gesture, path in GESTURE_PATHS.items():
    process_gesture_folder(gesture, path)

print("Data preprocessing and augmentation completed.")
