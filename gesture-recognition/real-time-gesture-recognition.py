import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, detrend
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
import re  # For pattern matching filenames

# ==========================================
# Configuration
# ==========================================

# Serial port settings (adjust for your system)
SERIAL_PORT = "COM12"     # Replace with your serial port (e.g., "COM3" or "/dev/ttyUSB0")
BAUD_RATE = 115200

# Gesture detection thresholds
ACCEL_THRESHOLD = 2       # Acceleration magnitude threshold (in g)
GYRO_THRESHOLD = 800      # Gyroscope magnitude threshold (in deg/s)

# Sampling settings
SAMPLING_RATE = 100       # Hz (must match Arduino)
RECORD_DURATION = 1       # Record duration after trigger (in seconds)
TOTAL_SAMPLES = int(SAMPLING_RATE * RECORD_DURATION)

# Buffer to store pre-gesture data
BUFFER_SIZE = 20          # Number of samples to keep before gesture trigger

# Segmentation settings
WINDOW_SIZE = 100         # Number of timesteps per segment
OVERLAP = 0.5             # Overlap ratio between segments (50%)

# Load the trained model
model = load_model("gesture_rnn_bilstm_model.keras")

# Initialize scaler for normalization
scaler = MinMaxScaler(feature_range=(-1, 1))

# ==========================================
# Gesture Detection
# ==========================================

def detect_gesture(ax, ay, az, gx, gy, gz):
    """Detect significant movement based on combined accelerometer and gyroscope magnitude."""
    accel_magnitude = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyro_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return accel_magnitude > ACCEL_THRESHOLD or gyro_magnitude > GYRO_THRESHOLD

# ==========================================
# Data Cleaning & Preprocessing Functions
# ==========================================

def low_pass_filter(data, cutoff=5, fs=50, order=4):
    """Apply a low-pass Butterworth filter to reduce noise."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def normalize(data):
    """Normalize data to the range (-1, 1)."""
    return scaler.fit_transform(data)

def remove_outliers(data, threshold=3):
    """Clip values beyond the threshold to mean value."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = (data - mean) / std
    return np.where(np.abs(z_scores) > threshold, mean, data)

def handle_missing_values(data):
    """Fill missing values using interpolation and forward/backward fill."""
    df = pd.DataFrame(data)
    df = df.interpolate(method='linear', axis=0).bfill().ffill()
    return df.values

def remove_drift(data):
    """Remove linear trend from the signal (zero-mean correction)."""
    return detrend(data, axis=0)

def resample_data(data, target_length):
    """Resample signal to a fixed length (not used directly here)."""
    return resample(data, target_length, axis=0)

def segment_data(data, window_size, overlap):
    """Segment data into overlapping windows for model input."""
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        segment = data[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

def clean_data(data):
    """Apply all cleaning steps to IMU data."""
    data = remove_outliers(data)
    data = handle_missing_values(data)
    data = remove_drift(data)
    return data

# ==========================================
# Main Loop for Real-Time Gesture Recognition
# ==========================================

def main():
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")

    buffer = []      # Pre-trigger data
    recording = []   # Active recording buffer

    try:
        while True:
            # Read line from serial
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue

            # Parse comma-separated IMU data
            try:
                timestamp, ax, ay, az, gx, gy, gz = map(float, line.split(", "))
            except ValueError:
                print(f"Invalid data: {line}")
                continue

            # Maintain rolling buffer of recent data
            buffer.append([timestamp, ax, ay, az, gx, gy, gz])
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)

            # Start recording if gesture is detected
            if detect_gesture(ax, ay, az, gx, gy, gz) and not recording:
                print("Gesture detected! Starting recording...")
                recording = buffer.copy()  # Start with buffered pre-trigger data

            # Continue recording after gesture trigger
            if recording:
                recording.append([timestamp, ax, ay, az, gx, gy, gz])

                # Stop recording once desired duration is reached
                if len(recording) >= TOTAL_SAMPLES + BUFFER_SIZE:
                    print("Recording complete!")

                    # Extract IMU data and drop timestamps
                    rec_data = pd.DataFrame(recording, columns=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
                    imu_data = rec_data.iloc[:, 1:].values

                    # Clean and preprocess
                    imu_data = clean_data(imu_data)
                    imu_data = np.array([low_pass_filter(imu_data[:, i]) for i in range(imu_data.shape[1])]).T
                    imu_data = normalize(imu_data)

                    # Segment and predict
                    segments = segment_data(imu_data, WINDOW_SIZE, OVERLAP)
                    imu_data_p = np.array(segments)
                    y_pred_class = np.argmax(model.predict(imu_data_p), axis=1)
                    print("Predicted gesture class:", y_pred_class)

                    # Reset recording buffer
                    recording = []

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    main()
