import socket
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, detrend
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# WiFi server configuration (this device will act as the server)
HOST = "0.0.0.0"  # Listen on all available network interfaces
PORT = 5000       # Must match the port used by the sender

# Thresholds for gesture detection
ACCEL_THRESHOLD = 2     # Acceleration magnitude threshold (in g)
GYRO_THRESHOLD = 800    # Gyroscope magnitude threshold (in degrees/s)

# Sampling configuration
SAMPLING_RATE = 100       # Hz – must match microcontroller sampling rate
BUFFER_SIZE = 20          # Number of samples to retain before gesture trigger
RECORD_DURATION = 1       # Seconds of data to record after gesture is detected
TOTAL_SAMPLES = int(SAMPLING_RATE * RECORD_DURATION)

# Sequence preprocessing configuration
WINDOW_SIZE = 100   # Sliding window size (number of time steps per input to model)
OVERLAP = 0.5       # 50% overlap between consecutive windows

# Load the trained gesture recognition model
model = load_model("gesture_rnn_bilstm_model.keras")

# Scaler for feature normalization
scaler = MinMaxScaler(feature_range=(-1, 1))

# Detect gesture based on magnitude of acceleration and gyroscope data
def detect_gesture(ax, ay, az, gx, gy, gz):
    accel_magnitude = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyro_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return accel_magnitude > ACCEL_THRESHOLD or gyro_magnitude > GYRO_THRESHOLD

# Apply a low-pass Butterworth filter to remove high-frequency noise
def low_pass_filter(data, cutoff=5, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Normalize data using MinMaxScaler
def normalize(data):
    return scaler.fit_transform(data)

# Divide the data into overlapping windows
def segment_data(data, window_size, overlap):
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        segment = data[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

# Remove statistical outliers based on z-score
def remove_outliers(data, threshold=3):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = (data - mean) / std
    return np.where(np.abs(z_scores) > threshold, mean, data)

# Fill in missing or NaN values
def handle_missing_values(data):
    df = pd.DataFrame(data)
    df = df.interpolate(method='linear', axis=0).bfill().ffill()
    return df.values

# Remove linear drift from signal
def remove_drift(data):
    return detrend(data, axis=0)

# Resample the data to a target length (optional utility)
def resample_data(data, target_length):
    return resample(data, target_length, axis=0)

# Clean IMU data: outliers, missing values, and drift
def clean_data(data):
    data = remove_outliers(data)
    data = handle_missing_values(data)
    data = remove_drift(data)
    return data

# === Main Program ===
def main():
    # Create a TCP socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Waiting for connection on {HOST}:{PORT}...")

    # Accept a single client connection
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    buffer = []      # Stores pre-trigger samples
    recording = []   # Active recording buffer for gesture

    try:
        data_buffer = ""  # For handling split/partial TCP messages

        while True:
            data = conn.recv(1024).decode()  # Receive a chunk of data
            if not data:
                break

            data_buffer += data  # Append to ongoing buffer

            while "\n" in data_buffer:
                # Process each complete line
                line, data_buffer = data_buffer.split("\n", 1)

                try:
                    ax, ay, az, gx, gy, gz = map(float, line.strip().split(","))
                except ValueError:
                    print(f"Invalid data: {line}")
                    continue

                # Add to buffer (pre-trigger window)
                buffer.append([ax, ay, az, gx, gy, gz])
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)

                # Detect gesture start
                if detect_gesture(ax, ay, az, gx, gy, gz) and not recording:
                    print("Gesture detected! Starting recording...")
                    recording = buffer.copy()

                # Continue recording until desired length
                if recording:
                    recording.append([ax, ay, az, gx, gy, gz])

                    if len(recording) >= TOTAL_SAMPLES + BUFFER_SIZE:
                        print("Recording complete!")

                        # Convert to DataFrame and extract only IMU values
                        rec_data = pd.DataFrame(recording, columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
                        imu_data = rec_data.values

                        # Clean and preprocess data
                        imu_data = clean_data(imu_data)
                        imu_data = np.array([low_pass_filter(imu_data[:, i]) for i in range(imu_data.shape[1])]).T
                        imu_data = normalize(imu_data)

                        # Segment into overlapping windows for model input
                        segments = segment_data(imu_data, WINDOW_SIZE, OVERLAP)
                        imu_data_p = np.array(segments)

                        # Run prediction
                        y_pred_class = np.argmax(model.predict(imu_data_p), axis=1)
                        print("Predicted class:", y_pred_class[0])

                        # Send result back to client
                        conn.sendall((str(y_pred_class[0]) + "\n").encode())

                        # Reset recording buffer
                        recording = []

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
    finally:
        conn.close()
        server_socket.close()
        print("Server closed.")

if __name__ == "__main__":
    main()
