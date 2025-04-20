import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import re  # For pattern matching filenames

# =======================
# Configuration Settings
# =======================

# Serial port configuration - update to match your system
SERIAL_PORT = "COM12"
BAUD_RATE = 115200

# Thresholds for detecting motion gestures
ACCEL_THRESHOLD = 2  # Acceleration threshold in g
GYRO_THRESHOLD = 800  # Gyroscope threshold in degrees/second

# Sampling and recording configuration
SAMPLING_RATE = 100  # Hz, must match IMU/Arduino sampling rate
BUFFER_SIZE = 20  # Pre-trigger buffer size (samples before gesture)
RECORD_DURATION = 1  # Duration of gesture capture in seconds
TOTAL_SAMPLES = int(SAMPLING_RATE * RECORD_DURATION)  # Total post-trigger samples to record

# Directory and label configuration for saving gesture data
GESTURE_DATA_DIR = "gesture_data/bounce"  # Folder to save data
GEST = "bounce"  # Label for the current gesture type

# Ensure data directory exists
if not os.path.exists(GESTURE_DATA_DIR):
    os.makedirs(GESTURE_DATA_DIR)


# =======================
# Helper Functions
# =======================

def get_next_file_number(label):
    """
    Determines the next sequential filename for the gesture being recorded.
    E.g., if 'bounce_3.csv' is the last file, this returns 4.
    """
    files = os.listdir(GESTURE_DATA_DIR)
    pattern = re.compile(f"^{label}_(\\d+)\\.csv$")
    numbers = []

    for file in files:
        match = pattern.match(file)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers, default=0) + 1


def detect_gesture(ax, ay, az, gx, gy, gz):
    """
    Detects whether the current IMU reading exceeds motion thresholds.
    Triggers data recording if significant movement is found.
    """
    accel_magnitude = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyro_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return accel_magnitude > ACCEL_THRESHOLD or gyro_magnitude > GYRO_THRESHOLD


def plot_data(data, label):
    """
    Visualizes the captured accelerometer and gyroscope data.
    Optional for quick inspection of signals.
    """
    data = np.array(data)
    timestamps = data[:, 0] - data[0, 0]  # Normalize time to start from zero
    ax, ay, az = data[:, 1], data[:, 2], data[:, 3]
    gx, gy, gz = data[:, 4], data[:, 5], data[:, 6]

    plt.figure(figsize=(12, 8))

    # Accelerometer data
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, ax, label="AccX")
    plt.plot(timestamps, ay, label="AccY")
    plt.plot(timestamps, az, label="AccZ")
    plt.title(f"Accelerometer Data - Gesture: {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid()

    # Gyroscope data
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, gx, label="GyroX")
    plt.plot(timestamps, gy, label="GyroY")
    plt.plot(timestamps, gz, label="GyroZ")
    plt.title(f"Gyroscope Data - Gesture: {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Rotation Rate (°/s)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# =======================
# Main Data Collection Loop
# =======================

def main():
    # Open serial port to read IMU data
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")

    buffer = []      # Temporary buffer to hold recent data before gesture
    recording = []   # List to store gesture data once detected

    try:
        while True:
            # Read one line of data from serial port
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue  # Skip empty lines

            # Try to parse the IMU data (timestamp, 3x acc, 3x gyro)
            try:
                timestamp, ax, ay, az, gx, gy, gz = map(float, line.split(", "))
            except ValueError:
                print(f"Invalid data: {line}")  # Skip malformed lines
                continue

            # Add the sample to the pre-trigger buffer
            buffer.append([timestamp, ax, ay, az, gx, gy, gz])
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)  # Maintain fixed buffer size

            # Check for gesture initiation
            if detect_gesture(ax, ay, az, gx, gy, gz) and not recording:
                print("Gesture detected! Starting recording...")
                gesture_label = GEST
                recording = buffer.copy()  # Start recording from buffered data

            # Continue recording if a gesture is being tracked
            if recording:
                recording.append([timestamp, ax, ay, az, gx, gy, gz])

                # Stop after recording enough post-trigger samples
                if len(recording) >= TOTAL_SAMPLES + BUFFER_SIZE:
                    print("Recording complete!")

                    # Generate unique filename for the current gesture label
                    file_number = get_next_file_number(gesture_label)
                    filename = os.path.join(GESTURE_DATA_DIR, f"{gesture_label}_{file_number}.csv")

                    # Save data to CSV file
                    np.savetxt(filename, recording, delimiter=",",
                               header="Timestamp, AccX, AccY, AccZ, GyroX, GyroY, GyroZ", comments="")
                    print(f"Data saved to {filename}")

                    # Optional: visualize the recorded gesture
                    # plot_data(recording, f"{gesture_label}_{file_number}")

                    recording = []  # Reset for the next gesture
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()  # Always close the serial port on exit


if __name__ == "__main__":
    main()
