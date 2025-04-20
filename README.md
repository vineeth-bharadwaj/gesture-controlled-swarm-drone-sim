# 🎯 Project Overview

This project presents a proof of concept for a gesture-controlled drone swarm simulation, showcasing intuitive human-swarm interaction through real-time hand gestures. Developed using PyGame and OpenGL, the system offers a smooth and visually accurate 3D simulation of drone swarm dynamics in response to custom control inputs. Gesture data is captured via an Arduino Nano 33 IoT equipped with an onboard LSM6DS3 IMU sensor, with hand tilts mapped to flight control parameters—Throttle, Yaw, Pitch, and Roll. Recognized gestures trigger predefined swarm coordination modes—March, Scanner, and Anchor—each emulating behaviours useful in real-world applications such as surveillance, area scanning, and collaborative motion. The simulation demonstrates high responsiveness, low latency, and reliable gesture recognition, validating the system’s design. This proof of concept establishes the viability of using natural human gestures to control autonomous drone swarms. Future enhancements include integrating full physics modelling, real-time stabilization logic, and migrating gesture processing to onboard microcontrollers for real world deployment

---

## Code Files

### ─ `imu-data-collect.py`
- Continuously reads IMU data from the serial port.
- Waits for a gesture that crosses predefined thresholds.
- Starts recording when a gesture is detected and captures data for a fixed duration.
- Automatically saves each gesture sample to a uniquely named CSV file.
- Supports pre-trigger buffering (so it captures a bit of data just before the gesture starts).
- Optional plotting is available to visually inspect the data.

### ─ `imu-data-preprocess.py`
- Reads raw gesture CSV files (no need to manually label files).
- Cleans data: removes outliers, fills gaps, and removes drift.
- Filters and normalizes signals.
- Splits data into sliding time windows (segments).
- Augments each window using noise, scaling, and shifting.
- Saves the final dataset as NumPy arrays, ready for model training.

### ─ `model-training.py`
- Loads the preprocessed data that was saved as a .npy file, for each gesture
- Aggregates all the data together with appropriate labelling and performs a train-test split
- Uses an RNN model with BiLSTM layers to train with SGD optimizers
- Trains in 10 epochs and performs evaluation of model
  
### ─ `real-time-gesture-recognition.py`
- Continuously reads IMU data from the serial port.
- Waits for a gesture that crosses predefined thresholds.
- Starts recording when a gesture is detected and captures data for a fixed duration.
- Preprocesses the recorded data in the way needed by the model
- Feeds the preprocessed data to the model for predicting the recorded gesture signal

### ─ `swarm-drone-simulator.py`
- Uses PyGame and OpenGL for simulation of the swarm drones
- Integrates the Gesture Recognition code for using gesture recognition to control swarm drone coordination
- Arduino code (arduino-nano-33-iot-control.ino) directly converts the tilt of the hand to throttle, roll, pitch and yaw values for the control
- On pressing G, the python code requests raw IMU values for gesture recogniton. Pressing G again will switch back to requesting throttle, roll, pitch and yaw values
- Currently uses three gesture for 3 coordination modes - March, Scanner and Anchor
  
### ─ `gesture_rnn_bilstm_model.keras`
- **Purpose:** Pre-trained Keras model for gesture classification.
- **Details:**
  - BiLSTM architecture
  - Input: IMU sequence of shape `(100, 6)`
  - Output: Gesture class

!(images/gestures3.png)

### ─ `README.md`
- You're here!
