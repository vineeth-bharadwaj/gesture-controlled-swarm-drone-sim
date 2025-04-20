import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import serial
import time
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample
from scipy.signal import detrend
from tensorflow.keras.models import load_model
import os
import re

ARDUINO_PORT = 'COM9'
BAUD_RATE = 115200
arduino_connected = False
ser = None

BAUD_RATE = 115200

# Thresholds for gesture detection
ACCEL_THRESHOLD = 2  # Adjust as needed (in g)
GYRO_THRESHOLD = 800  # Adjust as needed (in degrees/s)

# Sampling and recording configuration
SAMPLING_RATE = 100  # Hz (must match Arduino sampling rate)
BUFFER_SIZE = 20  # Number of samples to include before the gesture
RECORD_DURATION = 1  # Duration to record in seconds
TOTAL_SAMPLES = int(SAMPLING_RATE * RECORD_DURATION)

WINDOW_SIZE = 100  # Number of timesteps per sequence
OVERLAP = 0.5  # 50% overlap

model = load_model("gesture_rnn_bilstm_model.keras")

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))


def check_safe_zone(swarm_drones, drone_size=2.6, safety_factor=1.0):
    # Create a copy to modify
    modified_drones = swarm_drones.copy()

    # Calculate actual safe distance based on drone size and safety factor
    safe_distance = drone_size * safety_factor

    for i in range(len(swarm_drones)):
        for j in range(i + 1, len(swarm_drones)):
            # Calculate 3D distance between drones
            distance = np.sqrt(
                (swarm_drones[i]['pos_x'] - swarm_drones[j]['pos_x']) ** 2 +
                (swarm_drones[i]['pos_y'] - swarm_drones[j]['pos_y']) ** 2 +
                (swarm_drones[i]['pos_z'] - swarm_drones[j]['pos_z']) ** 2
            )

            # If drones are too close, adjust their velocities
            if distance < safe_distance:
                # Calculate repulsion vector
                repulsion_x = (swarm_drones[i]['pos_x'] - swarm_drones[j]['pos_x']) / (distance + 0.1)
                repulsion_y = (swarm_drones[i]['pos_y'] - swarm_drones[j]['pos_y']) / (distance + 0.1)
                repulsion_z = (swarm_drones[i]['pos_z'] - swarm_drones[j]['pos_z']) / (distance + 0.1)

                # Apply repulsion to velocities
                # Increased repulsion strength to be more responsive
                repulsion_strength = 0.2  # Adjusted from 0.1

                modified_drones[i]['vel_x'] += repulsion_x * repulsion_strength
                modified_drones[i]['vel_y'] += repulsion_y * repulsion_strength
                modified_drones[i]['vel_z'] += repulsion_z * repulsion_strength

                modified_drones[j]['vel_x'] -= repulsion_x * repulsion_strength
                modified_drones[j]['vel_y'] -= repulsion_y * repulsion_strength
                modified_drones[j]['vel_z'] -= repulsion_z * repulsion_strength

    return modified_drones

def detect_gesture(ax, ay, az, gx, gy, gz):
    """Detect significant movement based on thresholds."""
    accel_magnitude = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyro_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return accel_magnitude > ACCEL_THRESHOLD or gyro_magnitude > GYRO_THRESHOLD

# Low-pass filter
def low_pass_filter(data, cutoff=5, fs=50, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def normalize(data):
    return scaler.fit_transform(data)

# Data segmentation
def segment_data(data, window_size, overlap):
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        segment = data[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

def remove_outliers(data, threshold=3):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = (data - mean) / std
    return np.where(np.abs(z_scores) > threshold, mean, data)

def handle_missing_values(data):
    df = pd.DataFrame(data)
    df = df.interpolate(method='linear', axis=0)
    df = df.bfill().ffill()
    return df.values

def remove_drift(data):
    return detrend(data, axis=0)

def resample_data(data, target_length):
    return resample(data, target_length, axis=0)

# Cleaning functions
def clean_data(data):
    # Remove outliers
    data = remove_outliers(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Remove drift
    data = remove_drift(data)

    return data

try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
    arduino_connected = True
    print("Arduino connected successfully")
    time.sleep(2)  # Give the connection time to stabilize
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")
    print("Running in keyboard control mode")

def map_pulse_to_acceleration(pulse_ms, min_accel=-0.02, max_accel=0.02):
    """
    Map pulse width value (1ms to 2ms) to acceleration value
    1.5ms is neutral (no acceleration)
    """
    if pulse_ms < 1.0:
        pulse_ms = 1.0
    elif pulse_ms > 2.0:
        pulse_ms = 2.0

    # Map from [1.0, 2.0] to [-0.02, 0.02]
    return min_accel + (pulse_ms - 1.0) * (max_accel - min_accel)

def read_arduino_values(imu_mode=False):
    """
    Read and parse values from Arduino
    Expected format:
    - Normal mode: "T:1.5,R:1.5,P:1.5,Y:1.5"
    - IMU mode: "AX:value,AY:value,AZ:value,GX:value,GY:value,GZ:value"

    Args:
        imu_mode (bool): Flag to switch between normal control and IMU data reading

    Returns:
        dict or None: Parsed Arduino values or None if reading fails
    """
    if not arduino_connected or ser is None:
        return None

    try:
        # Clear any previous buffer to prevent stale data
        ser.reset_input_buffer()

        if imu_mode:
            # Send IMU data request
            ser.write(b'i')  # 'i' for IMU data
        else:
            # Reset to normal mode and request data
            ser.write(b'x')  # 'x' to reset modes
            ser.write(b'g')  # 'g' for control data


        line = ser.readline().decode('utf-8').strip()
        if not line:
            return None

        if imu_mode:
            # Return raw IMU values if in IMU mode
            return line
        else:
            values = {}
            parts = line.split(',')
            for part in parts:
                if ':' in part:
                    key, value = part.split(':')
                    try:
                        values[key] = float(value)
                    except ValueError:
                        pass
            # Normal control mode
            if 'T' in values and 'R' in values and 'P' in values and 'Y' in values:
                return {
                    'throttle': values['T'],
                    'roll': values['R'],
                    'pitch': values['P'],
                    'yaw': values['Y']
                }
    except Exception as e:
        print(f"Error reading from Arduino: {e}")

    return None

def draw_drone(throttle, tilt_x, tilt_z):
    # Throttle to motor speed calculation
    def calculate_motor_speed(base_throttle, tilt_adjustment):
        # Stop motors completely if throttle is at minimum
        if base_throttle <= 1.0:
            return 0

        # Base rotation speed scaled from throttle
        base_speed = max(abs(base_throttle - 1.5) * 40, 10)

        # Tilt adjustment: reduce speed of motors on the higher side
        tilt_speed_reduction = abs(tilt_adjustment) * 0.5

        return base_speed

    # Propeller positions and base rotation directions
    propeller_configs = [
        (0.8, 0.8, 1, -1),  # Front-right, clockwise, roll right adjustment
        (-0.8, 0.8, -1, 1),  # Front-left, counter-clockwise, roll left adjustment
        (0.8, -0.8, -1, -1),  # Back-right, counter-clockwise, pitch forward adjustment
        (-0.8, -0.8, 1, 1)  # Back-left, clockwise, pitch back adjustment
    ]

    # Drone body (main frame)
    glColor3f(0, 0, 0)
    glPushMatrix()
    glScalef(0.75, 0.1, 0.75)
    draw_cube()
    glPopMatrix()

    # Drone landing stands
    glColor3f(0, 0, 0)  # Gray color for stands
    stand_positions = [
        (0.7, -0.2, 0.7),  # Front-right
        (-0.7, -0.2, 0.7),  # Front-left
        (0.7, -0.2, -0.7),  # Back-right
        (-0.7, -0.2, -0.7)  # Back-left
    ]

    # Draw landing stands
    for x, y, z in stand_positions:
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(0.05, 0.3, 0.05)  # Thin, tall rectangular stand
        draw_cube(line_width=2)
        glPopMatrix()

    # Propeller rendering function
    def draw_propeller(throttle, roll_tilt, pitch_tilt, base_direction, tilt_direction):
        # Calculate motor speed with tilt considerations
        # Roll tilt: affects left-right motor speeds
        # Pitch tilt: affects front-back motor speeds
        roll_adjustment = roll_tilt * tilt_direction
        pitch_adjustment = pitch_tilt * tilt_direction

        # Combine roll and pitch adjustments
        total_tilt_adjustment = abs(roll_adjustment) + abs(pitch_adjustment)

        # Calculate base speed
        base_speed = calculate_motor_speed(throttle, total_tilt_adjustment)

        # Color variation based on speed
        intensity = min(base_speed / 30, 1.0) + 0.1
        glColor3f(0.86 * intensity, 0.2 * intensity, 0.05 * intensity)

        # Propeller blade rendering
        glPushMatrix()
        glRotatef(pygame.time.get_ticks() * base_speed * base_direction, 0, 1, 0)

        # Draw propeller blades
        for angle in [0, 120, 240]:
            glPushMatrix()
            glRotatef(angle, 0, 1, 0)
            glBegin(GL_TRIANGLES)
            glVertex3f(0, 0, 0)
            glVertex3f(0.5, 0.1, 0)
            glVertex3f(0.5, -0.1, 0)
            glEnd()
            # Draw outline
            glLineWidth(2)
            glBegin(GL_LINE_LOOP)
            glVertex3f(0, 0, 0)
            glVertex3f(0.5, 0.1, 0)
            glVertex3f(0.5, -0.1, 0)
            glEnd()
            glPopMatrix()

        glPopMatrix()

    # Render propellers
    for x, z, direction, tilt_direction in propeller_configs:
        glPushMatrix()
        glTranslatef(x, 0.2, z)  # Slightly raised from the body
        glScalef(0.4, 0.4, 0.4)
        draw_propeller(throttle, tilt_z, tilt_x, direction, tilt_direction)
        glPopMatrix()

    # Drone leg/motor mounts
    glColor3f(0, 0, 1)
    for x, z in [(0.8, 0.8), (-0.8, 0.8), (0.8, -0.8), (-0.8, -0.8)]:
        glPushMatrix()
        glTranslatef(x, 0.1, z)
        glScalef(0.4, 0.05, 0.4)
        draw_cube(line_width=2)
        glPopMatrix()

def draw_cube(scale=1, line_width=1):
    vertices = [[-1 * scale, -1 * scale, -1 * scale],
                [1 * scale, -1 * scale, -1 * scale],
                [1 * scale, 1 * scale, -1 * scale],
                [-1 * scale, 1 * scale, -1 * scale],
                [-1 * scale, -1 * scale, 1 * scale],
                [1 * scale, -1 * scale, 1 * scale],
                [1 * scale, 1 * scale, 1 * scale],
                [-1 * scale, 1 * scale, 1 * scale]]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    glLineWidth(line_width)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_target(position):
    glPushMatrix()
    glTranslatef(*position)
    glColor3f(0.0, 1.0, 0.0)
    glScalef(0.5, 0.5, 0.5)
    draw_cube()
    glPopMatrix()

def load_texture(filename):
    """
    Load an image texture for OpenGL rendering
    """
    try:
        # Open the image using Pillow
        img = Image.open(filename)

        # Convert image to RGBA format for consistent texture handling
        img = img.convert("RGBA")

        # Get image dimensions
        width, height = img.size

        # Convert image to byte array
        img_data = img.tobytes("raw", "RGBA")

        # Generate and bind a new texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # Generate the texture with mipmaps
        gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, width, height,
                          GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        return texture, width, height

    except Exception as e:
        print(f"Error loading texture: {e}")
        return None, 0, 0

def draw_textured_ground(ground_size=50, grid_spacing=2, ground_level=-3, texture=None):
    """
    Draw a textured ground surface with optional grid lines
    """
    if texture is None:
        return draw_ground()  # Fallback to original ground drawing

    # Bind the texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Draw textured ground surface
    glColor3f(1, 1, 1)  # Use white to preserve texture colors
    glBegin(GL_QUADS)
    # Use (0,0) to (1,1) texture coordinates to stretch the entire image
    glTexCoord2f(0, 0); glVertex3f(-ground_size, ground_level, -ground_size)
    glTexCoord2f(1, 0); glVertex3f(ground_size, ground_level, -ground_size)
    glTexCoord2f(1, 1); glVertex3f(ground_size, ground_level, ground_size)
    glTexCoord2f(0, 1); glVertex3f(-ground_size, ground_level, ground_size)
    glEnd()

    # Disable texture mapping
    glDisable(GL_TEXTURE_2D)

    # Optional: draw grid lines on top of texture
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.0, 0.0, 0.0, 0.1)
    glLineWidth(1.0)
    glBegin(GL_LINES)

    # Vertical lines
    for x in range(-int(ground_size), int(ground_size) + 1, grid_spacing):
        glVertex3f(x, ground_level, -ground_size)
        glVertex3f(x, ground_level, ground_size)

    # Horizontal lines
    for z in range(-int(ground_size), int(ground_size) + 1, grid_spacing):
        glVertex3f(-ground_size, ground_level, z)
        glVertex3f(ground_size, ground_level, z)

    glEnd()

def draw_ground():
    # Expanded ground area (much larger than before)
    ground_size = 50  # Increased from 20 to 50
    grid_spacing = 2  # Size of grid squares
    ground_level = -3  # Consistent with original ground level

    # Base ground color (slightly greenish)
    base_color = (0.4, 0.6, 0.4)

    # Draw solid ground surface
    glColor3f(*base_color)
    glBegin(GL_QUADS)
    glVertex3f(-ground_size, ground_level, -ground_size)
    glVertex3f(ground_size, ground_level, -ground_size)
    glVertex3f(ground_size, ground_level, ground_size)
    glVertex3f(-ground_size, ground_level, ground_size)
    glEnd()

    # Draw grid lines
    glColor3f(0.3, 0.5, 0.3)  # Slightly darker green for grid lines
    glLineWidth(1.0)
    glBegin(GL_LINES)

    # Vertical lines
    for x in range(-int(ground_size), int(ground_size) + 1, grid_spacing):
        glVertex3f(x, ground_level, -ground_size)
        glVertex3f(x, ground_level, ground_size)

    # Horizontal lines
    for z in range(-int(ground_size), int(ground_size) + 1, grid_spacing):
        glVertex3f(-ground_size, ground_level, z)
        glVertex3f(ground_size, ground_level, z)

    glEnd()

def ease_movement(current, target, factor=0.08):
    return current + (target - current) * factor

def main():
    pygame.init()
    display = (1280, 720)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Swarm Drone Simulation with Arduino Control")
    pygame.font.init()

    ground_texture, tex_width, tex_height = load_texture('map1.png')

    gluPerspective(100, (display[0] / display[1]), 0.1, 400.0)
    glTranslatef(0.0, 0.0, -10)

    # Initialize swarm of 3 drones in a line
    swarm_drones = [
        {
            'pos_x': -10, 'pos_y': -2.7, 'pos_z': 0,
            'vel_x': 0, 'vel_y': 0, 'vel_z': 0,
            'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
            'tilt_x': 0, 'tilt_z': 0,
            'rotation_y': 0,
            'throttle': 1.5, 'roll': 1.5, 'pitch': 1.5, 'yaw': 1.5
        },
        {
            'pos_x': 0, 'pos_y': -2.7, 'pos_z': 0,
            'vel_x': 0, 'vel_y': 0, 'vel_z': 0,
            'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
            'tilt_x': 0, 'tilt_z': 0,
            'rotation_y': 0,
            'throttle': 1.5, 'roll': 1.5, 'pitch': 1.5, 'yaw': 1.5
        },
        {
            'pos_x': 10, 'pos_y': -2.7, 'pos_z': 0,
            'vel_x': 0, 'vel_y': 0, 'vel_z': 0,
            'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
            'tilt_x': 0, 'tilt_z': 0,
            'rotation_y': 0,
            'throttle': 1.5, 'roll': 1.5, 'pitch': 1.5, 'yaw': 1.5
        }
    ]

    target_pos = (random.uniform(-5, 5), random.uniform(-2, 2), random.uniform(-8, -3))
    clock = pygame.time.Clock()
    running = True
    GRAVITY = 0.01  # Increased gravity constant
    HOVER_THRESHOLD = 1.2  # Throttle value below which gravity becomes dominant
    GROUND_LEVEL = -2.4
    GROUND_BOUNCE_DAMPING = 0
    max_speed = 1.0
    drag = 0.98

    gesture_recognition_mode = False
    current_gesture = 0
    last_g_press_time = 0
    G_PRESS_COOLDOWN = 500

    buffer = []  # Circular buffer for pre-trigger data
    recording = []  # List to store recorded data

    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_g and current_time - last_g_press_time > G_PRESS_COOLDOWN:
                    # Toggle gesture recognition mode
                    gesture_recognition_mode = not gesture_recognition_mode
                    last_g_press_time = current_time

                    if gesture_recognition_mode:
                        print("Gesture Recognition Mode: Enabled")
                    else:
                        print("Gesture Recognition Mode: Disabled")

        # Gesture Recognition Logic
        if gesture_recognition_mode:
            # Request IMU data and do gesture recognition
            arduino_values = read_arduino_values(imu_mode=True)

            if arduino_values:
                # Parse the IMU data
                try:
                    timestamp, ax, ay, az, gx, gy, gz = map(float, arduino_values.split(", "))
                except ValueError:
                    print(f"Invalid data: {arduino_values}")
                    continue

                # Add data to the buffer
                buffer.append([timestamp, ax, ay, az, gx, gy, gz])
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)

                # Check for a gesture
                if detect_gesture(ax, ay, az, gx, gy, gz) and not recording:
                    print("Gesture detected! Starting recording...")
                    recording = buffer.copy()  # Start with pre-trigger buffer

                # Continue recording if a gesture is detected
                if recording:
                    recording.append([timestamp, ax, ay, az, gx, gy, gz])

                    # Stop recording after the desired duration
                    if len(recording) >= TOTAL_SAMPLES + BUFFER_SIZE:
                        print("Recording complete!")
                        rec_data = pd.DataFrame(recording, columns=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
                        imu_data = rec_data.iloc[:, 1:].values  # Exclude Timestamp

                        # Data cleaning
                        imu_data = clean_data(imu_data)

                        # Filtering and normalization
                        imu_data = np.array([low_pass_filter(imu_data[:, i]) for i in range(imu_data.shape[1])]).T
                        imu_data = normalize(imu_data)

                        # Segmentation
                        segments = segment_data(imu_data, WINDOW_SIZE, OVERLAP)

                        imu_data_p = np.array(segments)
                        y_pred_class = np.argmax(model.predict(imu_data_p), axis=1)
                        print(y_pred_class)
                        current_gesture = y_pred_class[0]

                        recording = []
        else:
            # Use standard Arduino control or keyboard control
            arduino_values = read_arduino_values()

        count = 0

        # Update drone controls
        swarm_drones = check_safe_zone(swarm_drones)
        for drone in swarm_drones:
            # Use Arduino values if available, otherwise use keyboard
            if arduino_values:
                if gesture_recognition_mode:
                    # In IMU mode, keep all controls neutral
                    drone['throttle'] = 1.5
                    drone['roll'] = 1.5
                    drone['pitch'] = 1.5
                    drone['yaw'] = 1.5

                    # Optional: You might want to log IMU values here
                    #print("IMU Values:", arduino_values)
                else:
                    if current_gesture == 1:
                        if count==0:
                            throttle_mod = arduino_values['throttle']
                            roll_mod = 3-arduino_values['roll']
                            pitch_mod = arduino_values['pitch']
                            yaw_mod = arduino_values['yaw']
                        if count == 1:
                            throttle_mod = arduino_values['throttle']
                            roll_mod = 1.5
                            pitch_mod = arduino_values['pitch']
                            yaw_mod = arduino_values['yaw']
                        if count == 2:
                            throttle_mod = arduino_values['throttle']
                            roll_mod = arduino_values['roll']
                            pitch_mod = arduino_values['pitch']
                            yaw_mod = arduino_values['yaw']
                    elif current_gesture == 2:
                        if count==0:
                            throttle_mod = arduino_values['throttle']
                            roll_mod = 3-arduino_values['roll']
                            pitch_mod = 3-arduino_values['pitch']
                            yaw_mod = arduino_values['yaw']
                        if count == 1:
                            throttle_mod = arduino_values['throttle']
                            roll_mod = 1.5
                            pitch_mod = 1.5
                            yaw_mod = arduino_values['yaw']
                        if count == 2:
                            throttle_mod = arduino_values['throttle']
                            roll_mod = arduino_values['roll']
                            pitch_mod = arduino_values['pitch']
                            yaw_mod = arduino_values['yaw']
                    else:
                        throttle_mod = arduino_values['throttle']
                        roll_mod = arduino_values['roll']
                        pitch_mod = arduino_values['pitch']
                        yaw_mod = arduino_values['yaw']
                    drone['throttle'] = throttle_mod
                    drone['roll'] = roll_mod
                    drone['pitch'] = pitch_mod
                    drone['yaw'] = yaw_mod
                    count+=1
            else:
                # Keyboard control (same as before, but simpler)
                keys = pygame.key.get_pressed()

                # Throttle
                if keys[K_SPACE]:
                    drone['throttle'] = min(drone['throttle'] + 0.01, 2.0)
                elif keys[K_LSHIFT]:
                    drone['throttle'] = max(drone['throttle'] - 0.01, 1.0)
                else:
                    drone['throttle'] = ease_movement(drone['throttle'], 1.5, 0.05)

                # Pitch
                if keys[K_w]:
                    drone['pitch'] = min(drone['pitch'] + 0.01, 2.0)
                elif keys[K_s]:
                    drone['pitch'] = max(drone['pitch'] - 0.01, 1.0)
                else:
                    drone['pitch'] = ease_movement(drone['pitch'], 1.5, 0.05)

                # Roll
                if keys[K_d]:
                    drone['roll'] = min(drone['roll'] + 0.01, 2.0)
                elif keys[K_a]:
                    drone['roll'] = max(drone['roll'] - 0.01, 1.0)
                else:
                    drone['roll'] = ease_movement(drone['roll'], 1.5, 0.05)

                # Yaw
                if keys[K_q]:
                    drone['yaw'] = min(drone['yaw'] + 0.01, 2.0)
                elif keys[K_e]:
                    drone['yaw'] = max(drone['yaw'] - 0.01, 1.0)
                else:
                    drone['yaw'] = ease_movement(drone['yaw'], 1.5, 0.05)

            # Map pulse values to acceleration (same for all drones)
            if drone['throttle'] <= HOVER_THRESHOLD:
                drone['accel_y'] = -GRAVITY
                drone['accel_x'] = 0
                drone['accel_z'] = 0
                yaw_rate = 0
            else:
                drone['accel_y'] = map_pulse_to_acceleration(drone['throttle'], min_accel=0, max_accel=0.05) - 0.025
                drone['accel_x'] = map_pulse_to_acceleration(drone['roll']) * 1.5
                drone['accel_z'] = map_pulse_to_acceleration(drone['pitch'], max_accel=-0.03, min_accel=0.03) * 1.5
                yaw_rate = (drone['yaw'] - 1.5) * 0.8  # Increased yaw rotation speed

            # Update velocity with acceleration and drag
            drone['vel_x'] = (drone['vel_x'] + drone['accel_x']) * drag
            drone['vel_y'] = (drone['vel_y'] + drone['accel_y']) * drag
            drone['vel_z'] = (drone['vel_z'] + drone['accel_z']) * drag

            # Apply speed limits
            drone['vel_x'] = max(min(drone['vel_x'], max_speed), -max_speed)
            drone['vel_y'] = max(min(drone['vel_y'], max_speed), -max_speed)
            drone['vel_z'] = max(min(drone['vel_z'], max_speed), -max_speed)

            # Ground collision detection and response
            if drone['pos_y'] <= GROUND_LEVEL:
                drone['pos_y'] = GROUND_LEVEL
                if drone['vel_y'] < 0:
                    drone['vel_y'] = -drone['vel_y'] * GROUND_BOUNCE_DAMPING
                drone['vel_x'] *= 0.9
                drone['vel_z'] *= 0.9

            # Update position
            drone['pos_x'] += drone['vel_x']
            drone['pos_y'] += drone['vel_y']
            drone['pos_z'] += drone['vel_z']
            drone['rotation_y'] += yaw_rate

            # Calculate tilt based on speed
            target_tilt_x = drone['vel_z'] * 30
            target_tilt_z = -drone['vel_x'] * 30
            drone['tilt_x'] = ease_movement(drone['tilt_x'], target_tilt_x, 0.1)
            drone['tilt_z'] = ease_movement(drone['tilt_z'], target_tilt_z, 0.1)

        # Check for target collection (simplified)
        for drone in swarm_drones:
            target_distance = ((drone['pos_x'] - target_pos[0]) ** 2 +
                               (drone['pos_y'] - target_pos[1]) ** 2 +
                               (drone['pos_z'] - target_pos[2]) ** 2) ** 0.5
            if target_distance < 2.0:
                target_pos = (random.uniform(-5, 5), random.uniform(-2, 2), random.uniform(-8, -3))

        # Render scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Render Left View (Third-Person)
        glViewport(5, 5, 1025, 710)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(90, (1 / 1), 0.1, 400.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera follows the swarm's average position with some lag
        avg_x = sum(drone['pos_x'] for drone in swarm_drones) / len(swarm_drones)
        avg_y = sum(drone['pos_y'] for drone in swarm_drones) / len(swarm_drones)
        avg_z = sum(drone['pos_z'] for drone in swarm_drones) / len(swarm_drones)

        camera_x = ease_movement(0, -avg_x * 0.2, 0.05)
        camera_y = ease_movement(0, -avg_y * 0.2, 0.05)
        camera_z = ease_movement(0, -avg_z * 0.2, 0.05)  # New line for Z-axis camera movement

        glTranslatef(camera_x, camera_y, -12 + camera_z)  # Modified to include camera_z
        glRotatef(50, 1, 0, 0)
        glTranslatef(-avg_x, -avg_y, -avg_z)  # Also translate based on average Z

        # Draw environment
        draw_textured_ground(ground_size=200, texture=ground_texture)
        draw_target(target_pos)

        # Draw each drone
        for drone in swarm_drones:
            glPushMatrix()
            glTranslatef(drone['pos_x'], drone['pos_y'], drone['pos_z'])
            glRotatef(drone['rotation_y'], 0, 1, 0)  # Apply yaw rotation
            glRotatef(drone['tilt_x'], 1, 0, 0)  # Apply pitch tilt
            glRotatef(drone['tilt_z'], 0, 0, 1)  # Apply roll tilt
            draw_drone(drone['throttle'], drone['tilt_x'], drone['tilt_z'])
            glPopMatrix()

        # Render Right Views (First-Person/FPV)
        fpv_viewports = [
            (1035, 485, 235, 235),
            (1035, 245, 235, 235),
            (1035, 5, 235, 235)
        ]
        count = 0
        for viewport in fpv_viewports:
            drone = swarm_drones[count]
            x, y, width, height = viewport
            glViewport(x, y, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (width / height), 0.1, 400.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Unique FPV camera setup for each drone
            glPushMatrix()

            # Offset the camera based on the drone's index to separate views
            camera_offsets = [
                (0, 0.5, 0),  # Slight up offset for first drone
                (0.5, 0, 0),  # Slight right offset for second drone
                (-0.5, 0, 0)  # Slight left offset for third drone
            ]

            # Apply specific transformations for each drone's FPV view
            glTranslatef(*camera_offsets[count])

            # Position camera slightly above and behind the drone
            glTranslatef(0, 0.5, -1)

            # Rotate to match drone's orientation
            glRotatef(90, 1, 0, 0)  # Pitch
            glRotatef(drone['tilt_z'], 0, 0, 1)  # Roll
            glRotatef(drone['rotation_y'], 0, 1, 0)  # Yaw

            # Move camera relative to drone's position
            glTranslatef(-drone['pos_x'], -drone['pos_y'], -drone['pos_z'])

            # Draw environment for FPV view
            draw_textured_ground(ground_size=200, texture=ground_texture)
            draw_target(target_pos)

            glPopMatrix()
            count += 1

        # Update the display
        pygame.display.flip()
        clock.tick(100)

    # Clean up
    if arduino_connected and ser is not None:
        ser.close()
    pygame.quit()

if __name__ == "__main__":
    main()
