import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Input

# ==========================================
# Load Preprocessed Data for All Gestures
# ==========================================

# Gesture labels (must match preprocessed .npy files)
gestures = ["bounce", "chop", "iBounce"]

# Initialize data containers
X, y = [], []

# Load each gesture dataset and assign corresponding label
for idx, gesture in enumerate(gestures):
    data = np.load(f"preprocessed_data/{gesture}_preprocessed.npy")
    X.append(data[:, :-1])  # Exclude any label/timestamp column if present
    y.append(np.full(data.shape[0], idx))  # Assign label index

# Combine all data and shuffle
X = np.vstack(X)
y = np.hstack(y)
X, y = shuffle(X, y, random_state=42)

# Reshape features for LSTM input: (samples, timesteps, features)
timesteps, features = X.shape[1], X.shape[2]
X = X.reshape((-1, timesteps, features))

# ==========================================
# Split Data into Training, Validation, Test
# ==========================================

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ==========================================
# Model Architecture
# ==========================================

# Simple Bidirectional LSTM model
model = Sequential([
    Input(shape=(timesteps, features)),                     # Input layer
    Bidirectional(LSTM(110, activation='tanh')),           # BiLSTM with 110 units
    Dense(3, activation='softmax')                         # Output layer (3 gesture classes)
])

# Compile model with SGD optimizer and cross-entropy loss
model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Model Training
# ==========================================

# Train the model and track duration
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

print(f"\nTraining completed in {training_time:.2f} seconds\n")

# Print model summary
model.summary()

# ==========================================
# Plot Training History (Accuracy & Loss)
# ==========================================

def plot_history(hist):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Acc')
    plt.plot(hist.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# ==========================================
# Model Evaluation on Test Set
# ==========================================

# Evaluate overall accuracy on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}\n")

# Generate predicted class labels
y_pred = np.argmax(model.predict(X_test), axis=1)

# Display detailed classification report
print("Test Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=gestures))

# ==========================================
# Per-Class Accuracy Display
# ==========================================

def per_class_accuracy(y_true, y_pred, class_names):
    class_acc = []
    for i, class_name in enumerate(class_names):
        idx = (y_true == i)
        acc = accuracy_score(y_true[idx], y_pred[idx])
        class_acc.append([class_name, acc])
    print("\nPer-Class Accuracy:\n")
    print(tabulate(class_acc, headers=["Class", "Accuracy"], floatfmt=".2f"))

per_class_accuracy(y_test, y_pred, gestures)

# ==========================================
# Confusion Matrix Visualization
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=gestures, yticklabels=gestures)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Data')
plt.show()
