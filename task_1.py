# prompt: Task 1: Edge AI Prototype
# Tools: TensorFlow Lite, Raspberry Pi/Colab (simulation).
# Goal:
# Train a lightweight image classification model (e.g., recognizing recyclable items).
# Convert the model to TensorFlow Lite and test it on a sample dataset.

!pip install -q tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Task 1: Edge AI Prototype

# 1. Train a lightweight image classification model

# We'll use a simple CNN model for demonstration.
# For a real Edge AI project, you'd use a more appropriate dataset
# like a custom dataset of recyclable items.
# Here, we'll use a subset of the MNIST dataset for simplicity.

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add a channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Define a simple lightweight CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("Model training finished.")

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy of the trained model: {accuracy:.4f}")

# 2. Convert the model to TensorFlow Lite

# Create a converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Specify optimization for reduced size and latency (common for edge devices)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'mnist_cnn_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TensorFlow Lite model saved to: {tflite_model_path}")

# 3. Test the TFLite model on a sample dataset

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape and type
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Prepare a small sample of test data for inference
sample_size = 100
x_test_sample = x_test[:sample_size]
y_test_sample = y_test[:sample_size]

# Run inference on the sample data
print(f"\nRunning inference on {sample_size} test samples using the TFLite model...")
tflite_predictions = []
correct_predictions = 0

for i in range(sample_size):
    input_data = np.array(x_test_sample[i], dtype=input_dtype)
    # Add a batch dimension to the input data
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data[0])
    tflite_predictions.append(prediction)

    if prediction == y_test_sample[i]:
        correct_predictions += 1

tflite_accuracy = correct_predictions / sample_size
print(f"Accuracy of the TFLite model on sample data: {tflite_accuracy:.4f}")

