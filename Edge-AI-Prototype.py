# Edge AI Recyclable Items Classification System
# Complete implementation with TensorFlow Lite conversion

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RecyclableClassifier:
    def _init_(self, img_size=128, num_classes=4):
        """
        Initialize the recyclable items classifier
        Classes: 0-Paper, 1-Plastic, 2-Glass, 3-Metal
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.tflite_model = None
        self.class_names = ['Paper', 'Plastic', 'Glass', 'Metal']
        
    def create_model(self):
        """Create a lightweight CNN model optimized for edge deployment"""
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=(self.img_size, self.img_size, 3)),
            
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            # Classification head
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic data for demonstration purposes"""
        print("Generating synthetic recyclable items dataset...")
        
        # Generate synthetic images with different patterns for each class
        X = np.random.rand(num_samples, self.img_size, self.img_size, 3)
        y = np.random.randint(0, self.num_classes, num_samples)
        
        # Add class-specific patterns
        for i in range(num_samples):
            if y[i] == 0:  # Paper - add vertical lines
                X[i, :, ::10, :] = 0.8
            elif y[i] == 1:  # Plastic - add horizontal lines
                X[i, ::10, :, :] = 0.8
            elif y[i] == 2:  # Glass - add diagonal patterns
                for j in range(self.img_size):
                    if j < self.img_size:
                        X[i, j, j, :] = 0.9
            elif y[i] == 3:  # Metal - add circular patterns
                center = self.img_size // 2
                for row in range(self.img_size):
                    for col in range(self.img_size):
                        if (row - center)*2 + (col - center)*2 < 400:
                            X[i, row, col, :] = 0.7
        
        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        return X, y_categorical, y
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train the model with early stopping"""
        print("Training the recyclable items classifier...")
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test, y_test_labels):
        """Evaluate model performance"""
        print("Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        report = classification_report(y_test_labels, y_pred_classes, 
                                     target_names=self.class_names)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_classes)
        
        return test_accuracy, report, cm, y_pred_classes
    
    def convert_to_tflite(self, quantize=True):
        """Convert trained model to TensorFlow Lite format"""
        print("Converting model to TensorFlow Lite...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset
        
        self.tflite_model = converter.convert()
        
        # Save the model
        with open('recyclable_classifier.tflite', 'wb') as f:
            f.write(self.tflite_model)
        
        print(f"TensorFlow Lite model saved. Size: {len(self.tflite_model)} bytes")
        return self.tflite_model
    
    def _representative_dataset(self):
        """Generate representative dataset for quantization"""
        for _ in range(100):
            data = np.random.rand(1, self.img_size, self.img_size, 3).astype(np.float32)
            yield [data]
    
    def test_tflite_inference(self, X_test, num_samples=10):
        """Test TensorFlow Lite model inference speed"""
        print("Testing TensorFlow Lite model inference...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test inference speed
        inference_times = []
        predictions = []
        
        for i in range(min(num_samples, len(X_test))):
            start_time = time.time()
            
            # Prepare input
            input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(np.argmax(output_data))
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        avg_inference_time = np.mean(inference_times)
        print(f"Average inference time: {avg_inference_time:.4f} seconds")
        
        return predictions, avg_inference_time
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Main execution
def main():
    """Main function to demonstrate the complete pipeline"""
    
    print("=== Edge AI Recyclable Items Classification System ===\n")
    
    # Initialize classifier
    classifier = RecyclableClassifier(img_size=128, num_classes=4)
    
    # Create model
    model = classifier.create_model()
    print(f"Model created with {model.count_params()} parameters")
    model.summary()
    
    # Generate synthetic data
    X, y, y_labels = classifier.generate_synthetic_data(num_samples=2000)
    
    # Split data
    split_idx = int(0.8 * len(X))
    val_split_idx = int(0.9 * len(X))
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:val_split_idx], y[split_idx:val_split_idx]
    X_test, y_test = X[val_split_idx:], y[val_split_idx:]
    y_test_labels = y_labels[val_split_idx:]
    
    print(f"\nDataset split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Train model
    history = classifier.train_model(X_train, y_train, X_val, y_val, epochs=15)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Evaluate model
    test_accuracy, report, cm, predictions = classifier.evaluate_model(
        X_test, y_test, y_test_labels
    )
    
    print(f"\n=== Model Performance ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(cm)
    
    # Convert to TensorFlow Lite
    tflite_model = classifier.convert_to_tflite(quantize=True)
    
    # Test TensorFlow Lite inference
    tflite_predictions, avg_inference_time = classifier.test_tflite_inference(X_test)
    
    # Compare original vs TFLite predictions
    original_predictions = np.argmax(classifier.model.predict(X_test[:10]), axis=1)
    print(f"\n=== TensorFlow Lite Comparison ===")
    print(f"Original predictions: {original_predictions}")
    print(f"TFLite predictions: {tflite_predictions}")
    print(f"Predictions match: {np.array_equal(original_predictions, tflite_predictions)}")
    
    # Calculate model sizes
    original_size = len(classifier.model.get_weights()) * 4  # Rough estimate
    tflite_size = len(tflite_model)
    compression_ratio = original_size / tflite_size if tflite_size > 0 else 0
    
    print(f"\n=== Model Optimization Results ===")
    print(f"Original model size (estimated): {original_size:,} bytes")
    print(f"TensorFlow Lite model size: {tflite_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Average inference time: {avg_inference_time:.4f} seconds")
    
    print(f"\n=== Edge AI Benefits ===")
    print("1. Low Latency: Real-time classification without network dependency")
    print("2. Privacy: Data processing happens locally on device")
    print("3. Offline Operation: Works without internet connectivity")
    print("4. Reduced Bandwidth: No need to send images to cloud")
    print("5. Cost Effective: Lower operational costs for deployment")
    
    return classifier, test_accuracy, avg_inference_time

if _name_ == "_main_":
    classifier, accuracy, inference_time = main()
    
    print(f"\n=== Final Results ===")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print("TensorFlow Lite model saved as 'recyclable_classifier.tflite'")
