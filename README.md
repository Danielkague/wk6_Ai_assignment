# Edge AI Prototype for Image Classification

This project demonstrates the development of an Edge AI prototype for image classification using TensorFlow Lite. The goal is to train a lightweight model that can be deployed on edge devices like Raspberry Pi for real-time applications.

## 📋 Project Overview

This assignment implements a complete Edge AI pipeline:

- **Model Training**: Lightweight CNN for image classification
- **Model Conversion**: TensorFlow to TensorFlow Lite conversion
- **Edge Deployment**: Testing and validation for edge devices
- **Real-time Applications**: Demonstrating Edge AI benefits

## 🎯 Objectives

1. Train a lightweight CNN model for image classification
2. Convert the model to TensorFlow Lite format
3. Test the TFLite model performance
4. Demonstrate Edge AI benefits for real-time applications

## 📁 Project Structure

```
wk6_Ai_assignment/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── task_1.py               # Main Python script
├── edge_ai.ipynb           # Jupyter notebook version
├── mnist_cnn_model.tflite  # Generated TensorFlow Lite model
└── training_history.png     # Training visualization (generated)
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
   ```

## 📖 Usage

### Option 1: Run Python Script

```bash
python task_1.py
```

### Option 2: Use Jupyter Notebook

```bash
jupyter notebook edge_ai.ipynb
```

## 🔧 Implementation Details

### Model Architecture

- **Input**: 28x28x1 grayscale images
- **Convolutional Layers**: 2 Conv2D layers with ReLU activation
- **Pooling**: 2 MaxPooling2D layers for dimensionality reduction
- **Output**: 10-class softmax classification
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy

### Training Process

- **Dataset**: MNIST (60,000 training, 10,000 test samples)
- **Epochs**: 5
- **Validation**: Split from test data
- **Data Preprocessing**: Normalization to [0,1] range

### TensorFlow Lite Conversion

- **Optimization**: Default optimization for size and latency
- **Format**: `.tflite` binary format
- **Compatibility**: Edge devices (Raspberry Pi, mobile, IoT)

## 📊 Results

The implementation achieves:

- **Original Model Accuracy**: ~98% on MNIST test set
- **TFLite Model Accuracy**: Comparable performance maintained
- **Model Size**: Optimized for edge deployment
- **Inference Speed**: Suitable for real-time applications

## 🌟 Edge AI Benefits for Real-Time Applications

### 1. **Low Latency**

- Processing happens locally on the device
- Eliminates network round-trips to cloud servers
- Critical for applications requiring immediate action (autonomous vehicles, industrial automation)

### 2. **Offline Capabilities**

- Works without constant internet connection
- Vital for remote locations or unreliable connectivity
- Enables continuous operation in various environments

### 3. **Reduced Bandwidth Usage**

- Only processed insights need to be transmitted
- Significant reduction in data transfer costs
- Efficient for high-volume data streams (video, sensors)

### 4. **Enhanced Privacy & Security**

- Sensitive data processed locally
- No transmission of raw data to cloud
- Reduced risk of data breaches

### 5. **Lower Operational Costs**

- Reduced reliance on cloud infrastructure
- Lower data transfer and processing costs
- Scalable without proportional cost increase

### 6. **Increased Reliability**

- Continues operation if cloud services are unavailable
- Redundant processing capabilities
- Improved system resilience

## 🍓 Raspberry Pi Deployment

### Prerequisites

- Raspberry Pi (3 or 4 recommended)
- Camera module (optional)
- Python 3.7+

### Deployment Steps

1. **Install TensorFlow Lite Runtime:**

   ```bash
   pip install tflite-runtime
   ```

2. **Transfer Model File:**

   ```bash
   scp mnist_cnn_model.tflite pi@raspberrypi.local:/home/pi/
   ```

3. **Create Inference Script:**

   ```python
   import tflite_runtime.interpreter as tflite
   import numpy as np
   import cv2

   # Load model
   interpreter = tflite.Interpreter(model_path='mnist_cnn_model.tflite')
   interpreter.allocate_tensors()

   # Get input/output details
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()

   # Process camera input
   cap = cv2.VideoCapture(0)
   while True:
       ret, frame = cap.read()
       # Preprocess frame for model input
       processed_frame = preprocess_image(frame)

       # Run inference
       interpreter.set_tensor(input_details[0]['index'], processed_frame)
       interpreter.invoke()
       prediction = interpreter.get_tensor(output_details[0]['index'])

       # Display results
       cv2.imshow('Edge AI Classification', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   ```

4. **Run Application:**
   ```bash
   python inference_script.py
   ```

## 🔍 Testing and Validation

### Model Performance

- **Accuracy Comparison**: Original vs TFLite model
- **Inference Speed**: Time measurements on target device
- **Memory Usage**: Resource consumption analysis
- **File Size**: Model optimization verification

### Edge Device Testing

- **Compatibility**: TensorFlow Lite runtime verification
- **Performance**: Real-time inference capability
- **Resource Usage**: CPU, memory, and power consumption
- **Reliability**: Long-term operation testing

## 📈 Future Enhancements

### Model Improvements

- **Custom Dataset**: Domain-specific training data
- **Architecture Optimization**: MobileNet, EfficientNet variants
- **Quantization**: Further size and speed optimization
- **Pruning**: Remove unnecessary model parameters

### Deployment Enhancements

- **Hardware Acceleration**: GPU/TPU support
- **Multi-platform**: Android, iOS, embedded systems
- **Cloud Integration**: Hybrid edge-cloud processing
- **Real-time Monitoring**: Performance analytics

### Application Extensions

- **Multi-modal Input**: Audio, sensor data integration
- **Federated Learning**: Distributed model updates
- **Edge-to-Edge**: Device-to-device communication
- **Security**: Model encryption and authentication

## 🤝 Contributing

This is an academic assignment demonstrating Edge AI concepts. For educational purposes, feel free to:

- Experiment with different model architectures
- Test on various edge devices
- Extend functionality for specific use cases
- Improve documentation and examples

## 📚 References

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Edge AI Best Practices](https://www.tensorflow.org/lite/guide)
- [Raspberry Pi TensorFlow Setup](https://www.tensorflow.org/lite/guide/python)
- [Model Optimization Techniques](https://www.tensorflow.org/lite/performance)

## 📄 License

This project is created for educational purposes as part of an AI course assignment.

---

**Author**: Student Assignment  
**Course**: Week 6 AI Assignment  
**Date**: 2024  
**Technology Stack**: TensorFlow, TensorFlow Lite, Python, Jupyter Notebook
