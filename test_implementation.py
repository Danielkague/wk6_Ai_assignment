#!/usr/bin/env python3
"""
Test script for Edge AI implementation
Verifies that all components work correctly
"""

import os
import sys
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_tensorflow_lite():
    """Test TensorFlow Lite functionality"""
    print("\nTesting TensorFlow Lite...")
    try:
        import tensorflow as tf
        
        # Create a simple model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='softmax', input_shape=(5,))
        ])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Test interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        print("✓ TensorFlow Lite conversion and interpreter working")
        return True
    except Exception as e:
        print(f"✗ TensorFlow Lite test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    required_files = [
        'README.md',
        'requirements.txt',
        'task_1.py',
        'edge_ai.ipynb'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_requirements():
    """Test if requirements.txt is properly formatted"""
    print("\nTesting requirements.txt...")
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip()
        
        if requirements:
            print("✓ requirements.txt contains dependencies")
            print(f"  Dependencies: {requirements}")
            return True
        else:
            print("✗ requirements.txt is empty")
            return False
    except FileNotFoundError:
        print("✗ requirements.txt not found")
        return False

def test_python_script():
    """Test if the main Python script can be imported"""
    print("\nTesting Python script...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the script (this will test syntax)
        import task_1
        print("✓ task_1.py can be imported (syntax is correct)")
        return True
    except Exception as e:
        print(f"✗ task_1.py import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Edge AI Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Requirements", test_requirements),
        ("Imports", test_imports),
        ("TensorFlow Lite", test_tensorflow_lite),
        ("Python Script", test_python_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is ready.")
        print("\nTo run the Edge AI prototype:")
        print("  python task_1.py")
        print("\nOr use the Jupyter notebook:")
        print("  jupyter notebook edge_ai.ipynb")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 