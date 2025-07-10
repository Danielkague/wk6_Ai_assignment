#!/usr/bin/env python3
"""
Setup script for Edge AI Assignment
Installs dependencies and verifies the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that all packages are installed correctly"""
    print("\nVerifying installation...")
    
    packages = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib")
    ]
    
    all_installed = True
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name} {version} installed")
        except ImportError:
            print(f"✗ {name} not installed")
            all_installed = False
    
    return all_installed

def test_basic_functionality():
    """Test basic TensorFlow functionality"""
    print("\nTesting basic functionality...")
    try:
        import tensorflow as tf
        import numpy as np
        
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        
        print("✓ TensorFlow basic operations working")
        return True
    except Exception as e:
        print(f"✗ TensorFlow functionality test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 50)
    print("Edge AI Assignment Setup")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found")
        return False
    
    # Install dependencies
    if not install_requirements():
        return False
    
    # Verify installation
    if not verify_installation():
        print("\nSome packages failed to install. Please try:")
        print("  pip install tensorflow numpy matplotlib")
        return False
    
    # Test functionality
    if not test_basic_functionality():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nYou can now run the Edge AI prototype:")
    print("  python task_1.py")
    print("\nOr use the Jupyter notebook:")
    print("  jupyter notebook edge_ai.ipynb")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 