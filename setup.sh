#!/bin/bash
# Automated setup script for optimized RFID face recognition system

set -e  # Exit on error

echo "=========================================="
echo "RFID Face Recognition System Setup"
echo "=========================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    python3-opencv \
    python3-picamera2 \
    python3-pip \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    python3-pyqt5 \
    libqt4-test \
    wget

# Install Python packages
echo "🐍 Installing Python packages..."
pip3 install --upgrade pip
pip3 install opencv-contrib-python numpy

# Check if SPI is enabled
echo "🔌 Checking SPI interface..."
if lsmod | grep -q spi; then
    echo "✓ SPI is enabled"
else
    echo "⚠️  SPI is not enabled!"
    echo "Enabling SPI..."
    sudo raspi-config nonint do_spi 0
    echo "✓ SPI enabled (reboot may be required)"
fi

# Download DNN models
echo "🧠 Downloading DNN face detection models..."
if [ ! -f opencv_face_detector_uint8.pb ]; then
    wget -q --show-progress -O opencv_face_detector_uint8.pb \
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

if [ ! -f opencv_face_detector.pbtxt ]; then
    wget -q --show-progress -O opencv_face_detector.pbtxt \
        "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    echo "✓ Config downloaded"
else
    echo "✓ Config already exists"
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p faces
mkdir -p images
echo "✓ Directories created"

# Make scripts executable
echo "🔧 Setting permissions..."
chmod +x rfid_optimized.py
chmod +x download_models.sh
echo "✓ Permissions set"

# Test camera
echo "📷 Testing camera..."
if python3 -c "from picamera2 import Picamera2; import sys; p = Picamera2(); sys.exit(0)" 2>/dev/null; then
    echo "✓ Camera is working"
else
    echo "⚠️  Camera test failed"
    echo "   Please check camera connection and enable in raspi-config"
fi

# Test OpenCV
echo "🔍 Testing OpenCV..."
if python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null; then
    echo "✓ OpenCV is working"
else
    echo "⚠️  OpenCV test failed"
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect your RFID reader (make sure SPI is enabled)"
echo "2. Connect your camera module"
echo "3. Run: python3 rfid_optimized.py"
echo ""
echo "For best performance, make sure DNN models were downloaded."
echo "Check the README.md for configuration and usage instructions."
echo ""

# Check if reboot needed
if [ -f /var/run/reboot-required ]; then
    echo "⚠️  REBOOT REQUIRED"
    read -p "Reboot now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
fi
