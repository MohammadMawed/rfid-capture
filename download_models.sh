#!/bin/bash
# Download DNN face detector models for better accuracy and speed

echo "Downloading OpenCV DNN face detector models..."
echo "This will provide much better performance than Haar Cascade"

# Download the model file
wget -O opencv_face_detector_uint8.pb \
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"

# Download the config file
wget -O opencv_face_detector.pbtxt \
    "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"

echo ""
echo "✓ Download complete!"
echo "The optimized script will now use DNN face detection"
echo "This is 3-5x faster and more accurate than Haar Cascade"
