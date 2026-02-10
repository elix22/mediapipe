#!/bin/bash
# Rebuild OpenCV without Protobuf conflicts for macOS
set -e

echo "Uninstalling Homebrew OpenCV..."
brew uninstall opencv --ignore-dependencies || true

echo "Downloading OpenCV 4.13.0 and opencv_contrib..."
cd /tmp
rm -rf opencv-4.13.0 opencv-4.13.0.tar.gz opencv_contrib-4.13.0 opencv_contrib-4.13.0.tar.gz
curl -L -o opencv-4.13.0.tar.gz https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz
curl -L -o opencv_contrib-4.13.0.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/4.13.0.tar.gz
tar xzf opencv-4.13.0.tar.gz
tar xzf opencv_contrib-4.13.0.tar.gz
cd opencv-4.13.0
mkdir -p build && cd build

echo "Configuring OpenCV without DNN/Protobuf..."
# Configure without DNN and Protobuf to avoid conflicts with MediaPipe
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.13.0/modules \
      -D BUILD_PROTOBUF=OFF \
      -D BUILD_opencv_dnn=OFF \
      -D BUILD_opencv_dnn_objdetect=OFF \
      -D BUILD_opencv_dnn_superres=OFF \
      -D WITH_PROTOBUF=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON ..

echo "Building OpenCV (this may take 10-20 minutes)..."
make -j$(sysctl -n hw.ncpu)

echo "Installing OpenCV..."
sudo make install

echo "Cleaning up..."
cd /
rm -rf /tmp/opencv-4.13.0 /tmp/opencv-4.13.0.tar.gz /tmp/opencv_contrib-4.13.0 /tmp/opencv_contrib-4.13.0.tar.gz

echo ""
echo "✅ OpenCV installed successfully without Protobuf conflicts!"
echo "Now rebuild MediaPipe:"
echo "  cd $(pwd)"
echo "  bazel clean --expunge"
echo "  bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu"
