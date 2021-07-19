#!/bin/bash

# Compile the source code.
echo "Compiling the code"
#for f in `ls *.cpp`
#do
#    echo "Compiling $f"
#    ./compile.sh $f $1
#done
../common/compile.sh sobel-orig.cpp
../common/compile.sh sobel-openvino-2D-async.cpp

# Convert the model to OpenVINO IR.
#echo "Converting model to OpenVINO IR"
#../common/keras_to_ir-0D.sh nn_sobel-0D 9 > ir-0D.log 2>&1
#block_size=256
#../common/keras_to_ir-2D.sh nn_sobel-2D $block_size $block_size 1 > ir-2D.log 2>&1

# Run the app. Measure performance and output quality.
rm -rf results
mkdir results

set -v
#time ./sobel-opencv.out data/test.jpg results/test-opencv.png
time ./sobel-orig.out data/test.jpg results/test-orig.png
#time ./sobel-openvino.out data/test.jpg results/test-ov.png CPU
#time ./sobel-openvino-async.out data/test.jpg results/test-ov-async-cpu.png CPU 10000
#time ./sobel-orig-2D.out data/test.jpg results/test-orig-2D.png 256
#time ./sobel-openvino-2D.out data/test.jpg results/test-ov-2D-cpu.png CPU
#time ./sobel-openvino-2D.out data/test.jpg results/test-ov-2D-myr.png MYRIAD
#time ./sobel-openvino-2D-async.out data/test.jpg results/test-ov-2D-async-cpu.png CPU 4
time ./sobel-openvino-2D-async.out data/test.jpg results/test-ov-2D-async-myr.png MYRIAD 4

#../common/img_diff.py results/test-orig.png results/test-ov-2D-async-cpu.png results/test-diff-cpu.png
../common/img_diff.py results/test-orig.png results/test-ov-2D-async-myr.png results/test-diff-myr.png
set +v
