#!/bin/bash

# Compile the source code.
echo "Compiling the code"
../common/compile.sh kmeans-orig.cpp
#../common/compile.sh kmeans-openvino-1D.cpp
../common/compile.sh kmeans-openvino-2D.cpp

# Convert the model to OpenVINO IR.
#echo "Converting model to OpenVINO IR"
#block_size=64
#clusters=6
#in_size=6
#../common/keras_to_ir-1D.sh nn_kmeans-1D $((140 * clusters * in_size)) 1 > ir-1D.log 2>&1
#../common/keras_to_ir-2D.sh nn_kmeans-2D $((block_size * clusters)) $((block_size * clusters * in_size)) 1 > ir-2D.log 2>&1

# Run the app. Measure performance and output quality.
rm -rf results
mkdir results

set -v
#time ./kmeans-orig.out data/26.png results/26-orig.png
#time ./kmeans-openvino-1D.out data/26.png results/26-ov-1D-cpu.png CPU 4
#time ./kmeans-openvino-1D.out data/26.png results/26-ov-1D-myr.png MYRIAD 4
#time ./kmeans-openvino-2D.out data/26.png results/26-ov-2D-cpu.png CPU 4
#time ./kmeans-openvino-2D.out data/26.png results/26-ov-2D-myr.png MYRIAD 4

time ./kmeans-orig.out data/test.jpg results/test-orig.png
#time ./kmeans-openvino-1D.out data/test.jpg results/test-ov-1D-cpu.png CPU 4
#time ./kmeans-openvino-1D.out data/test.jpg results/test-ov-1D-cpu.png MYRIAD 4
#time ./kmeans-openvino-2D.out data/test.jpg results/test-ov-2D-cpu.png CPU 4
time ./kmeans-openvino-2D.out data/test.jpg results/test-ov-2D-myr.png MYRIAD 4

#../common/img_diff.py results/test-orig.png results/test-ov-2D-cpu.png results/test-diff-cpu.png
../common/img_diff.py results/test-orig.png results/test-ov-2D-myr.png results/test-diff-myr.png
set +v
