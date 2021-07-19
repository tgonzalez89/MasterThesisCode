#!/bin/bash

# Compile the source code.
echo "Compiling the code"
./compile-orig.sh
#./compile-openvino.sh inversek2j-openvino-1D.cpp
./compile-openvino.sh inversek2j-openvino-2D.cpp

# Convert the model to OpenVINO IR.
#echo "Converting model to OpenVINO IR"
#block_size=100
#in_size=2
#../common/keras_to_ir-1D.sh nn_inversek2j-1D $((2000 * in_size)) 1 > ir-1D.log 2>&1
#../common/keras_to_ir-2D.sh nn_inversek2j-2D $block_size $((block_size * in_size)) 1 > ir-2D.log 2>&1

# Run the app. Measure performance and output quality.
rm -rf results
mkdir results

set -v
time ./inversek2j-orig.out data/inputs_1000K.txt results/outputs_1000K-orig.txt
#time ./inversek2j-openvino-2D.out data/inputs_1000K.txt results/outputs_1000K-ov-2D-cpu.txt CPU 4
time ./inversek2j-openvino-2D.out data/inputs_1000K.txt results/outputs_1000K-ov-2D-myr.txt MYRIAD 4

time ./inversek2j-orig.out data/inputs_120K.txt results/outputs_120K-orig.txt
#time ./inversek2j-openvino-2D.out data/inputs_120K.txt results/outputs_120K-ov-2D-cpu.txt CPU 4
time ./inversek2j-openvino-2D.out data/inputs_120K.txt results/outputs_120K-ov-2D-myr.txt MYRIAD 4

#./calc_err.py data/inputs_120K.txt results/outputs_120K-ov-2D-cpu.txt
./calc_err.py data/inputs_120K.txt results/outputs_120K-ov-2D-myr.txt
set +v
