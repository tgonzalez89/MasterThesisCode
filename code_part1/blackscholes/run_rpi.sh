#!/bin/bash

# Compile the source code.
echo "Compiling the code"
../common/compile.sh blackscholes-orig.cpp
#../common/compile.sh blackscholes-openvino-1D.cpp
../common/compile.sh blackscholes-openvino-2D.cpp

# Convert the model to OpenVINO IR.
#echo "Converting model to OpenVINO IR"
#block_size=100
#in_size=6
#../common/keras_to_ir-1D.sh nn_blackscholes-1D $((800 * in_size)) 1 > ir-1D.log 2>&1
#../common/keras_to_ir-2D.sh nn_blackscholes-2D $block_size $((block_size * in_size)) 1 > ir-2D.log 2>&1

# Run the app. Measure performance and output quality.
rm -rf results
mkdir results

set -v
time ./blackscholes-orig.out data/inputs_200K.txt results/outputs_200K-orig.txt
#time ./blackscholes-openvino-2D.out data/inputs_200K.txt results/outputs_200K-ov-2D-cpu.txt CPU 4
time ./blackscholes-openvino-2D.out data/inputs_200K.txt results/outputs_200K-ov-2D-myr.txt MYRIAD 4

time ./blackscholes-orig.out data/inputs_120K.txt results/outputs_120K-orig.txt
#time ./blackscholes-openvino-2D.out data/inputs_120K.txt results/outputs_120K-ov-2D-cpu.txt CPU 4
time ./blackscholes-openvino-2D.out data/inputs_120K.txt results/outputs_120K-ov-2D-myr.txt MYRIAD 4

#./calc_err.py results/outputs_120K-orig.txt results/outputs_120K-ov-2D-cpu.txt
./calc_err.py results/outputs_120K-orig.txt results/outputs_120K-ov-2D-myr.txt
set +v
