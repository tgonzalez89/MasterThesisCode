#!/bin/bash

# Compile the source code.
echo "Compiling the code"
./compile-orig.sh
#./compile-openvino.sh jmeint-openvino-1D.cpp
../common/compile.sh jmeint-openvino-2D.cpp

# Convert the model to OpenVINO IR.
#echo "Converting model to OpenVINO IR"
#block_size=100
#in_size=18
#../common/keras_to_ir-1D.sh nn_jmeint-1D $((250 * in_size)) 1 > ir-1D.log 2>&1
#../common/keras_to_ir-2D.sh nn_jmeint-2D $block_size $((block_size * in_size)) 1 > ir-2D.log 2>&1

# Run the app. Measure performance and output quality.
rm -rf results
mkdir results

set -v
time ./jmeint-orig.out data/inputs_1000K.txt results/outputs_1000K-orig.txt
#time ./jmeint-openvino-2D.out data/inputs_1000K.txt results/outputs_1000K-ov-2D-cpu.txt CPU 4
time ./jmeint-openvino-2D.out data/inputs_1000K.txt results/outputs_1000K-ov-2D-myr.txt MYRIAD 4

time ./jmeint-orig.out data/inputs_40K.txt results/outputs_40K-orig.txt
#time ./jmeint-openvino-2D.out data/inputs_40K.txt results/outputs_40K-ov-2D-cpu.txt CPU 4
time ./jmeint-openvino-2D.out data/inputs_40K.txt results/outputs_40K-ov-2D-myr.txt MYRIAD 4

#./calc_acc.py results/outputs_40K-orig.txt results/outputs_40K-ov-2D-cpu.txt
./calc_acc.py results/outputs_40K-orig.txt results/outputs_40K-ov-2D-myr.txt
set +v
