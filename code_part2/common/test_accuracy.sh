#!/bin/bash


model_name="$(basename -- $(pwd))"
input_scaling=0.75
layers_scaling=0.75
pruning_sparcity=0.5
system=$1
input_size=$2
scaled_input_size=`python3 -c "print(int($input_size*$input_scaling))"`
if [ $system = PC ]
then
    device=CPU
else
    device=MYRIAD
fi


# Test accuracy

# Tensorflow
echo -e "\nTESTING ACCURARY TENSORFLOW\n"
if [ $system = PC ]
then
    export CUDA_VISIBLE_DEVICES=0
fi
python3 ../common/test_accuracy.py models/$model_name.h5
python3 ../common/test_accuracy.py models/${model_name}_input-scaling-$input_scaling.h5 $scaled_input_size
python3 ../common/test_accuracy.py models/${model_name}_layers-scaling-$layers_scaling.h5
python3 ../common/test_accuracy.py models/${model_name}_pruning-$pruning_sparcity.h5
python3 ../common/test_accuracy.py models/${model_name}_sep-conv.h5
python3 ../common/test_accuracy.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}_sep-conv.h5 $scaled_input_size
python3 ../common/test_accuracy.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}.h5 $scaled_input_size

# OpenVINO
echo -e "\nTESTING ACCURARY OPENVINO $device\n"
python3 ../common/test_accuracy_ov.py models/$model_name.xml $device
python3 ../common/test_accuracy_ov.py models/${model_name}_input-scaling-$input_scaling.xml $device $scaled_input_size
python3 ../common/test_accuracy_ov.py models/${model_name}_layers-scaling-$layers_scaling.xml $device
python3 ../common/test_accuracy_ov.py models/${model_name}_pruning-$pruning_sparcity.xml $device
python3 ../common/test_accuracy_ov.py models/${model_name}_sep-conv.xml $device
python3 ../common/test_accuracy_ov.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}_sep-conv.xml $device $scaled_input_size
python3 ../common/test_accuracy_ov.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}.xml $device $scaled_input_size
