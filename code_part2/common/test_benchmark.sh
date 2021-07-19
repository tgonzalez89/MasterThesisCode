#!/bin/bash


model_name="$(basename -- $(pwd))"
input_scaling=0.75
layers_scaling=0.75
pruning_sparcity=0.5
system=$1
input_size=$2
time=$3
scaled_input_size=`python3 -c "print(int($input_size*$input_scaling))"`
if [ $system = PC ]
then
    device=CPU
else
    device=MYRIAD
fi


# Benchmark

# Tensorflow
# CPU
echo -e "\nBENCHMARKING TENSORFLOW CPU\n"
if [ $system = PC ]
then
    export CUDA_VISIBLE_DEVICES=-1
fi
python3 ../common/test_benchmark.py models/$model_name.h5 $time
sleep 60
python3 ../common/test_benchmark.py models/${model_name}_input-scaling-$input_scaling.h5 $time $scaled_input_size
sleep 60
python3 ../common/test_benchmark.py models/${model_name}_layers-scaling-$layers_scaling.h5 $time
sleep 60
python3 ../common/test_benchmark.py models/${model_name}_pruning-$pruning_sparcity.h5 $time
sleep 60
python3 ../common/test_benchmark.py models/${model_name}_sep-conv.h5 $time
sleep 60
python3 ../common/test_benchmark.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}_sep-conv.h5 $time $scaled_input_size
sleep 60
python3 ../common/test_benchmark.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}.h5 $time $scaled_input_size
sleep 60
# GPU
if [ $system = PC ]
then
    echo -e "\nBENCHMARKING TENSORFLOW GPU\n"
    export CUDA_VISIBLE_DEVICES=0
    python3 ../common/test_benchmark.py models/$model_name.h5 $time
    python3 ../common/test_benchmark.py models/${model_name}_input-scaling-$input_scaling.h5 $time $scaled_input_size
    python3 ../common/test_benchmark.py models/${model_name}_layers-scaling-$layers_scaling.h5 $time
    python3 ../common/test_benchmark.py models/${model_name}_pruning-$pruning_sparcity.h5 $time
    python3 ../common/test_benchmark.py models/${model_name}_sep-conv.h5 $time
    python3 ../common/test_benchmark.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}_sep-conv.h5 $time $scaled_input_size
    python3 ../common/test_benchmark.py models/${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}.h5 $time $scaled_input_size
fi

# OpenVINO
echo -e "\nBENCHMARKING OPENVINO $device\n"
for m in $model_name ${model_name}_input-scaling-$input_scaling ${model_name}_layers-scaling-$layers_scaling ${model_name}_pruning-$pruning_sparcity ${model_name}_sep-conv ${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}_sep-conv ${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}
do
    echo -e "\nBENCHMARKING OPENVINO MODEL models/$m.xml\n"
    if [ $system = PC ]
    then
        python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py -t $time -m models/$m.xml -d $device -api async -nireq 4 -nstreams 4 -nthreads 12
    else
        python3 ~/openvino/inference-engine/tools/benchmark_tool/benchmark_app.py -t $time -m models/$m.xml -d $device -api async -nireq 8
    fi
    sleep 60
done
