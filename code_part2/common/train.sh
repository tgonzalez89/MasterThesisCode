#!/bin/bash


mkdir -p models

model_name="$(basename -- $(pwd))"
input_scaling=0.75
layers_scaling=0.75
pruning_sparcity=0.5
h=$1
w=$2
c=$3
h_s=`python3 -c "print(int($h*$input_scaling))"`
w_s=`python3 -c "print(int($w*$input_scaling))"`


# Train models

export CUDA_VISIBLE_DEVICES=0
python3 train.py
python3 train.py -i $input_scaling
python3 train.py -l $layers_scaling
python3 train.py -p $pruning_sparcity
python3 train.py -c SeparableConv2D
python3 train.py -i $input_scaling -l $layers_scaling -p $pruning_sparcity -c SeparableConv2D


# Convert models to OpenVINO IR

cd models
../../common/keras_to_ir.sh $model_name $h $w $c
../../common/keras_to_ir.sh ${model_name}_input-scaling-$input_scaling $h_s $w_s $c
../../common/keras_to_ir.sh ${model_name}_layers-scaling-$layers_scaling $h $w $c
../../common/keras_to_ir.sh ${model_name}_pruning-$pruning_sparcity $h $w $c
../../common/keras_to_ir.sh ${model_name}_sep-conv $h $w $c
../../common/keras_to_ir.sh ${model_name}_input-scaling-${input_scaling}_layers-scaling-${layers_scaling}_pruning-${pruning_sparcity}_sep-conv $h_s $w_s $c
cd ..
