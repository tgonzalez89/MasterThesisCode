#!/bin/bash

script_dir=$(realpath $(dirname $0))
name=$1
height=$2
width=$3
channels=$4

$script_dir/keras_to_tf.py -m $name.h5 -no 1

mo_tf.py --input_model $name.pb --input_shape [1,$height,$width,$channels] --data_type FP16 --generate_deprecated_IR_V7
