#!/bin/bash

script_dir=$(realpath $(dirname $0))
name=$1
size=$2

$script_dir/keras_to_tf.py -m $name.h5 -no 1

mo_tf.py --input_model $name.pb --input_shape [1,$size] --data_type FP16 --generate_deprecated_IR_V7
