#!/bin/bash

name=$1
h=$2
w=$3
c=$4

echo -e "\nCONVERTING MODEL TO OPENVINO IR: $name\n"

python3 ../../common/keras_to_tf.py -m $name.h5 -no 1

mo_tf.py --input_model $name.pb --input_shape [1,$h,$w,$c] --data_type=FP16 --generate_deprecated_IR_V7
