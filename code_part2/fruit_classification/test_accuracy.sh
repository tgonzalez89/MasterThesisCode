#!/bin/bash

img_size=100

if [ `uname -m` = x86_64 ]
then
    source ../common/test_accuracy.sh PC $img_size 2>&1 | tee test_accuracy_pc.log
else
    source ../common/test_accuracy.sh Pi $img_size 2>&1 | tee test_accuracy_pi.log
fi
