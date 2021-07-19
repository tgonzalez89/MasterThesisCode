#!/bin/bash

img_size=100
benchmark_time_s=60

if [ `uname -m` = x86_64 ]
then
    source ../common/test_benchmark.sh PC $img_size $benchmark_time_s 2>&1 | tee test_benchmark_pc.log
else
    source ../common/test_benchmark.sh Pi $img_size $benchmark_time_s 2>&1 | tee test_benchmark_pi.log
fi
