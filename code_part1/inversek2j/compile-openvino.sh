#!/bin/bash

arch=`lscpu | grep Architecture | awk '{print $2}'`
if [ "$arch" == "x86_64" ]
then
    march="x86-64"
    arch=intel64
elif [ "$arch" == "armv7l" ]
then
    march="armv7-a"
else
    echo "ERROR: Unsupported arch '$arch'"
    exit
fi
name=$(basename $1)
name=${name%.*}
dir=$(dirname $1)
name="$dir/$name"
rm $name.out
g++ -O3 -Wall -march=$march -std=c++11 -I/usr/local/include -c kinematics.cpp -o kinematics.o
g++ -O3 -Wall -march=$march -std=c++11 -I/opt/intel/openvino/deployment_tools/inference_engine/include -I/opt/intel/openvino/opencv/include -c $1 -o $name.o
g++ -O3 -Wall -march=$march -std=c++11 kinematics.o $name.o /opt/intel/openvino/deployment_tools/inference_engine/lib/$arch/libinference_engine.so /opt/intel/openvino/opencv/lib/libopencv_core.so /opt/intel/openvino/opencv/lib/libopencv_imgcodecs.so /opt/intel/openvino/opencv/lib/libopencv_imgproc.so -o $name.out
rm -rf *.o
