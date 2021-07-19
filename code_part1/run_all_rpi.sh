#!/bin/bash

for app in blackscholes inversek2j jmeint kmeans sobel; do
    echo "Running $app"
    cd $app
    ../scripts/redirect.sh ./run_rpi.sh results_rpi.txt
    cd ..
done

