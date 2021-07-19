#!/bin/bash

for app in blackscholes inversek2j jmeint kmeans sobel; do
    echo "Running $app"
    cd $app
    ../scripts/redirect.sh ./run.sh results.txt
    cd ..
done

