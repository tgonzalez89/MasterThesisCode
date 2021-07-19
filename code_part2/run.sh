#!/bin/bash

for dir in audio_classification fruit_classification scene_classification
do
    cd $dir
    source ./train.sh
    cd ..
done

for dir in audio_classification fruit_classification scene_classification
do
    cd $dir
    source ./test_accuracy.sh
    cd ..
done

for dir in audio_classification fruit_classification scene_classification
do
    cd $dir
    source ./test_benchmark.sh
    cd ..
done
