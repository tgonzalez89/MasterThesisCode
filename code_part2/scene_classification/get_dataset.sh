#!/bin/bash

# Manually download from https://www.kaggle.com/puneet6060/intel-image-classification/notebooks
mkdir dataset
mv archive.zip dataset/
cd dataset
unzip archive.zip
mv seg_test tmp
mv tmp/seg_test .
rm -rf tmp
mv seg_train tmp
mv tmp/seg_train .
rm -rf tmp
mv seg_pred tmp
mv tmp/seg_pred .
rm -rf tmp
rm -rf archive.zip
