#!/bin/bash

git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
mkdir free-spoken-digit-dataset/recordings/test_data
mkdir free-spoken-digit-dataset/recordings/train_data
mv free-spoken-digit-dataset/recordings/*_4?.wav free-spoken-digit-dataset/recordings/test_data
mv free-spoken-digit-dataset/recordings/*.wav free-spoken-digit-dataset/recordings/train_data
