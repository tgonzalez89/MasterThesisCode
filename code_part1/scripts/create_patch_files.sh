#!/bin/bash

cd ~/axbench
files=$(git status | grep modified | sed -E 's/\s+modified:\s+//' | xargs)
cd -
rm -rf axbench_patch_files
mkdir axbench_patch_files
cd axbench_patch_files
for f in $files
do
    mkdir -p $(dirname $f)
    diff ~/axbench.orig/$f ~/axbench/$f > $f
done
