#!/bin/bash

#rm -rf axbench
if [ ! -d "axbench" ]
then
    echo "Getting axbench"
    git clone https://bitbucket.org/act-lab/axbench.git
    #cp -r ~/axbench.orig axbench

    echo "Patching files"
    cur_dir=$(echo "$(pwd)" | sed 's/\//\\\//g')
    sed -E -i 's/BASE_DIR    := .+/BASE_DIR    := '"${cur_dir}\/axbench/" axbench/applications/config.mk

    cd axbench_patch_files
    patch_files=$(find -type f | xargs)
    cd ..
    for f in $patch_files
    do
        patch axbench/$f axbench_patch_files/$f
    done

    echo "Compiling applications"
    for app in `ls -d axbench/applications/* | grep -v sobel | xargs`
    do
        if [ -d $app ]
        then
            cd $app
            make
            cd ../../..
        fi
    done
fi
