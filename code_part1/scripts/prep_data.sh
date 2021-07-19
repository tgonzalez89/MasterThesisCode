#!/bin/bash

app=sobel
echo "Creating data for $app"
rm -rf $app/data
mkdir -p $app/data
for f in `ls axbench/applications/$app/t*.data/input/*.rgb`
do
    name=$(basename $f)
    name=${name%.*}
    echo "Converting $f to png..."
    ./scripts/rgb2png.py $f $app/data/$name.png
done
# 4K image for testing purposes
wget -O $app/data/test.jpg https://photos.smugmug.com/Wallpapers/i-nFqSkFs/0/68bfceaa/O/HDRshooter-4K-wallpaper-067-3840x2160.jpg

app=kmeans
echo "Creating data for $app"
rm -rf $app/data
mkdir -p $app/data
cp -r sobel/data/* $app/data

app=jmeint
echo "Creating data for $app"
rm -rf $app/data
mkdir -p $app/data
cp axbench/applications/$app/train.data/input/jmeint_10K.data $app/data/inputs_40K.txt
tail -n 10000 $app/data/inputs_40K.txt >> $app/data/inputs_40K.txt
tail -n 10000 $app/data/inputs_40K.txt >> $app/data/inputs_40K.txt
tail -n 10000 $app/data/inputs_40K.txt >> $app/data/inputs_40K.txt
sed -i 's/10000/40000/' $app/data/inputs_40K.txt
cp axbench/applications/$app/test.data/input/jmeint_1000K.data $app/data/inputs_1000K.txt
cd axbench/applications/$app
./bin/$app.out ../../../$app/data/inputs_40K.txt ../../../$app/data/outputs_40K.txt
./bin/$app.out ../../../$app/data/inputs_1000K.txt ../../../$app/data/outputs_1000K.txt
cd ../../..

app=blackscholes
echo "Creating data for $app"
rm -rf $app/data
mkdir -p $app/data
cp axbench/applications/$app/train.data/input/blackscholesTrain_100K.data $app/data/inputs_120K.txt
tail -n 20000 $app/data/inputs_120K.txt >> $app/data/inputs_120K.txt
sed -i 's/100000/120000/' $app/data/inputs_120K.txt
cp axbench/applications/$app/test.data/input/blackscholesTest_200K.data $app/data/inputs_200K.txt
cd axbench/applications/$app
./bin/$app.out ../../../$app/data/inputs_120K.txt ../../../$app/data/outputs_120K.txt > ../../../$app/data/inputs_nn_120K.txt
./bin/$app.out ../../../$app/data/inputs_200K.txt ../../../$app/data/outputs_200K.txt > ../../../$app/data/inputs_nn_200K.txt
cd ../../..

app=inversek2j
echo "Creating data for $app"
rm -rf $app/data
mkdir -p $app/data
cp axbench/applications/$app/train.data/input/theta_100K.data $app/data/inputs_120K.txt
tail -n 20000 $app/data/inputs_120K.txt >> $app/data/inputs_120K.txt
sed -i 's/100000/120000/' $app/data/inputs_120K.txt
cp axbench/applications/$app/test.data/input/theta_1000K.data $app/data/inputs_1000K.txt
cd axbench/applications/$app
./bin/$app.out ../../../$app/data/inputs_120K.txt ../../../$app/data/outputs_120K.txt > ../../../$app/data/inputs_nn_120K.txt
./bin/$app.out ../../../$app/data/inputs_1000K.txt ../../../$app/data/outputs_1000K.txt > ../../../$app/data/inputs_nn_1000K.txt
cd ../../..

