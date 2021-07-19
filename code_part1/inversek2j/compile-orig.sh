g++ -O3 -Wall -std=c++11 -I/usr/local/include -c kinematics.cpp -o kinematics.o
g++ -O3 -Wall -std=c++11 -I/usr/local/include -c inversek2j-orig.cpp -o inversek2j-orig.o
g++ kinematics.o inversek2j-orig.o -O3 -Wall -std=c++11 -L/usr/local/lib -o inversek2j-orig.out
rm -rf *.o
