g++ -O3 -Wall -std=c++11 -I/usr/local/include -c tritri.cpp -o tritri.o
g++ -O3 -Wall -std=c++11 -I/usr/local/include -c jmeint-orig.cpp -o jmeint-orig.o
g++ tritri.o jmeint-orig.o -O3 -Wall -std=c++11 -L/usr/local/lib -o jmeint-orig.out
rm -rf *.o
