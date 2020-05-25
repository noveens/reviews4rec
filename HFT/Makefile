CC = g++
CFLAGS = -Wall -O3 -fopenmp -Iliblbfgs-1.10/include -Igzstream
LDFLAGS = -llbfgs -lgomp -lgzstream -lz -lstdc++ -Lliblbfgs-1.10/lib/.libs -Lgzstream

all: train

liblbfgs-1.10/lib/.libs/liblbfgs.so:
	tar xzvvf liblbfgs-1.10.tar.gz
	cd liblbfgs-1.10 && ./configure && make

gzstream/gzstream.o:
	tar xzvvf gzstream.tgz
	cd gzstream && make

train: language.cpp language.hpp common.hpp liblbfgs-1.10/lib/.libs/liblbfgs.so gzstream/gzstream.o
	$(CC) $(CFLAGS) -o train language.cpp gzstream/gzstream.o $(LDFLAGS)

clean:
	rm train
