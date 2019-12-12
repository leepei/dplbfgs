CC = mpicc
CXX = mpicxx
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 3
OS = $(shell uname)
LIBS = -lblas -llapack

all: train predict

train: linear.o train.c
	$(CXX) $(CFLAGS) -o train train.c linear.o $(LIBS)

predict: linear.o predict.c
	$(CXX) $(CFLAGS) -o predict predict.c linear.o $(LIBS)

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

clean:
	rm -f *~ linear.o train predict
