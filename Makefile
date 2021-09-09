CC=gcc
MPICC=mpicc
CFLAGS=-O3
CBLAS=-lopenblas -lpthread -lm
N=2

default: all

all: v0

v0: v0.c utilities.c timer.c controller.c
	$(CC) $(CFLAGS) $(CBLAS) -o $@ $^

v1: v1.c utilities.c timer.c controller.c read.c
	$(MPICC) $(CFLAGS) $(CBLAS) -o $@ $^

v2: v2.c utilities.c timer.c controller.c read.c
	$(MPICC) $(CFLAGS) $(CBLAS) -o $@ $^

clean:
	rm -f v0 v1 v2




