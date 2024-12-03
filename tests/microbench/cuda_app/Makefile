.PHONY: all clean

CC = nvcc -ccbin g++
CFLAGS = -cudart shared -lcudnn -O2
SRC = $(wildcard *.cu)
OBJ = $(patsubst %.cu, %.out, $(SRC))

all: $(OBJ)

%.out: %.cu
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.out
