lib:
	nvcc -shared -Xcompiler "-fPIC" test_lib.cu -o test_lib.so -cudart shared

all: lib
	gcc load_test_lib.c -ldl -o load_test_lib.out -I/usr/local/cuda-11.3/targets/x86_64-linux/include
