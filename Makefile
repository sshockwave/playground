reduce: reduce.cu utils.cpp
	nvcc -O3 -std=c++11 reduce.cu -o $@ -lglog -lgflags -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60
