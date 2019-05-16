.PHONY: clean

all: reduce broadcast

broadcast: broadcast.cu utils.cpp
	nvcc -O3 -std=c++11 $< -o $@ -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 /usr/lib/x86_64-linux-gnu/libnccl_static.a

reduce: reduce.cu utils.cpp
	nvcc -O3 -std=c++11 reduce.cu -o $@ -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 /usr/lib/x86_64-linux-gnu/libnccl_static.a

clean:
	rm -rf reduce broadcast
