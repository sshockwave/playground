#include <bits/stdc++.h>
#include "utils.cpp"
#include "cuda_runtime.h"
#include "nccl.h"

using namespace std;

Clock tim;

typedef double db;

const int threadsPerBlock=256;

const int device_cnt=8;
size_t data_len=1<<20;
int virt_dev[device_cnt];

inline bool db_same(db a,db b){
	return fabs(a-b)<1e-9;
}

inline void get_devices(){
	int cnt=0;
	cudaGetDeviceCount(&cnt);
	assert(cnt>0);
	cout<<"Device list:{";
	for(int i=0;i<device_cnt;i++){
		virt_dev[i]=i%cnt;
		cout<<virt_dev[i]<<(i<device_cnt-1?",":"}");
	}
	cout<<endl;
}

inline void run_gpu(){
	db *input,*d_dest[device_cnt];
	cudaStream_t stream[device_cnt];
	const int root=0;
	cudaSetDevice(virt_dev[root]);
	cudaMalloc(&input,data_len*sizeof(db));
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamCreate(&stream[i]);
		cudaMalloc(&d_dest[i],data_len*sizeof(db));
	}
	tim.tic();
	for(int i=0;i<device_cnt;i++){
		cudaMemcpyAsync(d_dest[i],input,data_len*sizeof(db),cudaMemcpyDeviceToDevice,stream[i]);
	}
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamSynchronize(stream[i]);
	}
	cout<<"run_gpu:"<<tim.toc()<<endl;
	cudaSetDevice(virt_dev[root]);
	cudaFree(input);
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaFree(d_dest[i]);
		cudaStreamDestroy(stream[i]);
	}
}

//Not working for data_len>=29
inline void run_nccl(){
	const int root=0;
	db *input,*d_dest[device_cnt];
	cudaStream_t stream[device_cnt];
	ncclComm_t comm[device_cnt];
	ncclUniqueId id;
	ncclGetUniqueId(&id);
	ncclGroupStart();
	cudaSetDevice(virt_dev[root]);
	cudaMalloc(&input,data_len*sizeof(db));
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamCreate(&stream[i]);
		cudaMalloc(&d_dest[i],data_len*sizeof(db));
		ncclCommInitRank(&comm[i],device_cnt,id,i);
	}
	ncclGroupEnd();
	tim.tic();
	ncclGroupStart();
	for(int i=0;i<device_cnt;i++){
		ncclBroadcast(input,d_dest[i],data_len,ncclDouble,root,comm[i],stream[i]);
	}
	ncclGroupEnd();
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamSynchronize(stream[i]);
	}
	cout<<"run_nccl:"<<tim.toc()<<endl;
	cudaSetDevice(virt_dev[root]);
	cudaFree(input);
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaFree(d_dest[i]);
		cudaStreamDestroy(stream[i]);
		ncclCommDestroy(comm[i]);
	}
}

int main(){
	cout<<"data_len exp:";
	cin>>data_len;
	data_len=1ll<<data_len;
	get_devices();
	run_gpu();
	run_nccl();
}
