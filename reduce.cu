#include <bits/stdc++.h>
#include "utils.cpp"
#include "cuda_runtime.h"
#include "nccl.h"

Clock tim;

typedef double db;

const int device_cnt=8;
const int data_len=1e7;
int virt_dev[device_cnt];

inline void get_devices(){
	int cnt=0;
	cudaGetDeviceCount(&cnt);
	assert(cnt>0);
	for(int i=0;i<device_cnt;i++){
		virt_dev[i]=i%cnt;
	}
}

double *inputs[device_cnt];

double *cpu_output;
double *nccl_output;

inline void get_data(){
	for(int i=0;i<device_cnt;i++){
		inputs[i]=new db[data_len];
		for(int j=0;j<data_len;j++){
			static uniform_real_distribution<db>rnd(0,1);
			static default_random_engine eng;
			inputs[i][j]=rnd(eng);
		}
	}
	cout<<"get_data complete"<<endl;
}

inline void run_cpu(){
	cpu_output=new db[data_len];
	fill_n(cpu_output,n,0);
	tim.tic();
	for(int i=0;i<device_cnt;i++){
		for(int j=0;j<data_len;j++){
			cpu_output[j]+=inputs[i][j];
		}
	}
	cout<<"run_cpu:"<<tim.tok()<<endl;
}

inline void run_nccl(){
	db *d_input[device_cnt];
	cudaStream_t stream[device_cnt];
	ncclComm_t comm[device_cnt];
	ncclUniqueId id;
	ncclGetUniqueId(&id);
	ncclGroupStart();
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamCreate(&stream[i]);
		cudaMalloc(&d_input[i],data_len*sizeof(db));
		cudaMemcpy(input,d_input,data_len*sizeof(db),cudaMemcpyHostToDevice,stream[i]);
		ncclCommInitRank(comm+i,device_cnt,id,i);
	}
	ncclGroupEnd();
	for(int i=0;i<device_cnt;i++){
		cudaStreamSynchronize(stream[i]);
	}
	//data is ready on gpu
	const int root=0;
	cudaSetDevice(virt_dev[root]);
	db *dest;
	cudaMalloc(&dest,data_len*sizeof(db));
	tim.tic();
	ncclGroupStart();
	for(int i=0;i<device_cnt;i++){
		ncclReduce(d_input[i],dest,data_len,ncclDouble,ncclSum,root,comm[i],stream[i]);
	}
	ncclGroupEnd();
	for(int i=0;i<device_cnt;i++){
		cudaStreamSynchronize(stream[i]);
	}
	cout<<"run_nccl:"<<tim.tok()<<endl;
	gpu_output=new db[data_len];
	cudaMemcpy(gpu_output,d_input[i],);
}

int main(){
	get_devices();
	get_data();
	run_cpu();
	run_nccl();
}
