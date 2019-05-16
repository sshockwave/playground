#include <bits/stdc++.h>
#include "utils.cpp"
#include "cuda_runtime.h"
#include "nccl.h"

using namespace std;

Clock tim;

typedef double db;

const int threadsPerBlock=256;

const int device_cnt=8;
int data_len=1<<20;
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

double *inputs[device_cnt];

double *cpu_output;
double *gpu_output;
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
	fill_n(cpu_output,data_len,0);
	tim.tic();
	for(int i=0;i<device_cnt;i++){
		for(int j=0;j<data_len;j++){
			cpu_output[j]+=inputs[i][j];
		}
	}
	cout<<"run_cpu:"<<tim.toc()<<endl;
}

__global__ void doAdd(db* inputs,db* out,int data_len){
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	db sum=0;
	for(int j=0;j<device_cnt;j++){
		sum+=inputs[j*data_len+i];
	}
	out[i]=sum;
}

inline void run_gpu(){
	db *d_input[device_cnt];
	cudaStream_t stream[device_cnt];
	const int root=0;
	db *gather,*dest;
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamCreate(&stream[i]);
		cudaMalloc(&d_input[i],data_len*sizeof(db));
		cudaMemcpyAsync(d_input[i],inputs[i],data_len*sizeof(db),cudaMemcpyHostToDevice,stream[i]);
	}
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamSynchronize(stream[i]);
	}
	cudaSetDevice(virt_dev[root]);
	cudaMalloc(&gather,data_len*device_cnt*sizeof(db));
	cudaMalloc(&dest,data_len*sizeof(db));
	tim.tic();
	for(int i=0;i<device_cnt;i++){
		cudaMemcpyAsync(gather+i*data_len,d_input[i],data_len*sizeof(db),cudaMemcpyDeviceToDevice,stream[i]);
	}
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaStreamSynchronize(stream[i]);
	}
	cudaSetDevice(virt_dev[root]);
	doAdd<<<data_len/threadsPerBlock,threadsPerBlock>>>(gather,dest,data_len);
	cout<<"run_gpu:"<<tim.toc()<<endl;
	gpu_output=new db[data_len];
	cudaMemcpy(gpu_output,dest,data_len*sizeof(db),cudaMemcpyDeviceToHost);
	cudaFree(dest);
	cudaFree(gather);
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaFree(d_input[i]);
		cudaStreamDestroy(stream[i]);
	}
	cout<<"\tchecking gpu.."<<endl;
	for(int i=0;i<data_len;i++){
		if(!db_same(gpu_output[i],cpu_output[i])){
			cout<<"i="<<i<<"\tgpu:"<<gpu_output[i]<<"\tcpu:"<<cpu_output[i]<<endl;
		}
		assert(db_same(gpu_output[i],cpu_output[i]));
	}
	cout<<"\tpassed"<<endl;
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
		cudaMemcpyAsync(d_input[i],inputs[i],data_len*sizeof(db),cudaMemcpyHostToDevice,stream[i]);
		ncclCommInitRank(&comm[i],device_cnt,id,i);
	}
	ncclGroupEnd();
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
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
		cudaSetDevice(virt_dev[i]);
		cudaStreamSynchronize(stream[i]);
	}
	cout<<"run_nccl:"<<tim.toc()<<endl;
	cudaSetDevice(virt_dev[root]);
	nccl_output=new db[data_len];
	cudaMemcpy(nccl_output,dest,data_len*sizeof(db),cudaMemcpyDeviceToHost);
	cudaFree(dest);
	for(int i=0;i<device_cnt;i++){
		cudaSetDevice(virt_dev[i]);
		cudaFree(d_input[i]);
		cudaStreamDestroy(stream[i]);
		ncclCommDestroy(comm[i]);
	}
	cout<<"\tchecking nccl..."<<endl;
	for(int i=0;i<data_len;i++){
		assert(db_same(nccl_output[i],cpu_output[i]));
	}
	cout<<"\tpassed"<<endl;
}

int main(){
	cout<<"data_len exp:";
	cin>>data_len;
	data_len=1<<data_len;
	get_devices();
	get_data();
	run_cpu();
	run_gpu();
	run_nccl();
}
