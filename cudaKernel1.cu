#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__device__ int floor_device(float x) {
    return floor(x);
}

__device__ float fmod_device(float x, float y) {
    return fmod(x, y);
}

float computeCeil(float num) {
    return ceilf(num);
}

__device__ float ceil_device(float num) {
    return ceilf(num);
}

__global__ void setNegativeToZero(float* restored, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        restored[idx] = (restored[idx] < 0) ? 0 : restored[idx];
    }
}

__global__ void findMaxAndAssign(float* restored, float* maximum, int rows, int cols, int ind) {
    extern __shared__ float sharedData[];
	
	int bid = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int size_max = ceil_device(rows*cols/1024);

    if (idx < rows*cols){
		sharedData[tid] = restored[idx];
		
		for (int d = blockDim.x / 2; d > 0; d = d/2) {
			__syncthreads();
			if (tid < d) {
				sharedData[tid] = max(sharedData[tid], sharedData[tid + d]);
			}
		}

		if (tid == 0) {
			maximum[bid+ind*size_max] = sharedData[0];
		}
	}
}

__global__ void normalisation(float* cleaned, float* ma, int arraySize) {
    int idx =  blockDim.x * blockIdx.x + threadIdx.x;

    if (ma[0] != 0) {
        if (idx < arraySize) {
            cleaned[idx] = cleaned[idx] / ma[0];
        }
    }
}


__global__ void split(float* data_1, float* data_2, float* result, int size, int unit_size, int ima, int unit_num) {

    extern  __shared__  float sharedNumDen[];
    
    int bid = blockIdx.x; // tile index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    
    int i_id = floor_device(bid/unit_num);
    int j_id = fmod_device(bid,unit_num);
    int factor = ceil_device(unit_size*unit_size/1024);
    
    float C = 1e-4;
    int I_id;
    int J_id;
    float sum1;
    float sub1;
    int rows;
    int cols;
    
    for (int fac = 1; fac<= factor;fac = fac + 1){
		if (tid*factor < unit_size*unit_size){
			if (fac == 1) {
				sharedNumDen[tid] = 0; // Num
				sharedNumDen[tid+1024] = 0; // Den
			}
			rows = floor_device(tid*fac/unit_size);
			cols = fmod_device(tid*fac,unit_size);
			
			I_id = i_id * unit_size + rows;
			J_id = j_id * unit_size + cols;
			sum1 = data_1[I_id * ima + J_id] + data_2[I_id * ima + J_id];
			sub1 = data_1[I_id * ima + J_id] - data_2[I_id * ima + J_id];
			
			sharedNumDen[tid] = sharedNumDen[tid] + abs(sum1*sub1);
			sharedNumDen[tid+1024] = sharedNumDen[tid+1024] + data_1[I_id * ima + J_id] + data_2[I_id * ima + J_id];
		}
	}
    
    for (int d = blockDim.x/2;d>0;d = d/2){
		__syncthreads();
		if (tid<d) {
			sharedNumDen[tid] += sharedNumDen[tid+d];
			sharedNumDen[tid+1024] += sharedNumDen[tid+1024+d];
		}
	}
	
	if (tid==0) {
		result[bid] = 1-sharedNumDen[0]/(sharedNumDen[1024] + C);
	}
}

int splitaug(float* input_data_1, float* input_data_2, float* result_array, int size) {
    
    float* d_data_1;
    float* d_data_2;
    float* result_data;
    float* maximum;
    float* max_out;
    cudaError_t cudaStatus;

    int unit_size = 60;
    int ima = 3000;
    int ima2 = computeCeil(ima*ima/1024)*2;
    int ima3 = computeCeil(ima2/1024);
    int ima4 = computeCeil(ima3/1024);
    int unit_num = ima/unit_size;
    float maximum_out[1] = {0.0f};// max of the two input images
    float hostInitialValues[ima2];
    for (int i = 0; i < ima2; ++i) {
		hostInitialValues[i] = 0.0f;//max of each row of the two images, the row is not definitely the row of the image, but in the unit of block
    }
    float maximum_out2[ima3];//max of the images respectively
    float maximum_out3[ima4];//max of the images respectively
    
    //if (ima2 > 1024){
		float* max_out2;
		for (int i = 0; i < ima3; ++i) {
			maximum_out2[i] = 0.0f;
		}
		cudaMalloc((void**)&max_out2,  ima3 * sizeof(float));
		cudaMemcpy(max_out2, maximum_out2, ima3 * sizeof(float), cudaMemcpyHostToDevice);
		//if (ima3 > 1024){
			float* max_out3;
			for (int i = 0; i < ima4; ++i) {
				maximum_out3[i] = 0.0f;
			}
			cudaMalloc((void**)&max_out3,  ima4 * sizeof(float));
			cudaMemcpy(max_out3, maximum_out3, ima4 * sizeof(float), cudaMemcpyHostToDevice);
		//}
	//}
    
    cudaMalloc((void**)&d_data_1, size * sizeof(float));
    cudaMalloc((void**)&d_data_2,  size * sizeof(float));
    cudaMalloc((void**)&result_data, unit_num * unit_num * sizeof(float));
    cudaMalloc((void**)&maximum,  ima2 * sizeof(float));
    cudaMalloc((void**)&max_out,  1 * sizeof(float));
    
    cudaMemcpy(d_data_1, input_data_1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_2, input_data_2, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_data, result_array, unit_num * unit_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(maximum, hostInitialValues, ima2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(max_out, maximum_out, 1 * sizeof(float), cudaMemcpyHostToDevice);
    
    
    int num_threads = 1024;
    int num_blocks = computeCeil(ima*ima/num_threads);
    
    setNegativeToZero<<<num_blocks,num_threads>>>(d_data_1, ima, ima);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));
	}
    setNegativeToZero<<<num_blocks,num_threads>>>(d_data_2, ima, ima);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));
	}
    
    int shared_mem_size = num_threads * sizeof(float);
    findMaxAndAssign<<<num_blocks,num_threads,shared_mem_size>>>(d_data_1, maximum, ima, ima, 0);
    cudaDeviceSynchronize();
    findMaxAndAssign<<<num_blocks,num_threads,shared_mem_size>>>(d_data_2, maximum, ima, ima, 1);
    cudaDeviceSynchronize();
    
    if (ima2 <= 1024){
		num_blocks = 1;
		findMaxAndAssign<<<num_blocks,num_threads,shared_mem_size>>>(maximum, max_out, ima2, 1, 0); 
		cudaDeviceSynchronize();
	} else {
		num_blocks = ima3;
		findMaxAndAssign<<<num_blocks,num_threads,shared_mem_size>>>(maximum, max_out2, ima2, 1, 0); 
		cudaDeviceSynchronize();
		if (ima3 <= 1024){
			num_blocks = 1;
			findMaxAndAssign<<<num_blocks,num_threads,shared_mem_size>>>(max_out2, max_out, ima3, 1, 0); 
			cudaDeviceSynchronize();
		} else {
			num_blocks = ima4;
			findMaxAndAssign<<<num_blocks,num_threads,shared_mem_size>>>(max_out2, max_out3, ima3, 1, 0); 
			cudaDeviceSynchronize();
			num_blocks = 1;
			findMaxAndAssign<<<num_blocks,num_threads,shared_mem_size>>>(max_out3, max_out, ima4, 1, 0);  // default: ima4 <= 1024
			cudaDeviceSynchronize();
		}
	}
    
    num_threads = 1024;
    num_blocks = computeCeil(ima*ima/num_threads);
    normalisation<<<num_blocks,num_threads>>>(d_data_1, max_out, size);
    cudaDeviceSynchronize();
    normalisation<<<num_blocks,num_threads>>>(d_data_2, max_out, size);
    cudaDeviceSynchronize();
    
    num_threads = 1024;
    num_blocks = unit_num*unit_num;
    shared_mem_size = 2*num_threads * sizeof(float);
    split<<<num_blocks,num_threads,shared_mem_size>>>(d_data_1, d_data_2, result_data, size, unit_size, ima, unit_num);
    cudaDeviceSynchronize();

	cudaMemcpy(result_array, result_data, unit_num * unit_num * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaFree(result_data);
    return 0;
}
