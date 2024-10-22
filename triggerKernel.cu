#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cmath>

__device__ size_t floor_device(float x) {
    return floor(x);
}

size_t computefloor(float x) {
    return floor(x);
}

__device__ float fmod_device(float x, float y) {
    return fmod(x, y);
}

size_t computeCeil(float num) {
    return ceilf(num);
}

__device__ size_t ceil_device(float num) {
    return ceilf(num);
}

__global__ void setNegativeToZero(float* restored, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        restored[idx] = (restored[idx] < 0) ? 0 : restored[idx];
    }
}

__global__ void setNonPositiveToC(float* restored, size_t size, float C) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        restored[idx] = (restored[idx] <= 0) ? C : restored[idx];
    }
}

__global__ void subtraction_array(float* data_1, float* data_2, float* diff_out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        diff_out[idx] = abs(data_1[idx] - data_2[idx]);
    }
}


__global__ void tlisi(float* diff_out, float* snap, float* result, size_t size, size_t unit_size, size_t ima, size_t unit_num, float maxall) {

    extern  __shared__  float sharedNumDen[];
    
    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    
    size_t i_id = floor_device(bid/unit_num);
    size_t j_id = fmod_device(bid,unit_num);
    size_t factor = ceil_device(static_cast<float>(unit_size*unit_size)/1024);
    
    size_t I_id;
    size_t J_id;
    size_t rows;
    size_t cols;
    
    for (size_t fac = 1; fac <= factor;fac = fac + 1){
        if (tid+(fac-1)*1024 < unit_size*unit_size){
            if (fac == 1) {
                sharedNumDen[tid] = 0; // Sum of diff_out
                sharedNumDen[tid+1024] = 0; // Max of diff_out
                sharedNumDen[tid+2048] = 0; // Sum of r
            }
            rows = floor_device((tid+(fac-1)*1024)/unit_size);
            cols = fmod_device((tid+(fac-1)*1024),unit_size);
			
            I_id = i_id * unit_size + rows;
            J_id = j_id * unit_size + cols;
			
            sharedNumDen[tid] = sharedNumDen[tid] + diff_out[I_id * ima + J_id];
            sharedNumDen[tid+1024] = max(sharedNumDen[tid+1024], diff_out[I_id * ima + J_id]);
            sharedNumDen[tid+2048] = (diff_out[I_id * ima + J_id]/snap[I_id * ima + J_id] < 1) ? sharedNumDen[tid+2048] +  diff_out[I_id * ima + J_id]/snap[I_id * ima + J_id]: sharedNumDen[tid+2048] + 1;
        } else {
            if (fac == 1) {
                sharedNumDen[tid] = 0; // Sum of diff_out
                sharedNumDen[tid+1024] = 0; // Max of diff_out
                sharedNumDen[tid+2048] = 0; // Sum of r
            }
        }
    }
    
    for (size_t d = blockDim.x/2;d>0;d = d/2){
        __syncthreads();
        if (tid<d) {
            sharedNumDen[tid] += sharedNumDen[tid+d];
            sharedNumDen[tid+1024] = max(sharedNumDen[tid+1024], sharedNumDen[tid+1024+d]);
            sharedNumDen[tid+2048] += sharedNumDen[tid+2048+d];
        }
    }
	
    if (tid==0) {
        result[bid] = 1-(sharedNumDen[0]/unit_size/unit_size)*sharedNumDen[1024]*(sharedNumDen[2048]/unit_size/unit_size)/maxall/maxall;
    }
}

int splittlisi(float* input_data_1, float* input_data_2, float* input_snap, float* result_array, size_t size, size_t unit_size, size_t ima, float maxall) {
    
    float* d_data_1;
    float* d_data_2;
    float* diff_out;
    float* snap_data;
    float* result_data;
    cudaError_t cudaStatus;
    float C = 1e-6;
    
    size_t unit_num = ima/unit_size;
    float *diff_in = (float *) malloc(size * sizeof(float));
    
    cudaMalloc((void**)&d_data_1, size * sizeof(float));
    cudaMalloc((void**)&d_data_2,  size * sizeof(float));
    cudaMalloc((void**)&diff_out,  size * sizeof(float));
    cudaMalloc((void**)&snap_data,  size * sizeof(float));
    cudaMalloc((void**)&result_data, unit_num * unit_num * sizeof(float));
    
    cudaMemcpy(d_data_1, input_data_1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_2, input_data_2, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(diff_out, diff_in, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(snap_data, input_snap, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_data, result_array, unit_num * unit_num * sizeof(float), cudaMemcpyHostToDevice);
    
    size_t num_threads = 1024;
    size_t num_blocks = computeCeil(static_cast<float>(size)/num_threads);
    setNegativeToZero<<<num_blocks,num_threads>>>(d_data_1, size);
	
    setNegativeToZero<<<num_blocks,num_threads>>>(d_data_2, size);
    
    setNonPositiveToC<<<num_blocks,num_threads>>>(snap_data, size, C);

    num_threads = 1024;
    num_blocks = computeCeil(static_cast<float>(size)/num_threads);
    subtraction_array<<<num_blocks,num_threads>>>(d_data_1, d_data_2, diff_out, size);
    
    num_threads = 1024;
    num_blocks = unit_num*unit_num;
    size_t shared_mem_size = 3 * num_threads * sizeof(float);
    tlisi<<<num_blocks,num_threads,shared_mem_size>>>(diff_out, snap_data, result_data, size, unit_size, ima, unit_num, maxall);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error 13 : %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess){
        printf("ERROR! GPU Kernel error.\n");
        printf("CUDA error code: %d; string: %s;\n", (int) cudaError, cudaGetErrorString(cudaError));
    }
    else {
        printf("No CUDA error.\n");
    }

    cudaMemcpy(result_array, result_data, unit_num * unit_num * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaFree(result_data);
    return 0;
}
