INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart -lcurand
GCC := gcc
NVCC := ${CUDA_HOME}/bin/nvcc

GCC_OPTS :=-O0 -fPIC -Wall -Wextra $(INC)
NVCCFLAGS :=-O0 -arch=sm_86 --ptxas-options=-v -Xcompiler -fPIC -Xcompiler -Wextra -lineinfo $(INC) $(LIB)

all: clean sharedlibrary_gpu

sharedlibrary_gpu: trigger.o cudaKernel1.o
	$(NVCC) -g -o sharedlibrary_gpu $(NVCCFLAGS) trigger.o cudaKernel1.o

trigger.o: trigger.cpp
	$(GCC) -g -c trigger.cpp $(GCC_OPTS) -o trigger.o

cudaKernel1.o: cudaKernel1.cu
	$(NVCC) -g -c cudaKernel1.cu  $(NVCCFLAGS) -o cudaKernel1.o

clean:	
	rm -f *.o *.so