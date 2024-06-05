INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart -lcurand -lcfitsio
GCC := g++
NVCC := ${CUDA_HOME}/bin/nvcc

GCC_OPTS :=-O3 -fPIC -Wall -Wextra $(INC) -std=c++11
NVCCFLAGS :=-O3 -gencode arch=compute_90,code=sm_90 --ptxas-options=-v -Xcompiler -fPIC -Xcompiler -Wextra -lineinfo $(INC) $(LIB)

all: clean sharedlibrary_gpu

sharedlibrary_gpu: trigger.o triggerKernel.o
	$(NVCC) -o sharedlibrary_gpu $(NVCCFLAGS) trigger.o triggerKernel.o

trigger.o: trigger.cpp
	$(GCC) -c trigger.cpp $(GCC_OPTS) -o trigger.o

triggerKernel.o: triggerKernel.cu
	$(NVCC) -c triggerKernel.cu  $(NVCCFLAGS) -o triggerKernel.o

clean:	
	rm -f *.o *.so
