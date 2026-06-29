include ../../common/make.config

# C compiler
CC = gcc
CXX = g++
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include $(CUDA_GENCODE_FLAGS) -DGPU
CUDA_LIB_DIR ?= $(CUDA_DIR)/lib64

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif


backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o backprop_result.o
	$(CXX) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o backprop_result.o -o backprop -L$(CUDA_LIB_DIR) -lcudart -lcuda -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

backprop_result.o: backprop_result.c backprop_result.h backprop.h
	$(CC) $(CC_FLAGS) backprop_result.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
