include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	$(CC) $(CUDA_GENCODE_FLAGS) -cuda nn_cuda.cu
	$(CC) $(CUDA_CUDART_FLAGS) $(CUDA_GENCODE_FLAGS) -o nn nn_cuda.cu

clang: $(SRC)
	clang++ nn_cuda.cu -o nn -I../util --cuda-gpu-arch=$(CUDA_CLANG_ARCH) \
		-L$(CUDA_LIB_DIR) -lcudart -DTIMING

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
