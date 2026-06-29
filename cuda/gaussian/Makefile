include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(CUDA_CUDART_FLAGS) $(KERNEL_DIM) $(CUDA_GENCODE_FLAGS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clang: $(SRC)
	clang++ $(SRC) -o $(EXE) -I../util --cuda-gpu-arch=$(CUDA_CLANG_ARCH) \
		-L$(CUDA_LIB_DIR) -lcudart -DTIMING

clean:
	rm gaussian
