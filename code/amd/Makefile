SOURCE_DIR?=.
BIN_DIR?=bin
MKDIR=mkdir -p

ROCM_PATH=/opt/rocm

#hipcc
HIPCC=${ROCM_PATH}/bin/hipcc

#BLAS
ROCBLAS_LIB=rocblas

#CONV
MIOPEN_LIB?=MIOpen

#DeepBench
DEEPBENCH_INC=${SOURCE_DIR}/../kernels

RCCL_INC_DIR=${ROCM_PATH}/rccl/include
RCCL_LIB_DIR=${ROCM_PATH}/rccl/lib
RCCL_LIB=rccl

all: conv rnn gemm rccl_single

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-O3

conv:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/conv_bench_rocm.cpp -o $(BIN_DIR)/conv_bench -I$(DEEPBENCH_INC) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

rnn:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rnn_bench_rocm.cpp -o $(BIN_DIR)/rnn_bench -I$(DEEPBENCH_INC) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

gemm:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/gemm_bench.cpp -o $(BIN_DIR)/gemm_bench -I$(DEEPBENCH_INC) -l$(ROCBLAS_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

rccl_single:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rccl_single_all_reduce.cpp -o $(BIN_DIR)/rccl_single_all_reduce -I$(DEEPBENCH_INC) -I$(RCCL_INC_DIR) -L$(RCCL_LIB_DIR) -l$(RCCL_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

clean:
	rm -rf $(BIN_DIR)
