SOURCE_DIR?=.
BIN_DIR?=bin
MKDIR=mkdir -p

#hipcc
HIPCC=/opt/rocm/bin/hipcc

#BLAS
ROCBLAS_LIB=rocblas

#CONV
MIOPEN_LIB?=MIOpen

#DeepBench
DEEPBENCH_INC=${SOURCE_DIR}/../kernels
CONV_SOURCE=${SOURCE_DIR}/conv_bench_rocm.cpp
CONV_INCLUDES=${SOURCE_DIR}/tensor.h ${DEEPBENCH_INC}/conv_problems.h
GEMM_SOURCE=${SOURCE_DIR}/gemm_bench.cpp
GEMM_INCLUDES=${SOURCE_DIR}/tensor.h ${DEEPBENCH_INC}/gemm_problems.h

all: $(BIN_DIR)/conv_bench $(BIN_DIR)/rnn_bench $(BIN_DIR)/gemm_bench

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-O3

$(BIN_DIR)/conv_bench: ${CONV_SOURCE} ${CONV_INCLUDES}
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${CONV_SOURCE} -o $@ -I$(ROCM_PATH)/include -I$(DEEPBENCH_INC) -l$(CONV_LIBRARY) $(OPT) -std=c++11

$(BIN_DIR)/rnn_bench:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rnn_bench_rocm.cpp -o $@ -I$(ROCM_PATH)/include -I$(DEEPBENCH_INC) -l$(CONV_LIBRARY) $(OPT) -std=c++11

$(BIN_DIR)/gemm_bench: ${GEMM_SOURCE} ${GEMM_INCLUDES}
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${GEMM_SOURCE} -o $@ -I$(ROCM_PATH)/include -I$(DEEPBENCH_INC) -l$(ROCBLAS_LIB) $(OPT) -std=c++11

clean:
	rm -rf $(BIN_DIR)
