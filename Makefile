MAIN_CPU = main_cpu.c
MAIN_GPU = main_gpu.cu

CPP_SOURCES = $(wildcard matrix/*.cpp neural/*.cpp util/*.cpp *.cpp) 
CUDA_SOURCES = $(wildcard matrix/*.cu neural/*.cu util/*.cu *.cu)
HEADERS = $(wildcard matrix/*.hpp neural/*.hpp util/*.hpp *.hpp)
CUDA_HEADERS = $(wildcard matrix/*.cuh neural/*.cuh util/*.cuh *.cuh)
CPP_OBJ = ${CPP_SOURCES:.cpp=.o}
CUDA_OBJ = ${CUDA_SOURCES:.cu=.o}
CFLAGS = -lm -O3
CUDA_FLAGS = -O3


EXEC = exec
CC = /usr/bin/g++
CUDAC = /usr/local/cuda/bin/nvcc

default: ${EXEC}
	./${EXEC}

time: ${EXEC}
	time ./${EXEC}

${EXEC}: ${CUDA_OBJ} ${CUDA_HEADERS}
	${CUDAC} ${CUDA_OBJ} -o $@ ${CFLAGS}

# ${EXEC_GPU}: ${OBJ} ${CUDA_OBJ} ${MAIN_GPU_OBJ}
# 	${CUDAC} ${CUDA_FLAGS} $^ -o $@ -lm -L/usr/local/cuda-12.0/lib64/stubs -lcuda -L/usr/local/cuda-12.0/lib64 -lcudart -lcudadevrt

# ${CUDA_FINAL_OBJ}: ${CUDA_OBJ} ${MAIN_GPU_OBJ}
# 	${CUDAC} ${CUDA_FLAGS} --device-link $^ --output-file $@

# Generic rules
%.o: %.cpp
	${CC} -c $< -o $@ ${CFLAGS}

# Generic cuda rules
%.o: %.cu
	${CUDAC} -dc $< -o $@ ${CUDA_FLAGS}

clean:
	rm ${CPP_OBJ} ${CUDA_OBJ} ${EXEC}
