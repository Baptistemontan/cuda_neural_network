MAIN_CPU = main_cpu.c
MAIN_GPU = main_gpu.cu

CPP_SOURCES = $(wildcard matrix/*.cpp neural/*.cpp util/*.cpp *.cpp) 
CUDA_SOURCES = $(wildcard matrix/*.cu neural/*.cu util/*.cu *.cu)
HEADERS = $(wildcard matrix/*.hpp neural/*.hpp util/*.hpp *.hpp)
CUDA_HEADERS = $(wildcard matrix/*.cuh neural/*.cuh util/*.cuh *.cuh)
CPP_OBJ = ${CPP_SOURCES:.cpp=.o}
CUDA_OBJ = ${CUDA_SOURCES:.cu=.o}
CFLAGS = 
CUDA_FLAGS = 


EXEC = exec
CC = /usr/bin/g++
CUDAC = /usr/local/cuda/bin/nvcc

default: ${EXEC}
	./${EXEC}

time: ${EXEC}
	time ./${EXEC}

${EXEC}: ${CUDA_OBJ}
	${CUDAC} ${CFLAGS} $^ -o $@ -lm

# ${EXEC_GPU}: ${OBJ} ${CUDA_OBJ} ${MAIN_GPU_OBJ}
# 	${CUDAC} ${CUDA_FLAGS} $^ -o $@ -lm -L/usr/local/cuda-12.0/lib64/stubs -lcuda -L/usr/local/cuda-12.0/lib64 -lcudart -lcudadevrt

# ${CUDA_FINAL_OBJ}: ${CUDA_OBJ} ${MAIN_GPU_OBJ}
# 	${CUDAC} ${CUDA_FLAGS} --device-link $^ --output-file $@

# Generic rules
%.o: %.cpp
	${CC} ${CFLAGS} -c $< -o $@ -lm

# Generic cuda rules
%.o: %.cu
	${CUDAC} ${CUDA_FLAGS} -dc $< -o $@

clean:
	rm ${CPP_OBJ} ${CUDA_OBJ} ${EXEC}
