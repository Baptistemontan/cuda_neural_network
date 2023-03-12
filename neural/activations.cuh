#pragma once

#include <cmath>
#include "../matrix/vector.cuh"

template<typename T>
__device__ __host__ T sigmoid(const T& input) {
    return 1.0 / (1 + std::exp(-1 * input));
}

template<typename T>
__device__ __host__ T sigmoid_prime(const T& x) {
    return (1 - x) * x;
}

template<typename T>
__device__ __host__ T relu(const T& x) {
    if(x >= 0.0) {
		return x;
	}
	return 0;
}

template<typename T>
__device__ __host__ T relu_prime(const T& x) {
    if(x > 0.0) {
		return 1;
	} else {
		return 0;
	}
}

namespace ActivationCuda {
	typedef enum {
		Relu,
		Sigmoid
	} ActivationFunction;

	template<typename T, std::size_t SIZE>
    __global__ void apply_relu(Vector<T, SIZE>* out, const Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = relu(vec->get(tid));
        }
    }

	template<typename T, std::size_t SIZE>
    __global__ void apply_relu(Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            vec->get(tid) = relu(vec->get(tid));
        }
    }

	template<typename T, std::size_t SIZE>
    __global__ void apply_relu_prime(Vector<T, SIZE>* out, const Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = relu_prime(vec->get(tid));
        }
    }

	template<typename T, std::size_t SIZE>
    __global__ void apply_relu_prime(Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            vec->get(tid) = relu_prime(vec->get(tid));
        }
    }

	template<typename T, std::size_t SIZE>
    __global__ void apply_sigmoid(Vector<T, SIZE>* out, const Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = sigmoid(vec->get(tid));
        }
    }

	template<typename T, std::size_t SIZE>
    __global__ void apply_sigmoid(Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            vec->get(tid) = sigmoid(vec->get(tid));
        }
    }

	template<typename T, std::size_t SIZE>
    __global__ void apply_sigmoid_prime(Vector<T, SIZE>* out, const Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = sigmoid_prime(vec->get(tid));
        }
    }

	template<typename T, std::size_t SIZE>
    __global__ void apply_sigmoid_prime(Vector<T, SIZE>* vec) {
        for(std::size_t tid: TidRange(SIZE)) {
            vec->get(tid) = sigmoid_prime(vec->get(tid));
        }
    }

	// BATCH IMPLEMENTATION

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_relu(Vector<T, SIZE> (*out)[N_VEC], const Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			const Vector<T, SIZE>& a = (*vec)[vec_i];
			Vector<T, SIZE>& b = (*out)[vec_i];
            b[i] = relu(a[i]);
        }
    }

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_relu(Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			Vector<T, SIZE>& v = (*vec)[vec_i];
            v[i] = relu(v[i]);
        }
    }

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_relu_prime(Vector<T, SIZE> (*out)[N_VEC], const Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			const Vector<T, SIZE>& a = (*vec)[vec_i];
			Vector<T, SIZE>& b = (*out)[vec_i];
            b[i] = relu_prime(a[i]);
        }
    }

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_relu_prime(Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			Vector<T, SIZE>& v = (*vec)[vec_i];
            v[i] = relu_prime(v[i]);
        }
    }

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_sigmoid(Vector<T, SIZE> (*out)[N_VEC], const Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			const Vector<T, SIZE>& a = (*vec)[vec_i];
			Vector<T, SIZE>& b = (*out)[vec_i];
            b[i] = sigmoid(a[i]);
        }
    }

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_sigmoid(Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			Vector<T, SIZE>& v = (*vec)[vec_i];
            v[i] = sigmoid(v[i]);
        }
    }

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_sigmoid_prime(Vector<T, SIZE> (*out)[N_VEC], const Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			const Vector<T, SIZE>& a = (*vec)[vec_i];
			Vector<T, SIZE>& b = (*out)[vec_i];
            b[i] = sigmoid_prime(a[i]);
        }
    }

	template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_sigmoid_prime(Vector<T, SIZE> (*vec)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			Vector<T, SIZE>& v = (*vec)[vec_i];
            v[i] = sigmoid_prime(v[i]);
        }
    }

    // SPECIFIC IMPLEMENTATION

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_relu_prime_mul(Vector<T, SIZE> (*out)[N_VEC], const Vector<T, SIZE> (*vec)[N_VEC], const Vector<T, SIZE> (*rhs)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			const Vector<T, SIZE>& a = (*vec)[vec_i];
			const Vector<T, SIZE>& b = (*rhs)[vec_i];
			Vector<T, SIZE>& c = (*out)[vec_i];
            c[i] = relu_prime(a[i]) * b[i];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void apply_sigmoid_prime_mul(Vector<T, SIZE> (*out)[N_VEC], const Vector<T, SIZE> (*vec)[N_VEC], const Vector<T, SIZE> (*rhs)[N_VEC]) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
			std::size_t i = tid.x;
			std::size_t vec_i = tid.y;
			const Vector<T, SIZE>& a = (*vec)[vec_i];
			const Vector<T, SIZE>& b = (*rhs)[vec_i];
			Vector<T, SIZE>& c = (*out)[vec_i];
            c[i] = sigmoid_prime(a[i]) * b[i];
        }
    }

    template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
	__global__ void dot_relu(
		Vector<T, ROWS> (*out)[N_MAT],
		const Matrix<T, ROWS, COLS> *lhs,
		const Vector<T, COLS> (*rhs)[N_MAT]
	) {
		const Matrix<T, ROWS, COLS>& a = *lhs;
		for(auto tid: TidRange2D(ROWS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t mat_i = tid.y;
			const Vector<T, COLS>& b = (*rhs)[mat_i];
			Vector<T, ROWS>& c = (*out)[mat_i];
			T sum = T();
			for(size_t col = 0; col < COLS; col++) {
				sum += a[row][col] * b[col];
			}
			c[row] = relu(sum);
		}
	}

    template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
	__global__ void dot_sigmoid(
		Vector<T, ROWS> (*out)[N_MAT],
		const Matrix<T, ROWS, COLS> *lhs,
		const Vector<T, COLS> (*rhs)[N_MAT]
	) {
		const Matrix<T, ROWS, COLS>& a = *lhs;
		for(auto tid: TidRange2D(ROWS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t mat_i = tid.y;
			const Vector<T, COLS>& b = (*rhs)[mat_i];
			Vector<T, ROWS>& c = (*out)[mat_i];
			T sum = T();
			for(size_t col = 0; col < COLS; col++) {
				sum += a[row][col] * b[col];
			}
			c[row] = sigmoid(sum);
		}
	}
    
}