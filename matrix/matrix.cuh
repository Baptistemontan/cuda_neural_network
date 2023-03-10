#pragma once

#include "vector.cuh"

template <typename T, std::size_t ROWS, std::size_t COLS>
class Matrix {
public:

	Matrix() = default;

	__device__ __host__ Matrix(const T& e) {
		for(std::size_t i = 0; i < ROWS; i++) {
			this->data[i] = Vector<T, COLS>(e);
		}
	}

	__device__ __host__ Matrix(const Matrix<T, ROWS, COLS>& other) {
		std::memcpy((void*)this->data, (const void*)other.data, sizeof(T) * ROWS * COLS);
	}

	__host__ Matrix(std::initializer_list<std::initializer_list<T>> il) {
		if(il.size() != ROWS) {
            throw std::invalid_argument(string_format("Tried to initialize a matrix with %lu rows but %lu rows where supplied", ROWS, il.size()));
        }
        auto it = il.begin();
        for(std::size_t i = 0; i < ROWS; i++, it++) {
            this->data[i] = Vector<T, COLS>(*it);
        }
	}

	Matrix(std::ifstream& in) {
		std::size_t rows, cols;
		in.read((char*)&rows, sizeof(std::size_t));
		in.read((char*)&cols, sizeof(std::size_t));
		std::cout << "Matrix init with " << rows << " rows and " << cols << " columns" << std::endl;
		if(rows != ROWS || cols != COLS) {
            throw std::invalid_argument(string_format("Tried to initialize a %lux%lu Matrix but binary file contain %lux%lu Matrix", ROWS, COLS, rows, cols));
        }
		in.read((char*)&this->data, sizeof(T) * ROWS * COLS);
	}

	Matrix<T, ROWS, COLS>& operator=(Matrix<T, ROWS, COLS> rhs) {
		std::memcpy((void*)this->data, (const void*)rhs.data, sizeof(T) * ROWS * COLS);
		return *this;
	}

	Matrix<T, ROWS, COLS> apply(std::function<T(const T&)>& func) const {
		Matrix<T, ROWS, COLS> out;
		for(std::size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row].apply(func);
		}
		return out;
	}

	__device__ __host__ Vector<T, COLS>& get(size_t i) {
		#ifndef __CUDA_ARCH__
		if(i >= ROWS) {
			throw std::out_of_range(string_format("Tried to access row %lu but the matrix has %lu rows.", i, ROWS));
		}
		#endif
		return this->data[i];
	}

	__device__ __host__ Vector<T, COLS>& operator[](size_t i) {
		return get(i);
	}

	__device__ __host__ const Vector<T, COLS>& get(size_t i) const {
		#ifndef __CUDA_ARCH__
		if(i >= ROWS) {
			throw std::out_of_range(string_format("Tried to access row %lu but the matrix has %lu rows.", i, ROWS));
		}
		#endif
		return this->data[i];
	}

	__device__ __host__ const Vector<T, COLS>& operator[](size_t i) const {
		return get(i);
	}

	Matrix<T, ROWS, COLS> operator+(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] + rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator+=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] += rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator+(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] + rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator+=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] += rhs;
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator-(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] - rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator-=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] -= rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator-(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] - rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator-=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] -= rhs;
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator*(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] * rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator*=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] *= rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator*(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] * rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator*=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] *= rhs;
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator/(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] / rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator/=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] /= rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator/(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] / rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator/=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] /= rhs;
		}
		return *this;
	}

	template<size_t RHS_COLS>
	Matrix<T, ROWS, RHS_COLS> dot(const Matrix<T, COLS, RHS_COLS>& rhs) const {
		Matrix<T, ROWS, RHS_COLS> out;
		for (size_t i = 0; i < ROWS; i++) {
			for (size_t j = 0; j < RHS_COLS; j++) {
				T sum = T();
				for (size_t k = 0; k < COLS; k++) {
					sum += this->data[i][k] * rhs[k][j];
				}
				out[i][j] = sum;
			}
		}
		return out;
	}

	Vector<T, ROWS> dot(const Vector<T, COLS>& rhs) const {
		Vector<T, ROWS> out;
		for(size_t row = 0; row < ROWS; row++) {
			T sum = T();
			for(size_t col = 0; col < COLS; col++) {
				sum += this->data[row][col] * rhs[col];
			}
			out[row] = sum;
		}
		return out;
	}

	Matrix<T, ROWS * COLS, 1> flatten_vertical() const {
		Matrix<T, ROWS * COLS, 1> out;
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				out[row * ROWS + col][0] = this->data[row][col];
			}
		}
		return out;
	}

	Matrix<T, 1, ROWS * COLS> flatten_horizontal() const {
		Matrix<T, 1, ROWS * COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				out[0][row * ROWS + col] = this->data[row][col];
			}
		}
		return out;
	}

	Matrix<T, COLS, ROWS> transpose() const {
		Matrix<T, COLS, ROWS> out;
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				out[col][row] = this->data[row][col];
			}
		}
		return out;
	}

	std::ofstream& save_binary(std::ofstream& out) const {
		std::size_t rows = ROWS, cols = COLS;
        out.write((const char*)&rows, sizeof(std::size_t));
        out.write((const char*)&cols, sizeof(std::size_t));
		out.write((const char*)&this->data, sizeof(T) * ROWS * COLS);
		return out;
	}

	friend std::ostream &operator<<(std::ostream &output, const Matrix<T, ROWS, COLS> &mat) { 
        output << "[\n";
        for(std::size_t i = 0; i < ROWS - 1; i++) {
            output << "\t" << mat[i] << ",\n";
        }
        if(ROWS != 0) {
            output << "\t" << mat[ROWS - 1];
        }
        output << "\n]";
        return output;
    }

	void randomize(T n) {
		// Pulling from a random distribution of 
		// Min: -1 / sqrt(n)
		// Max: 1 / sqrt(n)
		T min = -1.0 / sqrt(n);
		T max = 1.0 / sqrt(n);
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				this->data[row][col] = uniform_distribution(min, max);
			}
		}
	}

private:
	static T uniform_distribution(T low, T high) {
		T difference = high - low; // The difference between the two
		int scale = 10000;
		int scaled_difference = (int)(difference * scale);
		return (T)(low + (1.0 * (rand() % scaled_difference) / scale));
	}

	Vector<T, COLS> data[ROWS];
};

namespace MatrixCuda {

	template<typename T, std::size_t ROWS, std::size_t COLS>
	__global__ void transpose(Matrix<T, COLS, ROWS>* out, const Matrix<T, ROWS, COLS>* rhs) {
		for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			out->get(col)[row] = rhs->get(row)[col];
		}
	}

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void muladd(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] += lhs->get(row)[col] * *rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void add(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] + rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void add(Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] += rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void add(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] + rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void add(Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] += rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void sub(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] - rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void sub(Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] -= rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void sub(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] - rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void sub(Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] -= rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void mul(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] * rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void mul(Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] *= rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void mul(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] * rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void mul(Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] *= rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void div(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] / rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void div(Matrix<T, ROWS, COLS>* lhs, const Matrix<T, ROWS, COLS>* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] /= rhs->get(row)[col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void div(Matrix<T, ROWS, COLS>* out, const Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            out->get(row)[col] = lhs->get(row)[col] / rhs;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS>
    __global__ void div(Matrix<T, ROWS, COLS>* lhs, const T* rhs) {
        for(auto tid: TidRange2D(ROWS, COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
            lhs->get(row)[col] /= rhs;
        }
    }

	template<typename T, std::size_t LHS_ROWS, std::size_t LHS_COLS, std::size_t RHS_COLS>
	__global__ void dot(
		Matrix<T, LHS_ROWS, RHS_COLS>* out, 
		const Matrix<T, LHS_ROWS, LHS_COLS>* lhs,
		const Matrix<T, LHS_COLS, RHS_COLS>* rhs
	) {
		for(auto tid: TidRange2D(LHS_ROWS, RHS_COLS)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			T sum = T();
			for(std::size_t k = 0; k < LHS_COLS; k++) {
				sum += lhs->get(row)[k] * rhs->get(k)[col];
			}
			out->get(row)[col] = sum;
		}
	}

	template<typename T, std::size_t ROWS, std::size_t COLS>
	__global__ void dot(
		Vector<T, ROWS>* out,
		const Matrix<T, ROWS, COLS>* lhs,
		const Vector<T, COLS>* rhs
	) {
		for(std::size_t row: TidRange(ROWS)) {
			T sum = T();
			for(std::size_t col = 0; col < COLS; col++) {
				sum += lhs->get(row)[col] * rhs->get(col);
			}
			out->get(row) = sum;
		}
	}

	// BATCH IMPLEMENTATIONS

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] + b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
            a[row][col] += b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] + b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
            a[row][col] += b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] - b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
            a[row][col] -= b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] - b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
            a[row][col] -= b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] * b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
            a[row][col] *= b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] * b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
            a[row][col] *= b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] / b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
            a[row][col] /= b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] / b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T (*rhs)[N_MAT]
	) {
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const T& b = (*rhs)[mat_i];
            a[row][col] /= b;
        }
    }

	template<typename T, std::size_t LHS_ROWS, std::size_t LHS_COLS, std::size_t RHS_COLS, std::size_t N_MAT>
	__global__ void dot(
		Matrix<T, LHS_ROWS, RHS_COLS> (*out)[N_MAT], 
		const Matrix<T, LHS_ROWS, LHS_COLS> (*lhs)[N_MAT],
		const Matrix<T, LHS_COLS, RHS_COLS> (*rhs)[N_MAT]
	) {
		for(auto tid: TidRange3D(LHS_ROWS, RHS_COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, LHS_ROWS, LHS_COLS>& a = (*lhs)[mat_i];
			const Matrix<T, LHS_COLS, RHS_COLS>& b = (*rhs)[mat_i];
			Matrix<T, LHS_ROWS, RHS_COLS>& c = (*out)[mat_i];
			T sum = T();
			for(std::size_t k = 0; k < LHS_COLS; k++) {
				sum += a[row][k] * b[k][col];
			}
			c[row][col] = sum;
		}
	}

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
	__global__ void dot(
		Vector<T, ROWS> (*out)[N_MAT],
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT],
		const Vector<T, COLS> (*rhs)[N_MAT]
	) {
		for(auto tid: TidRange2D(ROWS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t mat_i = tid.y;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			const Vector<T, COLS>& b = (*rhs)[mat_i];
			Vector<T, ROWS>& c = (*out)[mat_i];
			T sum = T();
			for(size_t col = 0; col < COLS; col++) {
				sum += a[row][col] * b[col];
			}
			c[row] = sum;
		}
	}

	// BATCH WITH SINGLE

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] + b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] += b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] + b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void add(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] += b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] - b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] -= b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] - b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void sub(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] -= b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] * b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] *= b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] * b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void mul(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] *= b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] / b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const Matrix<T, ROWS, COLS> *rhs
	) {
		const Matrix<T, ROWS, COLS>& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] /= b[row][col];
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*out)[N_MAT], 
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Matrix<T, ROWS, COLS>& c = (*out)[mat_i];
            out[row][col] = a[row][col] / b;
        }
    }

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void div(
		Matrix<T, ROWS, COLS> (*lhs)[N_MAT], 
		const T *rhs
	) {
		const T& b = *rhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
            a[row][col] /= b;
        }
    }

	template<typename T, std::size_t LHS_ROWS, std::size_t LHS_COLS, std::size_t RHS_COLS, std::size_t N_MAT>
	__global__ void dot(
		Matrix<T, LHS_ROWS, RHS_COLS> (*out)[N_MAT], 
		const Matrix<T, LHS_ROWS, LHS_COLS> (*lhs)[N_MAT],
		const Matrix<T, LHS_COLS, RHS_COLS> *rhs
	) {
		const Matrix<T, LHS_COLS, RHS_COLS>& b = *rhs;
		for(auto tid: TidRange3D(LHS_ROWS, RHS_COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, LHS_ROWS, LHS_COLS>& a = (*lhs)[mat_i];
			Matrix<T, LHS_ROWS, RHS_COLS>& c = (*out)[mat_i];
			T sum = T();
			for(std::size_t k = 0; k < LHS_COLS; k++) {
				sum += a[row][k] * b[k][col];
			}
			c[row][col] = sum;
		}
	}

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
	__global__ void dot(
		Vector<T, ROWS> (*out)[N_MAT],
		const Matrix<T, ROWS, COLS> (*lhs)[N_MAT],
		const Vector<T, COLS> *rhs
	) {
		const Vector<T, COLS>& b = *rhs;
		for(auto tid: TidRange2D(ROWS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t mat_i = tid.y;
			const Matrix<T, ROWS, COLS>& a = (*lhs)[mat_i];
			Vector<T, ROWS>& c = (*out)[mat_i];
			T sum = T();
			for(size_t col = 0; col < COLS; col++) {
				sum += a[row][col] * b[col];
			}
			c[row] = sum;
		}
	}

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
	__global__ void dot(
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
			c[row] = sum;
		}
	}

	template<typename T, std::size_t ROWS, std::size_t COLS, std::size_t N_MAT>
    __global__ void reduce(
		Matrix<T, ROWS, COLS> *lhs,
		const Matrix<T, ROWS, COLS> (*rhs)[N_MAT]
	) {
		Matrix<T, ROWS, COLS>& a = *lhs;
        for(auto tid: TidRange3D(ROWS, COLS, N_MAT)) {
			std::size_t row = tid.x;
			std::size_t col = tid.y;
			std::size_t mat_i = tid.z;
			const Matrix<T, ROWS, COLS>& b = (*rhs)[mat_i];
            a[row][col] += b[row][col];
        }
    }
}
