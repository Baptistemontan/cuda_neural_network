#pragma once

#include <cstddef>
#include <cmath> 
#include <stdexcept>

#include <memory>
#include <string>
#include <stdexcept>

#include <iostream>

#include <functional>
#include <initializer_list>
#include <fstream>
#include <cstring>

#include <device_launch_parameters.h>

#include "../util/cuda_box.cuh"

template <typename T, std::size_t ROWS, std::size_t COLS>
class Matrix;


template<typename ... Args>
std::string string_format(const std::string& format, Args... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if(size_s <= 0) { 
        throw std::runtime_error("Error during formatting."); 
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template <typename T, std::size_t SIZE>
class Vector {
public:

    Vector() = default;

    __device__ __host__ Vector(const T &e) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] = e;
        }
    }

    Vector(const Vector<T, SIZE>& other) {
        std::memcpy((void*)this->data, (const void*)other.data, sizeof(T) * SIZE);
    }

    __device__ __host__ Vector(std::initializer_list<T> il) {
        #ifndef __CUDA_ARCH__
        if(il.size() != SIZE) {
            throw std::invalid_argument(string_format("Tried to initialize a vector of size %lu with %lu elements", SIZE, il.size()));
        }
        #endif
        auto it = il.begin();
        for(std::size_t i = 0; i < SIZE; i++, it++) {
            this->data[i] = *it;
        }
    }

    Vector(std::ifstream& in) {
        std::size_t size;
        in.read((char*)&size, sizeof(std::size_t));
        std::cout << "Vector init of size " << size << std::endl;
        if(size != SIZE) {
            throw std::invalid_argument(string_format("Tried to initialize a vector of size %lu from a binary file with %lu elements", SIZE, size));
        }
        in.read((char*)this->data, sizeof(T) * SIZE);
    }

    __device__ __host__ std::size_t size() const {
        return SIZE;
    }

    __device__ __host__ Vector<T, SIZE>& operator=(Vector<T, SIZE> rhs) {
        std::memcpy((void*)this->data, (void*)rhs.data, sizeof(T) * SIZE);
        return *this;
    }

    // __device__ Vector<T, SIZE>& operator=(Vector<T, SIZE> rhs) {
    //     memcpy((void*)this->data, (void*)rhs.data, sizeof(T) * SIZE);
    //     return *this
    // }

    Vector<T, SIZE> apply(std::function<T(const T&)>& func) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out[i] = func(this->data[i]);
        }
        return out;
    }

    template<std::size_t RHS_SIZE>
    Matrix<T, SIZE, RHS_SIZE> dot(const Vector<T, RHS_SIZE>& rhs) const {
        Matrix<T, SIZE, RHS_SIZE> out;
        for(std::size_t row = 0; row < SIZE; row++) {
            for(std::size_t col = 0; col < RHS_SIZE; col++) {
                out[row][col] = this->data[row] * rhs[col];
            }
        }
        return out;
    }

    Vector<T, SIZE> softmax() const {
        T total = T();
        for (std::size_t i = 0; i < SIZE; i++) {
            total += std::exp(this->data[i]);
        }
        Vector<T, SIZE> out;
		for (std::size_t i = 0; i < SIZE; i++) {
            out[i] = std::exp(this->data[i]) / total;
        }
        return out;
    }

    std::size_t argmax() const {
        if(SIZE == 0) {
            throw std::runtime_error("The argmax method is not possible on a vector of size 0.");
        }
        std::size_t max_index = 0;
        T max_elem = this->data[0];
        for(std::size_t i = 1; i < SIZE; i++) {
            if(this->data[i] > max_elem) {
                max_elem = this->data[i];
                max_index = i;
            }
        }
        return max_index;
    }

    std::ofstream& save_binary(std::ofstream& out) const {
        std::size_t size = SIZE;
        out.write((const char*)&size, sizeof(std::size_t));
        out.write((const char*)this->data, sizeof(T) * SIZE);
		return out;
	}

    __device__ __host__ T& get(std::size_t i) {
        #ifndef __CUDA_ARCH__
        if(i >= SIZE) {
			throw std::out_of_range(string_format("Tried to access index %lu but the vector has %lu elements.", i, SIZE));
		}
        #endif
        return this->data[i];
    }

    __device__ __host__ T& operator[](std::size_t i) {
        return get(i);
    }

    __device__ __host__ const T& get(std::size_t i) const {
        #ifndef __CUDA_ARCH__
        if(i >= SIZE) {
			throw std::out_of_range(string_format("Tried to access index %lu but the vector has %lu elements.", i, SIZE));
		}
        #endif
        return this->data[i];
    }

    __device__ __host__ const T& operator[](std::size_t i) const {
        return get(i);
    }

    Vector<T, SIZE> operator+(const Vector<T, SIZE>& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] + rhs[i];
        }
        return out;
    }

    Vector<T, SIZE>& operator+=(const Vector<T, SIZE>& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] += rhs[i];
        }
        return *this;
    }

    Vector<T, SIZE> operator+(const T& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] + rhs;
        }
        return out;
    }

    Vector<T, SIZE>& operator+=(const T& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] += rhs;
        }
        return *this;
    }

    Vector<T, SIZE> operator-(const Vector<T, SIZE>& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] - rhs[i];
        }
        return out;
    }

    Vector<T, SIZE>& operator-=(const Vector<T, SIZE>& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] -= rhs[i];
        }
        return *this;
    }

    Vector<T, SIZE> operator-(const T& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] - rhs;
        }
        return out;
    }

    Vector<T, SIZE>& operator-=(const T& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] -= rhs;
        }
        return *this;
    }

    Vector<T, SIZE> operator*(const Vector<T, SIZE>& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] * rhs[i];
        }
        return out;
    }

    Vector<T, SIZE>& operator*=(const Vector<T, SIZE>& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] *= rhs[i];
        }
        return *this;
    }

    Vector<T, SIZE> operator*(const T& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] * rhs;
        }
        return out;
    }

    Vector<T, SIZE>& operator*=(const T& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] *= rhs;
        }
        return *this;
    }

    Vector<T, SIZE> operator/(const Vector<T, SIZE>& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] / rhs[i];
        }
        return out;
    }

    Vector<T, SIZE>& operator/=(const Vector<T, SIZE>& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] /= rhs[i];
        }
        return *this;
    }

    Vector<T, SIZE> operator/(const T& rhs) const {
        Vector<T, SIZE> out;
        for(std::size_t i = 0; i < SIZE; i++) {
            out.data[i] = this->data[i] / rhs;
        }
        return out;
    }

    Vector<T, SIZE>& operator/=(const T& rhs) {
        for(std::size_t i = 0; i < SIZE; i++) {
            this->data[i] /= rhs;
        }
        return *this;
    }


    friend std::ostream &operator<<(std::ostream &output, const Vector<T, SIZE> &vec) { 
        output << "[";
        for(std::size_t i = 0; i < SIZE - 1; i++) {
            output << vec[i] << ", ";
        }
        if(SIZE != 0) {
            output << vec[SIZE - 1];
        }
        output << "]";
        return output;
    }

private:
    T data[SIZE];
};


namespace VectorCuda {
    template<typename T, std::size_t SIZE>
    __global__ void add(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) + rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void add(Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) += rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void add(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) + *rhs;
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void add(Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) += *rhs;
        }
    }
    

    template<typename T, std::size_t SIZE>
    __global__ void sub(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) - rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void sub(Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) -= rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void sub(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) - *rhs;
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void sub(Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) -= *rhs;
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void mul(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) * rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void mul(Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) *= rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void mul(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) * *rhs;
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void mul(Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) *= *rhs;
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void div(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) / rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void div(Vector<T, SIZE>* lhs, const Vector<T, SIZE>* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) /= rhs->get(tid);
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void div(Vector<T, SIZE>* out, const Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            out->get(tid) = lhs->get(tid) / *rhs;
        }
    }

    template<typename T, std::size_t SIZE>
    __global__ void div(Vector<T, SIZE>* lhs, const T* rhs) {
        for(std::size_t tid: TidRange(SIZE)) {
            lhs->get(tid) /= *rhs;
        }
    }

    template<typename T, std::size_t LHS_SIZE, std::size_t RHS_SIZE>
    __global__ void dot(Matrix<T, LHS_SIZE, RHS_SIZE>* out, const Vector<T, LHS_SIZE>* lhs, const Vector<T, RHS_SIZE>* rhs) {
        for(auto tid: TidRange2D(LHS_SIZE, RHS_SIZE)) {
            printf("position: (%lu, %lu)\n", tid.x, tid.y);
            out->get(tid.x)[tid.y] = lhs->get(tid.x) * rhs->get(tid.y);
        }
    }

    // BATCH IMPLEMENTATIONS

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] + b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            a[tid.x] += b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] + b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            a[tid.x] += b;
        }
    }
    
    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] - b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            a[tid.x] -= b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] - b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            a[tid.x] -= b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] * b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            a[tid.x] *= b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] * b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            a[tid.x] *= b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] / b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const Vector<T, SIZE>& b = (*rhs)[tid.y];
            a[tid.x] /= b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] / b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            const T& b = (*rhs)[tid.y];
            a[tid.x] /= b;
        }
    }

    template<typename T, std::size_t N_VEC, std::size_t LHS_SIZE, std::size_t RHS_SIZE>
    __global__ void dot(
        Matrix<T, LHS_SIZE, RHS_SIZE> (*out)[N_VEC], 
        const Vector<T, LHS_SIZE> (*lhs)[N_VEC], 
        const Vector<T, RHS_SIZE> (*rhs)[N_VEC]
    ) {
        for(auto tid: TidRange3D(LHS_SIZE, RHS_SIZE, N_VEC)) {
            const Vector<T, LHS_SIZE>& a = (*lhs)[tid.z];
            const Vector<T, RHS_SIZE>& b = (*rhs)[tid.z];
            Matrix<T, LHS_SIZE, RHS_SIZE>& c = (*out)[tid.z]; 
            c[tid.x][tid.y] = a[tid.x] * b[tid.y];
        }
    }

    // BATCH WITH SINGLE

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] + b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] += b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T *rhs
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] + *rhs;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void add(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T *rhs
    ) {
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] += *rhs;
        }
    }
    
    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] - b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] -= b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T *rhs
    ) {
        const T& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] - b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void sub(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T *rhs
    ) {
        const T& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] -= b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] * b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] *= b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T *rhs
    ) {
        const T& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] * b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void mul(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T *rhs
    ) {
        const T& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] *= b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] / b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const Vector<T, SIZE> *rhs
    ) {
        const Vector<T, SIZE>& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] /= b[tid.x];
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*out)[N_VEC], 
        const Vector<T, SIZE> (*lhs)[N_VEC],
        const T *rhs
    ) {
        const T& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            const Vector<T, SIZE>& a = (*lhs)[tid.y];
            Vector<T, SIZE>& c = (*out)[tid.y];
            c[tid.x] = a[tid.x] / b;
        }
    }

    template<typename T, std::size_t SIZE, std::size_t N_VEC>
    __global__ void div(
        Vector<T, SIZE> (*lhs)[N_VEC], 
        const T *rhs
    ) {
        const T& b = *rhs;
        for(auto tid: TidRange2D(SIZE, N_VEC)) {
            Vector<T, SIZE>& a = (*lhs)[tid.y];
            a[tid.x] /= b;
        }
    }

    template<typename T, std::size_t N_VEC, std::size_t LHS_SIZE, std::size_t RHS_SIZE>
    __global__ void dot(
        Matrix<T, LHS_SIZE, RHS_SIZE> (*out)[N_VEC], 
        const Vector<T, LHS_SIZE> (*lhs)[N_VEC], 
        const Vector<T, RHS_SIZE> *rhs
    ) {
        const Vector<T, RHS_SIZE>& b = *rhs;
        for(auto tid: TidRange3D(LHS_SIZE, RHS_SIZE, N_VEC)) {
            const Vector<T, LHS_SIZE>& a = (*lhs)[tid.z];
            Matrix<T, LHS_SIZE, RHS_SIZE>& c = (*out)[tid.z]; 
            c[tid.x][tid.y] = a[tid.x] * b[tid.y];
        }
    }
}