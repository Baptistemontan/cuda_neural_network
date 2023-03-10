#pragma once

#include <exception>
#include <cuda_runtime.h>
#include <stdio.h>

class CudaException: public std::exception {
public:
    CudaException(cudaError_t cuda_err, int line, const char* file, const char* function) {
        snprintf(this->message, 150, "A cuda error happened with code %d at line %d in file %s at function %s.\n", cuda_err, line, file, function);
    }

    const char* what() const throw() {
        return this->message;
    }

private:
    char message[150];
};

#define CHECK_ERR(stmt) \
    { \
        cudaError_t __err = (stmt); \
        if(__err != cudaSuccess) { \
            throw CudaException(__err, __LINE__, __FILE__, __FUNCTION__); \
        } \
    }

#define CHECK_SYNC CHECK_ERR(cudaDeviceSynchronize())

// struct KernelParams {
//     std::size_t blocks_count;
//     std::size_t threads_count;
// };

// struct KernelParams calculate_kernel_params(std::size_t operations_count) {
//     struct KernelParams params;

//     return params;
// }

class TidRange {
public:
    struct Iterator {
        public:
            __device__ Iterator(std::size_t c): current(c) {}
            
            __device__ const std::size_t& operator*() const { return this->current; }
            
            __device__ std::size_t& operator*() { return this->current; }
            
            __device__ Iterator& operator++() {
                current += gridDim.x * blockDim.x;
                return *this;
            }

            __device__ friend bool operator== (const Iterator& a, const Iterator& b) { 
                return a.current == b.current;
            };

            __device__ friend bool operator!= (const Iterator& a, const Iterator& b) { 
                return !(a == b);
            };

        private:
            std::size_t current;
    };

    __device__ TidRange(std::size_t max): max(max), start(blockDim.x * blockIdx.x + threadIdx.x) { }

    __device__ Iterator begin() {
        return Iterator(this->start);
    }

    __device__ Iterator end() {
        return Iterator(calculate_end(this->max, this->start));
    }

    __device__ static std::size_t calculate_end(std::size_t max, std::size_t start) {
        if(start >= max) {
            return start;
        }
        std::size_t step = gridDim.x * blockDim.x;
        std::size_t rem = (max - start) % step;
        std::size_t offset = max - start - rem;
        return start + offset + (start == 0 ? 0 : step);
    }

private:
    std::size_t max;
    std::size_t start;
};

class TidRange2D {
public:
    struct Iterator {
    public:
        struct Position {
        public:
            std::size_t x;
            std::size_t y;
        };

        __device__ Iterator(std::size_t maxX, std::size_t c): maxX(maxX), current(c) {}

        __device__ Position calculate_pos() const {
            // current = x + maxX * y
            Position pos;
            pos.x = this->current % this->maxX;
            pos.y = (this->current - pos.x) / this->maxX;
            return pos; 
        }
        
        __device__ Position operator*() const {
            return calculate_pos();
        };
        
        __device__ Position operator*() {
            return calculate_pos();
        }
        
        __device__ Iterator& operator++() { 
            current += gridDim.x * blockDim.x;
            return *this;
        }

        __device__ friend bool operator== (const Iterator& a, const Iterator& b) { 
            return a.current == b.current;
        };

        __device__ friend bool operator!= (const Iterator& a, const Iterator& b) { 
            return !(a == b);
        };

    private:
        std::size_t maxX;
        std::size_t current;
    };

    __device__ TidRange2D(std::size_t maxX, std::size_t maxY): maxX(maxX), maxY(maxY), start(blockDim.x * blockIdx.x + threadIdx.x) { }

    __device__ Iterator begin() {
        return Iterator(this->maxX, this->start);
    }

    __device__ Iterator end() {
        return Iterator(this->maxX, TidRange::calculate_end(this->maxX * this->maxY, this->start));
    }

private:
    std::size_t maxX;
    std::size_t maxY;
    std::size_t start;
};

class TidRange3D {
public:
    struct Iterator {
    public:
        struct Position {
        public:
            std::size_t x;
            std::size_t y;
            std::size_t z;
        };

        __device__ Iterator(std::size_t maxX, std::size_t maxY, std::size_t c): maxX(maxX), maxY(maxY), current(c) {}

        __device__ Position calculate_pos() const {
            // current = x + maxX * y + maxX * maxY * z
            Position pos;
            std::size_t rem = this->current % (this->maxX * this->maxY);
            pos.z = (this->current - rem) / (this->maxX * this->maxY);
            pos.x = rem % this->maxX;
            pos.y = (rem - pos.x) / this->maxX;
            return pos; 
        }
        
        __device__ Position operator*() const {
            return calculate_pos();
        };
        
        __device__ Position operator*() {
            return calculate_pos();
        }
        
        __device__ Iterator& operator++() { 
            current += gridDim.x * blockDim.x;
            return *this;
        }

        __device__ friend bool operator== (const Iterator& a, const Iterator& b) { 
            return a.current == b.current;
        };

        __device__ friend bool operator!= (const Iterator& a, const Iterator& b) { 
            return !(a == b);
        };

    private:
        std::size_t maxX;
        std::size_t maxY;
        std::size_t current;
    };

    __device__ TidRange3D(std::size_t maxX, std::size_t maxY, std::size_t maxZ): maxX(maxX), maxY(maxY), maxZ(maxZ), start(blockDim.x * blockIdx.x + threadIdx.x) { }

    __device__ Iterator begin() {
        return Iterator(this->maxX, this->maxY, this->start);
    }

    __device__ Iterator end() {
        return Iterator(this->maxX, this->maxY, TidRange::calculate_end(this->maxX * this->maxY * this->maxZ, this->start));
    }

private:
    std::size_t maxX;
    std::size_t maxY;
    std::size_t maxZ;
    std::size_t start;
};

