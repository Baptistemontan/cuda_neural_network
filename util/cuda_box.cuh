#pragma once

#include "./cuda_handling.cuh"

template<typename T>
class CudaBox {
public:

    template<class... Args>
    CudaBox(Args... args) {
        T element(args...);
        CHECK_ERR(cudaMalloc((void**)&this->data, sizeof(T)));
        CHECK_ERR(cudaMemcpy((void*)this->data, (void*)&element, sizeof(T), cudaMemcpyHostToDevice));
    }

    ~CudaBox() {
        cudaFree((void*)this->data); // could return an error, but can't throw in destructor, so whatever
    }

    T to_host() const {
        T element;
        CHECK_ERR(cudaMemcpy((void*)&element, (const void*)this->data, sizeof(T), cudaMemcpyDeviceToHost));
        return element;
    }

    T replace(const T* new_element) {
        T old_element = to_host();
        put(new_element);
        return old_element;
    }

    void put(const T* new_element) {
        CHECK_ERR(cudaMemcpy((void*)this->data, (const void*)new_element, sizeof(T), cudaMemcpyHostToDevice));
    }

    T* get_ptr() {
        return this->data;
    }

    const T* get_ptr() const {
        return this->data;
    }

    T* operator&() {
        return this->data;
    }

    const T* operator&() const {
        return this->data;
    }



private:
    T* data;
};

template<typename T, std::size_t SIZE>
class CudaArray {
public:

    typedef T (array)[SIZE];

    CudaArray() {
        CHECK_ERR(cudaMalloc((void**)&this->data, sizeof(array)));
    }

    CudaArray(array arr) {
        CHECK_ERR(cudaMalloc((void**)&this->data, sizeof(array)));
        CHECK_ERR(cudaMemcpy((void*)this->data, (const void*)arr, sizeof(array), cudaMemcpyHostToDevice));
    }

    ~CudaArray() {
        cudaFree((void*)this->data); // could return an error, but can't throw in destructor, so whatever
    }

    void to_host(array buff) const {
        CHECK_ERR(cudaMemcpy((void*)buff, (const void*)this->data, sizeof(array), cudaMemcpyDeviceToHost));
    }

    void put(array new_elements) {
        CHECK_ERR(cudaMemcpy((void*)this->data, (const void*)new_elements, sizeof(array), cudaMemcpyHostToDevice));
    }

    void set_all(T element) {
        T new_elements[SIZE];
        for(std::size_t i = 0; i < SIZE; i++) {
            new_elements[i] = element;
        }
        put(new_elements);
    }

    void replace(array buff, array new_elements) {
        to_host(buff);
        put(new_elements);
    }


    array* get_ptr() {
        return this->data;
    }

    const array* get_ptr() const {
        return this->data;
    }

    array* operator&() {
        return this->data;
    }

    const array* operator&() const {
        return this->data;
    }

private:
    array* data;
};