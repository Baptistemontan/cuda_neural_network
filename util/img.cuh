#pragma once

#include "../matrix/vector.cuh"

#define IMAGE_SIZE (28 * 28)

typedef struct {
	Vector<float, IMAGE_SIZE> img_data;
	int label;
} Img;

int csv_to_imgs(Img** imgs_array, const char* file_string, size_t number_of_imgs);
void img_print(Img* img);
void imgs_free(Img *imgs, size_t n);
