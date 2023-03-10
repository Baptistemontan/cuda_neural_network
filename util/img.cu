#include "img.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 10000

#define CHECK_FGETS_ERROR(result, error_label) \
	if(result == NULL) { \
		goto error_label; \
	}

int csv_to_imgs(Img** imgs_array, const char* file_string, size_t number_of_imgs) {
	FILE *fp;
	Img* imgs = (Img*)malloc(number_of_imgs * sizeof(Img));
	char row[MAXCHAR];
	fp = fopen(file_string, "r");
	
	char* fgets_result;
	// Read the first line 
	fgets_result = fgets(row, MAXCHAR, fp);
	size_t i = 0;
	CHECK_FGETS_ERROR(fgets_result, handle_error);
	while (feof(fp) != 1 && i < number_of_imgs) {
		size_t j = 0;
		fgets_result = fgets(row, MAXCHAR, fp);
		CHECK_FGETS_ERROR(fgets_result, handle_error);
		char* token = strtok(row, ",");
		while (token != NULL) {
			if (j == 0) {
				imgs[i].label = atoi(token);
			} else {
				imgs[i].img_data[j - 1] = atoi(token) / 256.0;
			}
			token = strtok(NULL, ",");
			j++;
		}
		i++;
	}
	fclose(fp);
	*imgs_array = imgs;
	return 0;
handle_error:
	return 1;
}

void img_print(Img* img) {
	// matrix_print(img->img_data);
	printf("Img Label: %d\n", img->label);
}

void imgs_free(Img* imgs, size_t n) {
	free(imgs);
}