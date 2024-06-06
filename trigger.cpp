#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include <vector>
#include <iostream>
#include <chrono>

int splittlisi(float* input_data_1, float* input_data_2, float* input_snap, float* result_array, size_t size, size_t unit_size, size_t ima, float maxall);

using namespace std;

float* read_fits_image(const char* filename, long* naxes) {
    fitsfile *fptr;
    int status = 0;

    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(stderr, status);
        return NULL;
    }

    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    float *image_data = (float *)malloc(naxes[0] * naxes[1] * sizeof(float));
    if (image_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    fits_read_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], NULL, image_data, NULL, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        fits_close_file(fptr, &status);
        return NULL;
    }

    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        return NULL;
    }

    return image_data;
}

int write_fits_image(const char* filename, float *image_data, long* naxes) {
    fitsfile *fptr;
    int status = 0;

    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    long naxis = 2;
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    fits_write_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], image_data, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    char input_data_file_1[1000];
    char input_data_file_2[1000];
    char input_snap_file[1000];
    size_t unit_size = 32;
    size_t ima = 4096;
    size_t imapow = ima*ima;
    size_t unit_num = ima/unit_size;
    long imasize[2];
    float maxall = 1.0203;
    
    sprintf(input_data_file_1, "%s", argv[1]);
    float *input_data_1 = read_fits_image(input_data_file_1, imasize);
    if (input_data_1 == NULL) {
        return 1;
    }
    
    sprintf(input_data_file_2, "%s", argv[2]);
    float *input_data_2 = read_fits_image(input_data_file_2, imasize);
    if (input_data_2 == NULL) {
        return 1;
    }
    
    sprintf(input_snap_file, "%s", argv[3]);
    float *input_snap = read_fits_image(input_snap_file, imasize);
    if (input_snap == NULL) {
        return 1;
    }
    
    float* data_1_array = input_data_1;
    float* data_2_array = input_data_2;
    
    float* result_array = (float*)malloc(unit_num*unit_num*sizeof(float));
    
    splittlisi(data_1_array, data_2_array, input_snap, result_array, imapow, unit_size, ima, maxall);
	
    long naxes[2] = {long(unit_num), long(unit_num)};
    int status = write_fits_image("output_tLISI.fits", result_array, naxes);
    if (status) {
        fprintf(stderr, "Error writing FITS image\n");
        return 1;
    }
}
