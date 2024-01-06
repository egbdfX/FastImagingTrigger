#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "debug.h"

int splitaug(float* input_data_1, float* input_data_2, float* result_array, int size);

using namespace std;

int DEBUG_LEVEL = 1;
int image_size = 1024*1024;

long int get_file_size(ifstream &FILEIN){
	long int count = 0;
	FILEIN.seekg(0,ios::beg);
	for(std::string line; std::getline(FILEIN, line); ++count){}
	return(count);
}

int Load_data(std::vector<float> *data, char *filename){
	int error;
	error=0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		long int file_size = get_file_size(FILEIN);
		
		FILEIN.clear();
		FILEIN.seekg(0,ios::beg);
		for(long int f = 0; f < file_size; f++){
			double tp1;
			FILEIN >> tp1;
			//printf("loaded element: %f\n", tp1);
			data->push_back((float) tp1);
		}

		if(file_size==0){
			printf("\nFile is empty!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}

void Save_data(float *data, size_t length, char *filename){
	//writing results to disk
	std::ofstream FILEOUT;
	FILEOUT.open(filename);
	for(int i = 0; i<length; i++){
		FILEOUT << data[i] << std::endl;
	}
	FILEOUT.close();
}


int main(int argc, char* argv[]) {
    char input_data_file_1[1000];
    char input_data_file_2[1000];

    // Process the first file
    sprintf(input_data_file_1, "%s", argv[1]);
    vector<float> input_data_1;
    Load_data(&input_data_1, input_data_file_1);

    // Process the second file
    sprintf(input_data_file_2, "%s", argv[2]);
    vector<float> input_data_2;
    Load_data(&input_data_2, input_data_file_2);

    float* data_1_array = input_data_1.data();
    float* data_2_array = input_data_2.data();
    
    int unit_size = 60;
    int ima = 3000;
    int unit_num = ima/unit_size;
    
    vector<float> output_data;
    float* result_array = (float*)malloc(unit_num*unit_num*sizeof(float));
    
    splitaug(data_1_array, data_2_array, result_array, input_data_1.size());

	
	Save_data(result_array, unit_num*unit_num, "output_result.txt");
}
