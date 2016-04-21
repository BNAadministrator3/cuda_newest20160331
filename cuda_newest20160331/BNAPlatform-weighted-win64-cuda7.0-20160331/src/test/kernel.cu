#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <iostream>
#include <ctime>
# include <fstream>
#include <vector>
#include <Windows.h>
#include<iomanip>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

const int thread_num = 256; //maybe redefinition
const int block_num = 48;     
const int blocksize = 1024*1024*48;

void main()
{
	unsigned int a = 3641862;
	long negata = a * (-1);
	std::cout<<negata<<std::endl;
	while(1);
}