#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <ctime>
#include <cmath>
#include <iomanip>
#include "dirent.h"
#include <iostream>
#include <fstream>
#include <stack>
#include<vector>
#include"fileMapping.h"
#include "shlwapi.h"                      //used for creating folder
#pragma comment(lib,"shlwapi.lib")
using namespace std;

typedef float real__t;
typedef unsigned int uint__t;

const int HdrLen = 352;
double ProbCut = 0.5; 
const int blocksize = 1024*1024*48;

bool rs_flag = true;
bool cormat_flag = false;
char mask_dt;

real__t interval(void);
//int CorMat_gpu(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *r_thresh,clock_t* aggregrate);
//int CorMat_gpu(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *r_thresh,real__t width, real__t *s_thresh, clock_t* aggregrate);
void CorMat_cpu(real__t * Cormat, real__t * BOLD, int N, int L);
int CorMat_gpu(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *r_thresh,clock_t* aggregrate);
real__t CorMat_spa2rth(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *s_thresh, clock_t* aggregate);
long long FormFullGraph(bool * adjacent, real__t * Cormat, int N, real__t threshold);
real__t FormFullGraph_s(bool * adjacent, real__t * Cormat, int N, long long threshold);
void post_block (real__t * Cormat, real__t * Cormat_blocked, int dim, int block_size,bool fishtran_flag);
void FormCSRGraph(int * R, int * C, real__t *V, bool * adjacent, int N , real__t *Cormat);
long long find_max(real__t *Cormat, long long M1);
real__t select_GPU(real__t *Cormat, long long M1, long long k);

real__t fishertrans(real__t r)
{
	real__t z;
	if (r==1) r -= 1e-6; 
	z = 0.5*log((1+r)/(1-r));
	return z;	
}

real__t inv_fishertrans(real__t z)
{
	real__t r;
	r = exp(2*z);
	r = (r-1)/(r+1);
	return r;	
}

int main(int argc, char * argv[])
{
	clock_t total_time = clock();
	if (argc < 7) 
	{
		cerr<<"Input format: .\\CorMat.exe  Dir_for_BOLD Path_for_mask  threshold_for_mask(0~1) to_average(yf/yn/bf/bn/n) to_save_cormatrix(y/n) threshold_type(r/s) threshold_for_correletaion_coefficient(s)(0~1)\n"
			<<"For example: .\\CorMat.exe  d:\\BOLD d:\\MASK\\mask.nii 0.5 y n r 0.2 0.25 0.3\n"<<endl;
		exit(1);	
	}
	int L, N = 0, i = 0, j = 0, k = 0, l = 0, total_size;
	clock_t time;

	DIR *dp;
	struct dirent *dirp;
	if (NULL == (dp = opendir(argv[1])))
	{
		printf("can't open %s", argv[1]);
		exit (1);  
	}
	int FileNumber = 0;
	string filenametmp;
	while((dirp = readdir(dp)) != NULL)
	{
		filenametmp = string(dirp->d_name);
		//cout<<filenametmp.c_str()<<"!"<<endl;
		if (filenametmp.find_last_of('.') == -1)
			continue;
		if(filenametmp.length()>4 && filenametmp.substr(filenametmp.find_last_of('.'),4).compare(".nii") == 0 && filenametmp.size() - filenametmp.find_last_of('.') - 1 == 3)
		{
			if (filenametmp.compare("mask.nii")!=0&&filenametmp.compare("grey_mask.nii")!=0&&filenametmp.compare("1mm_grey.nii")!=0&&filenametmp.compare("1mm_grey_mask.nii")!=0)
				FileNumber++;
		}
	}
	cout<<FileNumber<<" file(s) to be processed."<<endl;

	closedir(dp);
	string *filename = new string[FileNumber];
	dp = opendir(argv[1]);
	i = 0;
	while((dirp = readdir(dp)) != NULL)
	{
		filenametmp = string(dirp->d_name);
		//	cout<<"here";
		if (filenametmp.find_last_of('.') == -1)
			continue;
		if(filenametmp.length()>4 && filenametmp.substr(filenametmp.find_last_of('.'),4).compare(".nii") == 0 && filenametmp.size() - filenametmp.find_last_of('.') - 1 == 3)
		{
			if (filenametmp.compare("mask.nii")!=0&&filenametmp.compare("grey_mask.nii")!=0&&filenametmp.compare("1mm_grey.nii")!=0&&filenametmp.compare("1mm_grey_mask.nii")!=0)
				filename[i++] = filenametmp;
		}
	}

	real__t ProbCut = (real__t)atof(argv[3]);
	int NumS = argc - 7;
	real__t * r_thresh = new real__t [NumS];
	real__t * s_thresh = new real__t [NumS];
	if (argv[6][0] == 'r' || argv[6][0] == 'R' )
		for (i = 0; i < NumS; i++)
			r_thresh[i] = (real__t)atof(argv[7+i]);
	else if (argv[6][0] == 's' || argv[6][0] == 'S' )
	{
		rs_flag = false;
		memset(r_thresh, 0, sizeof(real__t)*NumS);
		for (i = 0; i < NumS; i++)
			s_thresh[i] = (real__t)atof(argv[7+i]);
	}
	else {
		cout << "threshold type error! \nr for correlation threshold that is sole currently.\n";
		exit(1);
	}

	if(argv[5][0]=='y' || argv[5][0]=='Y')
	{
		cormat_flag = true;
	}
	else if (argv[5][0]=='N' || argv[5][0]=='n')
	{
		cormat_flag = false;
	}
	else
	{
		cout << "to_save_cor_matrix type error! \ny to save the whole correlation matrix \nn to save only csr format of adjacency matrix\n";
		exit(1);
	}

	// read input files and parameters
	if (argv[1][strlen(argv[1]) - 1] == '\\')
		argv[1][strlen(argv[1]) - 1] = 0;
	/************************************mask file****************************************/
	string mask_file = string(argv[2]); 
	ifstream fin(mask_file.c_str(), ios_base::binary);
	if (!fin.good())
	{	cout<<"Can't open\t"<<mask_file.c_str()<<endl;	return 0; }
/*	ifstream fin(string(argv[1]).append("\\mask.nii").c_str(), ios::binary);
	if (!fin.good())
	{	cout<<"Can't open\t"<<string(argv[1]).append("\\mask.nii").c_str()<<endl;	return 0;}*/
	short hdr[HdrLen / 2];
	fin.read((char*)hdr, HdrLen);  //35 for mask and 36 for bold
	cout<<"mask datatype : "<< hdr[35]<<"  "<<hdr[36]<<endl;
	mask_dt = hdr[35];

	total_size = hdr[21] * hdr[22] * hdr[23];	// Total number of voxels
	
	real__t * mask = new float [total_size];

	if (mask_dt==2)
	{
		unsigned char *mask_uc = new unsigned char[total_size];
		fin.read((char *) mask_uc, sizeof(unsigned char) * total_size);
		for (int vm = 0; vm<total_size; vm++)
			mask[vm] = (float) mask_uc[vm];
		delete [] mask_uc;
	}
	else if(mask_dt==16)
	{
		fin.read((char *)mask, sizeof(float) * total_size);
	}
	else
	{
		cout<<"mask data-type error, Only the type of unsigned char and float can be handled.\n";
		//system("pause");
		return -1;
	}
	fin.close();
	// Count the number of the valid voxels	
	for (k = 0; k < total_size; k++)
		N += (mask[k] >= ProbCut);	
	cout<<"Data size: "<<hdr[21] <<"x"<<hdr[22]<<"x"<<hdr[23]  <<", Grey voxel count: "<<N<<"."<<endl;
	// swap the largest threshold to the beginning
	int min_idx = 0;
	for (i = 0; i < NumS; i++)
		if (r_thresh[i] < r_thresh[min_idx])
			min_idx = i;
		real__t temp = r_thresh[0];
		r_thresh[0] = r_thresh[min_idx];
		r_thresh[min_idx] = temp;
	//process, do not average	
		for (int i = 0; i < FileNumber; i++)
		{
/****************************************BOLD file***********************************************/ 
			string a = string(argv[1]).append("\\").append(filename[i]);
			cout<<"\ncalculating coarse-grained threshold for "<<a.c_str()<<" ..."<<endl;
			ifstream fin(a.c_str(), ios_base::binary);

			// Get the length of the time sequence 
			if (!fin.good())
			{	cout<<"Can't open\t"<<a.c_str()<<endl;	return 0;}
			fin.read((char*)hdr, HdrLen);
			L = hdr[24];
			//L = 1200;
			real__t * BOLD = new real__t [L * N];
			if (hdr[36] == 64) // double
			{
				double *  InData = new double [L * total_size];
				fin.read((char*)InData, sizeof(double) * L * total_size);
				fin.close();
				// Get the BOLD signal for all the valid voxels
				for (int i = -1, k = 0; k < total_size; k++)
					if (mask[k] >= ProbCut)
						for (i++, l = 0; l < L; l++)
						{
							BOLD[l*N+i] = InData[l*total_size+k];
						}
						cout<<"BOLD length: "<<L<<", Data type: double."<<endl;
						delete []InData;
			}
			else if (hdr[36] == 32)	   //float
			{
				//places that need amend
				//verification
				/*
				real__t *InData = new float [L * total_size];
				fin.read((char*)InData, sizeof(float) * L * total_size); 
				//L = 120;
				
				// Get the BOLD signal for all the valid voxels
				for (int i = -1, k = 0; k < total_size; k++)
					if (mask[k] >= ProbCut)
						for (i++, l = 0; l < L; l++)
						{
							BOLD[l*N+i] = InData[l*total_size+k];
						}

				fin.seekg(-sizeof(float) * L * total_size,ios::cur);  
				real__t * BOLD_b = new real__t [L * N];
				uint__t piece = 100;
				int sheet = L / piece + 1;
				uint__t tile = total_size*piece;;
				real__t *InData_b = new float [tile];//[L * total_size];
				for (uint__t z = 0; z < sheet; z++)
				{
					cout<<z<<endl;
					uint__t bound = piece*(z+1)< L?piece*(z+1):L;
					fin.read((char*)InData_b, sizeof(float) * tile);//sizeof(float) * (z==sheet-1?(total_size*L-total_size*tile*z):tile));
					for (int i = -1, k = 0; k < total_size; k++)
						if (mask[k] >= ProbCut)
							for (i++, l = 0+piece*z; l <bound; l++)
							{
								BOLD_b[l*N+i] = InData_b[(l-piece*z)*total_size+k];
							}
				}
				delete []InData;
				for (long i = 0; i < L * N; i++)
				{
					if (BOLD_b[i]!=BOLD[i])
					{
						cout<<"Trouble£¡"<<endl;
						cout<<i<<endl;
						break;
					}
				}
				fin.close();
				*/
				try{
				real__t *InData = new float [L * total_size];
				fin.read((char*)InData, sizeof(float) * L * total_size); 
				//L = 120;
				fin.close();
				// Get the BOLD signal for all the valid voxels
				for (int i = -1, k = 0; k < total_size; k++)
					if (mask[k] >= ProbCut)
						for (i++, l = 0; l < L; l++)
						{
							BOLD[l*N+i] = InData[l*total_size+k];
						}
				delete []InData;
				}
				catch(...){
				 cout<<total_size<<"*"<<L<<endl;
					//real__t * BOLD_com = new real__t [L * N];
					uint__t piece = 100;
					uint__t tile = total_size*piece;
					int sheet = L / piece + 1;
					real__t *InData = new float [tile];//[L * total_size];
					for (uint__t z = 0; z < sheet; z++)
					{
						cout<<z<<endl;
						uint__t bound = piece*(z+1)< L?piece*(z+1):L;
						fin.read((char*)InData, sizeof(float) * tile);
						for (int i = -1, k = 0; k < total_size; k++)
							if (mask[k] >= ProbCut)
								for (i++, l = 0+piece*z; l <bound; l++)
								{
									BOLD[l*N+i] = InData[(l-piece*z)*total_size+k];
								}
					}
					fin.close();
					delete []InData;
				}  
				
				
				cout<<"BOLD length: "<<L<<", Data type: float."<<N<<endl;
						
			}
			else
			{
			  cerr<<"Error: Data type is neither float nor double."<<endl;
			}
			
			// set some parameters 
			int Batch_size = 1024 * 3;
			int GPU_Batch_size = Batch_size * 2;
			int CPU_Batch_size = Batch_size *5;
			// should not be smaller than 1024 !
			int Num_Blocks = (N + Batch_size - 1) / Batch_size;
			long long M2 = Num_Blocks * (Num_Blocks + 1) / 2;
			M2 *= Batch_size * Batch_size;
			// cout<<M2<<endl;
			//real__t * Cormat_blocked = new real__t [M2];

			long long M1 = (N-1);
			M1 *= N;
			M1 /= 2;
			//real__t * Cormat_gpu = new real__t [M1];

			// Begin computing correlation	
			//time = clock();
			char sparsity[30];
			char Graph_size[30];
			string str = string(argv[1]).append("\\").append("weighted");
			if (!PathIsDirectory(str.c_str()))
			{
				::CreateDirectory(str.c_str(), NULL);
			}
			a=string(argv[1]).append("\\").append("weighted").append("\\").append(filename[i]);
			string OutCor = a.substr(0, a.find_last_of('.')).append("_0").append(string(itoa(N, Graph_size, 10)));

			real__t rthresh = 0;
			uint__t sortOrder = 0;
			clock_t *aggregate =(clock_t *) malloc(sizeof(clock_t));
			rthresh = CorMat_spa2rth(OutCor, BOLD, N, L, GPU_Batch_size,s_thresh, aggregate);
			//rthresh = 0.8;
			//CorMat_gpu(OutCor, BOLD, N, L, CPU_Batch_size,&rthresh,interval(),s_thresh, aggregate);
			//CorMat_gpu(OutCor, BOLD, N, L, CPU_Batch_size,r_thresh,aggregate);
			//crucial error!rthresh rather than r_thresh!
			CorMat_gpu(OutCor, BOLD, N, L, CPU_Batch_size,&rthresh, aggregate);

			delete []BOLD;
		}
		delete []mask;
		delete []r_thresh;
		delete []filename;
		total_time = clock() - total_time;
		cout<<"total elapsed time: "<<1.0*total_time/1000<<" s."<<endl;
		cout<<"==========================================================="<<endl;
		return 0;
}

