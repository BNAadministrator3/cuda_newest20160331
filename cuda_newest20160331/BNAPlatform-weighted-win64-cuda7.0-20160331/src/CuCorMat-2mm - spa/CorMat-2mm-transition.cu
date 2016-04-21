#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <iostream>
#include <ctime>
# include <fstream>
#include <vector>
#include <Windows.h>
#include<iomanip>
//#include <sm_20_atomic_functions.h>
//#include <algorithm> 
#include "quicksort.h"
#include<functional> /* greater in function object */


using namespace std;

#define ep  1e-6  //third question
#define width 0.1//best to be the multiples 

#pragma comment(lib,"cublas.lib")
typedef float real__t;
typedef unsigned int uint__t;

const int thread_num = 256; //maybe redefinition
const int block_num = 48;     
const int blocksize = 1024*1024*48;

bool IsNumber(double x)
{
	return (x == x);
}
bool IsFiniteNumber(double x)
{
	return (x <= DBL_MAX && x >= -DBL_MAX);
}
bool myfunction (real__t i,real__t j) { return (i>j); }
void select(vector<real__t>::iterator A,long long n,long long k);

#define TOM(byteValue) (byteValue/1024/1024)
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
	if(cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
		//exit(-1);
		system("pause");
	}
}
/*
__global__ void diagnalKernel(real__t* devCormat, int Batch_size)
{
	uint__t i = (threadIdx.x + blockIdx.x * blockDim.x) * (Batch_size + 1);
	devCormat[i] = 0; //take care!

}*/

/*__global__ void histoKernel(real__t* devCormat, int Batch_size, uint__t * histo, uint__t Num_Bins)
{
	__shared__ uint__t temp[100];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;
  if (i<100)
	{
		temp[i] = 0;
	}
	
  __syncthreads();

  while(i<Batch_size*Batch_size)
  {
	  //map devCormat[i] to subscript of temp array
	  int x = (int)(devCormat[i] * 10.0); //take care!
	  atomicAdd(&(temp[x]), 1);   //maybe overflow tak places when encountering NaN ,which means exceeding the Num_Bins-defined range of array temp. 
      i += offset;
  }

  __syncthreads();
  if (i<100)
  {
	  atomicAdd(&(histo[i]), temp[i]);//take care!
   }
  
}*/

real__t CorMat_spa2rth(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *s_thresh)
{
	real__t * BOLD_t1, * BOLD_t2, * tempout;
	const int Num_Blocks = (N + Batch_size - 1) / Batch_size;
	uint__t N0 = Num_Blocks * Batch_size;
	uint__t amount =	N * (*s_thresh) * (N-1)  / 100.0 /2.0 ;
	amount += amount%2;
	cout<<"nonzero numbers: "<<amount<<endl;
	//uint__t invaccount = N - account;
	uint__t Num_Bins = 1.0 / width + 1; //take care! 
	uint__t position = 0;

	// transposing the BOLD signal
	real__t * BOLD_t = new real__t [L * N0];
	tempout = new real__t[Batch_size * Batch_size];
	memset(BOLD_t, 0, sizeof(real__t) * L * N0);
	for (int i = 0; i < L; i ++)
		for (int j = 0; j < N; j++)
		{
			BOLD_t[j * L + i] = BOLD[i * N + j];
		}
		
		for (long i = L * N; i < L * N0; i++)
		{
			BOLD_t[i] = 0;
		}	
		
		// Normalize
		for (int i = 0; i < N; i++)
		{
			real__t * row = BOLD_t + i * L;
			double sum1 = 0, sum2 = 0;
			for (int l = 0; l < L; l++)
			{
				sum1 += row[l];
			}
			sum1 /= L;
			for (int l = 0; l < L; l++)
			{
				sum2 += (row[l] - sum1) * (row[l] - sum1);
			}
			sum2 = sqrt(sum2);
			for (int l = 0; l < L; l++)
			{
				row[l] = (row[l] - sum1) / sum2;;
			}
		}
		cudaError_t cudaStat;
		cublasStatus_t stat;
		cublasHandle_t handle;
		real__t * devBOLD, * devCormat;
		//uint__t *devhisto;
		cudaStat = cudaMalloc ((void**)&devBOLD, sizeof(real__t) * L * N0) ;
		if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
		cudaStat = cudaMalloc ( (void**)&devCormat, sizeof(real__t) * Batch_size * Batch_size) ;
		if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
		stat = cublasSetMatrix(N0, L, sizeof(real__t), BOLD_t, N0, devBOLD, N0);
		stat = cublasCreate(&handle) ;
		if (stat != CUBLAS_STATUS_SUCCESS)
			return stat;
		delete []BOLD_t;

		//是指GPU block的个数！
		cout<<"block numbers: "<<Num_Blocks<<endl;
		const float alpha = 1.0;
		const float beta = 0;
		vector< vector<real__t> > bin;
		uint__t *histogram = new uint__t[Num_Bins]; 
		bin.resize(Num_Bins);   
		memset(histogram,0,sizeof(uint__t)*Num_Bins);
		//checkCudaErrors( cudaMemcpy( devhisto, histogram, sizeof(uint__t) * Num_Bins, cudaMemcpyHostToDevice) ); 
		real__t bottomLine = 0;
		clock_t time;
		time = clock();
		for (int kk = 0, ii = 0; ii < Num_Blocks; ii++)
		{
			for (int jj = ii; jj < Num_Blocks; jj++)
			{
				  
				BOLD_t1 = BOLD_t + ii * Batch_size * L;
				BOLD_t2 = BOLD_t + jj * Batch_size * L;
			//	  real__t *v425 = new real__t[L];
   				real__t *out = new real__t[Batch_size * Batch_size];

#ifdef CPUCormat
                MatrixMultiplication_s(BOLD_t1, BOLD_t2, out, Batch_size,L);//need modify as well.
#else
				stat = cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, Batch_size, Batch_size, L,  &alpha, devBOLD + jj * Batch_size * L, L, devBOLD + ii * Batch_size * L, L, &beta, devCormat, Batch_size);
				if (stat != CUBLAS_STATUS_SUCCESS)
					return stat;
				cudaStat = cudaMemcpy(out, devCormat, sizeof(real__t) * Batch_size * Batch_size, cudaMemcpyDeviceToHost);
				if (cudaStat != cudaSuccess) 
					return cudaStat;
				/*
				if (ii==jj)
				{
					diagnalKernel<<<block_num,thread_num>>>(devCormat, Batch_size);
				}  */
             //   histoKernel<<<block_num,thread_num>>>(devCormat, Batch_size,  devhisto,  Num_Bins); //maybe overflow
				
			 //	checkCudaErrors( cudaMemcpy(histogram, devhisto, sizeof(uint__t) * Num_Bins, cudaMemcpyDeviceToHost) );
				
#endif
				
				if (ii==jj)
				{
					for (int i = 0; i < Batch_size; i++)
					{
						out[i*Batch_size+i] = 0;
					}
				}
				for (int i = 0; i < Batch_size * Batch_size; i++)
				{
					if ((!IsNumber(out[i]))||(!IsFiniteNumber(out[i]))||out[i]<0)
					{
						out[i] = 0;
					}
					//cout<<BOLD_t[i]<<endl;
				}  
				//1.split flow to every single vector
				if(ii!=jj)
				{
					for (long i = 0; i<Batch_size*Batch_size;i++)
					{
					
						if (out[i]>1+ep)
						{
							cout<<"data error: "<<out[i]<<endl;
						}
						uint__t temp = (int)(out[i]/width); 
						if (temp>=bottomLine&&temp<=Num_Bins)
						{
							bin[temp].push_back(out[i]);
						}
						else
						{
							if (temp<0||temp>Num_Bins)
							{
								cout<<"Error occur:  "<<i<<"  "<<out[i]<<endl;
								break;					
							}
							
						}
					
					}
				}
				else
				{
					for (int i = 0; i < Batch_size; i++)
					{
						for (int j = i; j < Batch_size; j++)
						{
							if (out[i*Batch_size+j]>1+ep)
						{
							cout<<"data error: "<<out[i*Batch_size+j]<<endl;
						}
						uint__t temp = (int)(out[i*Batch_size+j]/width); 
						if (temp>=bottomLine&&temp<=Num_Bins)
						{
							bin[temp].push_back(out[i*Batch_size+j]);
						}
						else
						{
							if (temp<0||temp>Num_Bins)
							{
								cout<<"Error occur:  "<<i<<"  "<<out[i]<<endl;
								break;					
							}
						}
						}
					}
				}
				delete []out;
				//2.make a judement: positioning and then clear former vector(s); 
				for (int i = 0; i < Num_Bins; i++)
				{
					histogram[i] = bin[i].size();
				}
				bool clearFlag = false;
				for (int i = 0; i < Num_Bins-1; i++)
				{
					 histogram[Num_Bins-i-1-1] += histogram[Num_Bins-i-1];

				}
				for (int i = 0; i < Num_Bins-1; i++)
				{
					  if(histogram[Num_Bins-i-1]>amount)
					  {
						  clearFlag = true;
						  position = Num_Bins-i-1;
						  break;
					  }
				}
 				if (clearFlag&&position!=0)
				{
					for (int i = 0; i < position; i++)
					{
						vector<real__t>().swap(bin[i]);
					}
					bottomLine = position * width;
				}
			//cout<<"NO. "<<"ii: "<<ii<<"  jj: "<<jj<<endl;
			//cout<<"position: "<<position<<endl;
			/*uint__t x = 0;
			for (uint__t i = 0; i < Num_Bins; i++)
			{
				x += bin[i].capacity();
			}
			cout<<"capacity: "<<x<<endl;
			*/
			}
			cout<<"Fulfill the "<<ii+1<<"th block."<<endl;
		}
		//3.in the end clear latter vector(s); rank residual sole vector to find the corresponding correlation
		if(position != Num_Bins-1)
		{
			for (int i = position; i < Num_Bins-1; i++)
			{
				vector<real__t>().swap(bin[i+1]);
			}
		}
		long subscript = amount - histogram[position+1];
		for (int i = 0; i < 11; i++)
		{
			cout<<histogram[i]<<endl;
		}
		if (subscript<0)
		{
			cout<<"Error: unable to sort."<<endl;
		}
		else
		{
			//sort (bin[position].begin(), bin[position].end(), myfunction);
			select(bin[position].begin(),distance(bin[position].begin(),bin[position].end()), subscript);
			
		}
				
		real__t result = bin[position][subscript-1];

		    //display and put out 
			time = clock() - time;
			cout<<"calculation time: "<<time<<"ms"<<endl;
			//time_t nowTime;
			unsigned int FreeMem = 0;
			MEMORYSTATUS MemStat;
			MemStat.dwLength = sizeof(MEMORYSTATUS);
			GlobalMemoryStatus(&MemStat);
			FreeMem = TOM(MemStat.dwAvailPhys);
			cout << "bytes of physical memory: " << TOM(MemStat.dwTotalPhys) <<"M" <<endl;
			cout << "percent of memory in use: " << MemStat.dwMemoryLoad <<"%" <<endl;
			cout << "free physical memory bytes: " << TOM(MemStat.dwAvailPhys) <<"M" <<endl;
			
			cudaFree (devBOLD); 
			cudaFree (devCormat);
			stat = cublasDestroy(handle);
			if (stat != CUBLAS_STATUS_SUCCESS)
				return stat;
		//	delete []BOLD_t;
			return result;
}
void MatrixMultiplication_s(real__t * BOLD_t1, real__t * BOLD_t2,real__t * out,int Batch_size,int L)//do not announce
{
	long kk = 0;
	for (int k = 0; k < Batch_size; k++)
	{
		for (int i = 0; i < Batch_size; i++)
		{   
			double sum3 = 0.0;
			for (int j = 0; j < L; j++)
			{
				sum3 += 1.0*BOLD_t1[k*L+j] * BOLD_t2[i*L+j];
			}
			out[kk++] = sum3;
		}
	}
	
}
long long partition(vector<real__t>::iterator A, long long  m,long long  p){
        long long i;
		real__t	tem;
		real__t v;
        v=*(A+m);i=m+1;
		//cout<<v<<endl;
        while(1){
			while(*(A+i)>v && i<p)
				i++;
            while(*(A+p)<=v && i<=p)
				p--;
			if(i<p){
                tem=*(A+i);*(A+i)=*(A+p);*(A+p)=tem;
            }else break;
        }
        *(A+m)=*(A+p);*(A+p)=v;return (p);
}

void select(vector<real__t>::iterator A,long long n,long long k){
	long long j,m,r;
	m=0;r=n-1;
	//cout<<"k = "<<k<<endl;
	while(1){		
		j=r;
		//cout<<"m = "<<m<<" ; r = "<< r<< endl;
		//cout<<"A[m] = "<<A[m]<<" ; A[r] = "<< A[r]<< endl;
		//clock_t time = clock();
		j=partition(A, m,j);
		//time = clock() - time;
		//cout<<"partition time : "<<time<<"; j ="<<j<<endl;
		//cout<<"j = "<<j<<endl;
		if(k-1==j)break;
		else if(k-1<j) r=j-1;
		else m=j+1;
	}
}

