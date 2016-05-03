#include "cublas_v2.h"
#include "cusparse.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>
#include <Windows.h>
#include<iomanip>
#pragma comment(lib,"cusparse.lib")
#include "Utilities.cuh"


using namespace std;

#define ep  1e-6  //third question

#pragma comment(lib,"cublas.lib")
typedef float real__t;
typedef unsigned int uint__t;

#define TOM(byteValue) (byteValue/1024/1024)
extern void cusparseSafeCall(cusparseStatus_t err);

//#define CPUCormat 0

typedef struct cv
		{  
		 int column;
		 real__t value;
		} ColumnValueInfo;    //Global definition is necessary



const int thread_num = 256;
const int block_num = 48;
const int blocksize = 1024*1024*48;

void select(real__t *A,long long n,long long k);
void MatrixMultiplication(real__t * BOLD_t1, real__t * BOLD_t2,real__t * out,int Batch_size,int L);

void Thrust(vector <vector<ColumnValueInfo>>::iterator begin, real__t *out, int ii, int Batch_size, real__t r_thresh, real__t er);
void ThrustAsymmetrical(vector <vector<ColumnValueInfo>>::iterator begin, real__t *out, int ii, int jj, int Batch_size, real__t r_thresh, real__t er);

__global__ void standardAndThresholdingKernel(real__t* devCormat, int Batch_size, bool diagnoal, real__t thres)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
	while(i<Batch_size*Batch_size) 
	{
		if (!(devCormat[i]>(thres-ep) && devCormat[i]<=(1+ep)) ) //flag!
		{
			devCormat[i] = 0;
		}
		if(diagnoal==true)
		{
			if (i%(Batch_size+1)==0)
		    {
			devCormat[i] = 0;
		    }
		}
		 i += offset;
	}
}
int CorMat_gpu(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *r_thresh,clock_t* aggregrate)
{
	real__t * BOLD_t1, * BOLD_t2, * tempout;
	const int Num_Blocks = (N + Batch_size - 1) / Batch_size;
	uint__t N0 = Num_Blocks * Batch_size;

	// transposing the BOLD signal
	real__t * BOLD_t = new real__t [L * N0];
	tempout = new real__t[Batch_size * Batch_size];
	memset(BOLD_t, 0, sizeof(real__t) * L * N0);
	for (int i = 0; i < L; i ++)
		for (int j = 0; j < N; j++)
		{
			BOLD_t[j * L + i] = BOLD[i * N + j];
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
		cusparseStatus_t  sparseStat;
		cublasHandle_t handle;
		cusparseHandle_t sparseHandle;
		cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;
		cusparseMatDescr_t descrA = 0;
		sparseStat= cusparseCreate(&sparseHandle);
        if (sparseStat != CUSPARSE_STATUS_SUCCESS)
			 return sparseStat;
		/* create and setup matrix descriptor */ 
		sparseStat= cusparseCreateMatDescr(&descrA); 
		if (sparseStat != CUSPARSE_STATUS_SUCCESS) 
	          return sparseStat; 
		cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);  
		 //sparseStat = cusparse_create_mat_descr(descrA); 
		real__t * devBOLD, * devCormat,*csrValA;
		int *nnzPerRowColumn,*csrRowPtrA,*csrColIndA;
		cudaStat = cudaMalloc ((void**)&nnzPerRowColumn, sizeof(int) * Batch_size) ;
		if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
		int nnzTotalDevHostPtr = 0;
		cudaStat = cudaMalloc ((void**)&csrRowPtrA, sizeof(int) * (Batch_size+1) ) ;
		if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
//		stat = cublasAlloc(L*N0, sizeof(real__t), (void**)&devBOLD);
		cudaStat = cudaMalloc ((void**)&devBOLD, sizeof(real__t) * L * N0) ;
		if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
//		stat = cublasAlloc(Batch_size * Batch_size, sizeof(real__t), (void**)&devCormat);		
		cudaStat = cudaMalloc ( (void**)&devCormat, sizeof(real__t) * Batch_size * Batch_size) ;
		if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
		stat = cublasSetMatrix(N0, L, sizeof(real__t), BOLD_t, N0, devBOLD, N0);
//		cudaStat = cudaMemcpy(devBOLD, BOLD_t, sizeof(real__t) * L * N0, cudaMemcpyHostToDevice);
		stat = cublasCreate(&handle) ;
		if (stat != CUBLAS_STATUS_SUCCESS)
			return stat;
				                                       

		//是指GPU block的个数！
		cout<<"block numbers: "<<Num_Blocks<<endl;
		const float alpha = 1.0;
		const float beta = 0;
		vector <vector<ColumnValueInfo>> ColumnAndValue; 
		vector <int> Row; 
		ColumnAndValue.resize(Num_Blocks*Batch_size);  
		Row.resize(Num_Blocks*Batch_size+1);  //consider whether allocate space just here or other.
		for (int i = 0; i < Num_Blocks*Batch_size+1; i++)
		{
			Row.push_back(0);
		}
		clock_t correlationTime = clock();
		real__t *out = new real__t[Batch_size * Batch_size];
		clock_t time;
		time = clock();
		for (int kk = 0, ii = 0; ii < Num_Blocks; ii++)
		{
			for (int jj = ii; jj < Num_Blocks; jj++)
			{
				  
				BOLD_t1 = BOLD_t + ii * Batch_size * L;
				BOLD_t2 = BOLD_t + jj * Batch_size * L;
				//  real__t *v425 = new real__t[L];
#ifdef CPUCormat
                MatrixMultiplication(BOLD_t1, BOLD_t2, out, Batch_size,L);
#else																																			//has been amending					
				stat = cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, Batch_size, Batch_size, L,  &alpha, devBOLD + ii * Batch_size * L, L, devBOLD + jj * Batch_size * L, L, &beta, devCormat, Batch_size);//virtually kernel
				if (stat != CUBLAS_STATUS_SUCCESS)
					return stat;
				//so 1.thresholding 2.dense2csr 3.transfer 4.free
				//start 1.thresholding
				standardAndThresholdingKernel<<<block_num,thread_num>>>(devCormat, Batch_size, ii==jj, *r_thresh);  
				//2. dense2csr
				sparseStat = cusparseSnnz(sparseHandle, CUSPARSE_DIRECTION_ROW, Batch_size, Batch_size, descrA, devCormat,  Batch_size, nnzPerRowColumn, &nnzTotalDevHostPtr);
				if (sparseStat !=CUSPARSE_STATUS_SUCCESS)
					return sparseStat;
				cudaStat = cudaMalloc ((void**)&csrValA, sizeof(real__t) * nnzTotalDevHostPtr) ;
				if (cudaStat != CUBLAS_STATUS_SUCCESS) 
				return cudaStat;
				cudaStat = cudaMalloc ((void**)&csrColIndA, sizeof(int) * nnzTotalDevHostPtr) ;
				if (cudaStat != CUBLAS_STATUS_SUCCESS) 
				return cudaStat;
				sparseStat = cusparseSdense2csr(sparseHandle, Batch_size, Batch_size, descrA, devCormat, Batch_size, nnzPerRowColumn, csrValA, csrRowPtrA, csrColIndA);
				if (sparseStat != CUSPARSE_STATUS_SUCCESS)
					return sparseStat;
				//3. transfer
				int *hostCsrColIndA = new int[nnzTotalDevHostPtr];
				int *hostCsrRowPtrA = new int[Batch_size+1];
				real__t *hostCsrValA = new real__t[nnzTotalDevHostPtr];
				cudaStat = cudaMemcpy(hostCsrColIndA, csrColIndA, sizeof(int) * nnzTotalDevHostPtr, cudaMemcpyDeviceToHost);
				if (cudaStat != cudaSuccess) 
					return cudaStat;
				cudaStat = cudaMemcpy(hostCsrValA, csrValA, sizeof(real__t) * nnzTotalDevHostPtr, cudaMemcpyDeviceToHost);
				if (cudaStat != cudaSuccess) 
					return cudaStat;
				cudaStat = cudaMemcpy(hostCsrRowPtrA, csrRowPtrA, sizeof(real__t) * (Batch_size+1), cudaMemcpyDeviceToHost);
				if (cudaStat != cudaSuccess) 
					return cudaStat;
				//4.1 free one
				cudaFree(csrValA);
				cudaFree(csrColIndA);
#endif	
				ColumnValueInfo tmp;
				if(ii==jj)
				{
					//Thrust(ColumnAndValue.begin(), out, ii, Batch_size,  *r_thresh,  ep);
					//for (int i = 0; i < Batch_size; i ++)
					//{
					//	for (int j = 0; j < Batch_size; j++)
					//    { 
					//		if(out[i * Batch_size + j]>(*r_thresh-ep)&&out[i * Batch_size + j]<=(1+ep)&&(i!=j))
					//		{
					//			//count ++;
					//			nonzerocount++;
					//		    tmp.column = j;
					//			tmp.column += ii * Batch_size;
					//			tmp.value = out[i*Batch_size+j];
					//			ColumnAndValue[ii*Batch_size+i].push_back(tmp);
					//		}
					//	}
					// }
					for (int i = 0; i < Batch_size; i++)
					{
						for (int j = hostCsrRowPtrA[i]; j < hostCsrRowPtrA[i+1]-1; j++)
						{
							tmp.column = hostCsrColIndA[j];
							tmp.column += ii * Batch_size;
							tmp.value = hostCsrValA[j];
							ColumnAndValue[ii*Batch_size+i].push_back(tmp);
						}
					}
				}
				else
				{
					//ThrustAsymmetrical(ColumnAndValue.begin(), out, ii, jj, Batch_size, *r_thresh, ep);
					//for (int i = 0; i < Batch_size; i ++)
					//{
					//	for (int j = 0; j < Batch_size; j++)
					//	{ 
					//		if( out[i * Batch_size + j]>(*r_thresh-ep) && out[i * Batch_size + j]<=(1+ep) )
					//		{
					//		  	//nonzerocount += 2;
					//		    //1.push row
					//			tmp.column = j;
					//			tmp.column += jj * Batch_size;
					//			tmp.value = out[i*Batch_size+j];
					//			ColumnAndValue[ii*Batch_size+i].push_back(tmp);
					//			//2.push column
					//			tmp.column = i;
					//			tmp.column += ii * Batch_size;
					//			tmp.value = out[i*Batch_size+j];
					//			ColumnAndValue[jj*Batch_size+j].push_back(tmp);
					//		}
					//	}
					//}
					for (int i = 0; i < Batch_size; i++)
					{
						for (int j = hostCsrRowPtrA[i]; j < hostCsrRowPtrA[i+1]-1; j++)
						{
							
							tmp.column = hostCsrColIndA[j];
							tmp.column += jj * Batch_size;
							tmp.value = hostCsrValA[j];
							ColumnAndValue[ii*Batch_size+i].push_back(tmp);
							tmp.column = i;
							tmp.column += ii * Batch_size;
							tmp.value = hostCsrValA[j];
							ColumnAndValue[jj*Batch_size+hostCsrColIndA[j]].push_back(tmp);	
					//		    //1.push row
					//			tmp.column = j;
					//			tmp.column += jj * Batch_size;
					//			tmp.value = out[i*Batch_size+j];
					//			ColumnAndValue[ii*Batch_size+i].push_back(tmp);
					//			//2.push column
					//			tmp.column = i;
					//			tmp.column += ii * Batch_size;
					//			tmp.value = out[i*Batch_size+j];
					//			ColumnAndValue[jj*Batch_size+j].push_back(tmp);
						}
					}
				}
				/*time = clock()-time;
				cout<<"1.gpu time:"<<time<<" ms"<<endl;
				time = clock();*/
				cout<<"Loop flag: "<<ii<<":"<<jj<<endl;
				//4.2 free two
				delete []hostCsrColIndA;
				delete []hostCsrRowPtrA;
				delete []hostCsrValA;
			}
			cout<<"Fulfill the "<<ii+1<<"th disposition."<<endl;
		}
		//4.3 free three
		cudaFree(nnzPerRowColumn);
		sparseStat = cusparseDestroyMatDescr(descrA);  
		if (sparseStat != CUSPARSE_STATUS_SUCCESS)
			return sparseStat; 
		sparseStat = cusparseDestroy(sparseHandle); 
		if (sparseStat != CUSPARSE_STATUS_SUCCESS)
			return sparseStat;
		delete []out;
		Row[0] = 0;
		for (vector <vector<ColumnValueInfo>>::iterator x = ColumnAndValue.begin(); x != ColumnAndValue.end(); x++)
		{
			Row[x-ColumnAndValue.begin() + 1] = (*x).size();
			Row[x-ColumnAndValue.begin() + 1] += Row[x-ColumnAndValue.begin()];
		}
		//display and put out 
		correlationTime = clock() - correlationTime;
		cout<<"correlation time: "<<correlationTime<<"ms"<<endl;
		* aggregrate += correlationTime;
		cout<<"overall time for histogram plus correlation: "<<*aggregrate<<"ms"<<endl;
		//time_t nowTime;
		unsigned int FreeMem = 0;
		MEMORYSTATUS MemStat;
		MemStat.dwLength = sizeof(MEMORYSTATUS);
		GlobalMemoryStatus(&MemStat);
		FreeMem = TOM(MemStat.dwAvailPhys);
		cout << "bytes of physical memory: " << TOM(MemStat.dwTotalPhys) <<"M" <<endl;
		cout << "percent of memory in use: " << MemStat.dwMemoryLoad <<"%" <<endl;
		cout << "free physical memory bytes: " << TOM(MemStat.dwAvailPhys) <<"M" <<endl;
		cout<<"number of non-zero elements: "<<Row[N]<<endl;
		long long M1 = (N-1);
		M1 *= N;
		M1 /= 2;
		real__t spa = 100.0 * Row[N] / M1 / 2.0;
		char sparsity[100];
		sprintf(sparsity, "_spa%.3f%%_cor%.3f", spa,*r_thresh);
		string Outfilename = OutCor;
		Outfilename.append(string(sparsity)).append("_weighted.csr");
		ofstream fout;
		cout<<"generating "<<Outfilename.c_str()<< "..."<<endl;
		fout.open(Outfilename.c_str(), ios::binary | ios::out);
		//fout.open(OutCor.c_str(),ios::binary | ios::out);
		if (!fout)
		{
			cout<<"create unsuccessfully. error code:  "<<GetLastError()<<endl;
			exit(false);

		}
		int Rlength = N+1;
		fout.write((char*)&Rlength, sizeof(int));
		for (int i = 0; i < Rlength; i++)
		{
				int R =Row[i];
				fout.write((char*)&R, sizeof(int));
		}
		int Clength = Row[N];
		fout.write((char*)&Clength, sizeof(int));
		for (vector <vector<ColumnValueInfo>>::iterator i = ColumnAndValue.begin(); i != ColumnAndValue.end(); i++)
		{
			for (vector<ColumnValueInfo>::iterator j = (*i).begin(); j !=(*i).end(); j++)
			{
				int C = (*j).column;
				fout.write((char*)&C, sizeof(int));
			}
				
		}
		fout.write((char*)&Clength, sizeof(int));
		for (vector <vector<ColumnValueInfo>>::iterator i = ColumnAndValue.begin(); i != ColumnAndValue.end(); i++)
		{
			for (vector<ColumnValueInfo>::iterator j = (*i).begin(); j !=(*i).end(); j++)
			{
				real__t V = (*j).value;
				fout.write((char*)&V, sizeof(real__t));
			}
				
		}
		fout.close();
		cout<<"Transmition finished."<<endl;
		cudaFree (devBOLD); 
		cudaFree (devCormat);
		stat = cublasDestroy(handle);
		if (stat != CUBLAS_STATUS_SUCCESS)
			return stat;
		delete []BOLD_t;
		return true;
}
void MatrixMultiplication(real__t * BOLD_t1, real__t * BOLD_t2,real__t * out,int Batch_size,int L)
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








	