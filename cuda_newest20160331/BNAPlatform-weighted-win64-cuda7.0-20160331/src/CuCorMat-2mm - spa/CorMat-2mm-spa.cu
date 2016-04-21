#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <iostream>
#include <ctime>
# include <fstream>
#include <vector>
#include <Windows.h>
#include<iomanip>

using namespace std;

#define ep  1e-6  //third question

#pragma comment(lib,"cublas.lib")
typedef float real__t;
typedef unsigned int uint__t;

#define TOM(byteValue) (byteValue/1024/1024)

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

int CorMat_gpu(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *r_thresh)
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

		//// column major in every block
		//real__t * BOLD_t_col = new real__t [L * N0];
		//for (int k = 0; k < Num_Blocks; k++)
		//{
		//	for (int i = 0; i < Batch_size; i ++)
		//		for (int j = 0; j < L; j++)
		//		{
		//			BOLD_t_col[k * Batch_size * L + j * Batch_size + i] = BOLD_t[k * Batch_size * L + i * L + j];
		//		}
		//}

		cudaError_t cudaStat;
		cublasStatus_t stat;
		cublasHandle_t handle;
		real__t * devBOLD, * devCormat;
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
		clock_t time;
		time = clock();
		for (int kk = 0, ii = 0; ii < Num_Blocks; ii++)
		{
			for (int jj = ii; jj < Num_Blocks; jj++)
			{
				  
				BOLD_t1 = BOLD_t + ii * Batch_size * L;
				BOLD_t2 = BOLD_t + jj * Batch_size * L;
				  real__t *v425 = new real__t[L];
				real__t *out = new real__t[Batch_size * Batch_size];
			//	out = Cormat + (long long) kk * Batch_size * Batch_size;
			//	kk++;

#ifdef CPUCormat
                MatrixMultiplication(BOLD_t1, BOLD_t2, out, Batch_size,L);
#else
				stat = cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, Batch_size, Batch_size, L,  &alpha, devBOLD + jj * Batch_size * L, L, devBOLD + ii * Batch_size * L, L, &beta, devCormat, Batch_size);
				if (stat != CUBLAS_STATUS_SUCCESS)
					return stat;
				cudaStat = cudaMemcpy(out, devCormat, sizeof(real__t) * Batch_size * Batch_size, cudaMemcpyDeviceToHost);
				if (cudaStat != cudaSuccess) 
					return cudaStat;
#endif
				//dump out
			/*	if (ii==0&&jj==7)
				{
					ofstream fout;
                string OutCor = "E:\\severve-2mm-outg.csr"; 
                fout.open(OutCor.c_str(),ios::binary | ios::out);
                if (!fout)
                {
                    cout<<"create local output unsuccessfully:"<<GetLastError()<<endl;

                }
                fout.write((char*)out, sizeof(real__t)*Batch_size*Batch_size);
                fout.close();
				}*/
				  
				//right top triangle matrix
				//1.thresholding
				real__t *out_t = new real__t[Batch_size * Batch_size];
				memset(out_t, 0, sizeof(real__t) * Batch_size * Batch_size);
				int nonzerocount=0;
				if(ii==jj)
				{
				  for (int i = 0; i < Batch_size; i ++)
					for (int j = 0; j < Batch_size; j++)
		            { 
						if(out[i * Batch_size + j]>(*r_thresh-ep)&&out[i * Batch_size + j]<=(1+ep)&&(i!=j))
					    {

						  out_t[i * Batch_size + j] = out[i * Batch_size + j]; 

						  nonzerocount++;
					    }
				    }
					delete out;
				
				}
				else
				{
					for (int i = 0; i < Batch_size; i ++)
					for (int j = 0; j < Batch_size; j++)
		            { 
						if(out[i * Batch_size + j]>(*r_thresh-ep)&&out[i * Batch_size + j]<=(1+ep))
					  {

						  out_t[j * Batch_size + i] = out[i * Batch_size + j]; 

						  nonzerocount++;
					  }
						else
						{			
							out[i * Batch_size + j] = 0; //out_t will be push to (jj,ii), out respond to (ii,jj)
						}
				    }
				}
				
				//2.generate local CSR  //do not forget to modify data survival	
				int count = 0;
				long long Cindex = 0;
				vector <ColumnValueInfo> LocalColumnAndValue;
				vector <int> LocalRow;
				LocalRow.resize(Batch_size+1);
				LocalColumnAndValue.resize(nonzerocount);
				LocalRow[0]=0;
				for (int i = 0; i < Batch_size; i ++)
				{
					for (int j = 0; j < Batch_size; j++)
		            {
						if(out_t[i*Batch_size+j])
						{
						count ++;
						LocalColumnAndValue[Cindex].column = j;
						LocalColumnAndValue[Cindex].value = out_t[i*Batch_size+j];
						Cindex++;
		            	}	
		            }
					LocalRow[i+1] = count;
					
				}
				delete out_t;
				
				//3.push local csr format into global csr storage.
				if(ii==jj)
				{
					//local data->column offset; prepare for column && value pushing
					for (vector <ColumnValueInfo>::iterator i = LocalColumnAndValue.begin(); i != LocalColumnAndValue.end(); i++)
					{
						(*i).column += ii*Batch_size;
					}
					for (int i = 0; i < Batch_size; i++)
					{
						//push value && column
						if (LocalRow[i+1]!=LocalRow[i])
						{
							
							ColumnAndValue[ii*Batch_size+i].insert(ColumnAndValue[ii*Batch_size+i].end(),LocalColumnAndValue.begin()+LocalRow[i],LocalColumnAndValue.begin()+LocalRow[i+1]);//really?
						}
						//push row offset
					    Row[ii*Batch_size+i] += LocalRow[i];   
					}
					    // Row[ii*Batch_size+Batch_size] += LocalRow[Batch_size];
						//longitudinal superposition 
					for (long i = ii*Batch_size+Batch_size; i <= Num_Blocks*Batch_size+1; i++)
					{
						Row[i] += LocalRow[Batch_size];
					}
				}
				else
				{   //ii.0 generate compressed format of out
					count = 0;
				    Cindex = 0;
				    vector <ColumnValueInfo> iLocalColumnAndValue;
				    vector <int> iLocalRow;
					iLocalRow.resize(Batch_size+1);
					iLocalColumnAndValue.resize(nonzerocount);
					iLocalRow[0]=0;
					for (int i = 0; i < Batch_size; i ++)
					{
						for (int j = 0; j < Batch_size; j++)
						{
							if(out[i*Batch_size+j])
							{
							count ++;
							iLocalColumnAndValue[Cindex].column = j;
							iLocalColumnAndValue[Cindex].value = out[i*Batch_size+j];
							Cindex++;
		            		}	
						}
						iLocalRow[i+1] = count;
					
					}
					delete out;
					//ii.1 column offset  supposed to be out!
					for (vector <ColumnValueInfo>::iterator i = iLocalColumnAndValue.begin(); i != iLocalColumnAndValue.end(); i++)
					{
						(*i).column += jj*Batch_size;
					}
					//ii.2 push
					for (int i = 0; i < Batch_size; i++)
					{
						if (iLocalRow[i+1]!=iLocalRow[i])
						{
						  	
							ColumnAndValue[ii*Batch_size+i].insert(ColumnAndValue[ii*Batch_size+i].end(),iLocalColumnAndValue.begin()+iLocalRow[i],iLocalColumnAndValue.begin()+iLocalRow[i+1]);
						}
						Row[ii*Batch_size+i] += iLocalRow[i];   
					}
					//Row[ii*Batch_size+Batch_size] += LocalRow[Batch_size];
					//ii.3 update R
					for (long i = ii*Batch_size+Batch_size; i <= Num_Blocks*Batch_size+1; i++)
					{
						Row[i] += iLocalRow[Batch_size];
					} 
					//ii.4 free storage
					vector<int>().swap(iLocalRow);
				    vector<ColumnValueInfo>().swap(iLocalColumnAndValue);  
					//jj.1 column offset  supposed to be out_t!
					for (vector <ColumnValueInfo>::iterator i = LocalColumnAndValue.begin(); i != LocalColumnAndValue.end(); i++)
					{
						//(*i).column -= ii*Batch_size;
						(*i).column += ii*Batch_size;
					}
					//jj.2 push
					for (int i = 0; i < Batch_size; i++)
					{
						if (LocalRow[i+1]!=LocalRow[i])
						{
							ColumnAndValue[jj*Batch_size+i].insert(ColumnAndValue[jj*Batch_size+i].end(),LocalColumnAndValue.begin()+LocalRow[i],LocalColumnAndValue.begin()+LocalRow[i+1]);
						}
						Row[jj*Batch_size+i] += LocalRow[i];   
					}
					//Row[jj*Batch_size+Batch_size] += LocalRow[Batch_size];
					//jj.3 update R
					for (long i = jj*Batch_size+Batch_size; i <= Num_Blocks*Batch_size+1; i++)
					{
						Row[i] += LocalRow[Batch_size];
					}
				}
				//4.free local storage
				vector<int>().swap(LocalRow);
				vector<ColumnValueInfo>().swap(LocalColumnAndValue); 	
			}
			cout<<"Fulfill the "<<ii+1<<"th disposition."<<endl;
		}
			//display and put out 
			time = clock() - time;
			cout<<"calculation time: "<<time<<"ms"<<endl;
			//time_t nowTime;
			unsigned int FreeMem = 0;
			MEMORYSTATUS MemStat;
			MemStat.dwLength = sizeof(MEMORYSTATUS);
			GlobalMemoryStatus(&MemStat);
			FreeMem = TOM(MemStat.dwAvailPhys);
			//std::time(&nowTime);
			//cout << ctime(&nowTime)<<endl;  //set time
			cout << "bytes of physical memory: " << TOM(MemStat.dwTotalPhys) <<"M" <<endl;
			cout << "percent of memory in use: " << MemStat.dwMemoryLoad <<"%" <<endl;
			cout << "free physical memory bytes: " << TOM(MemStat.dwAvailPhys) <<"M" <<endl;
			
			/*
			string a=string("E:\BNAlauncher\test_new\paper").append("\\").append("weighted");
			char tmp[10];
			string OutCor = a.substr(0, a.find_last_of('.')).append("_").append("cor").append(string(itoa(*r_thresh,tmp,10))).append("_").append("2mm").append(".txt");
			*/
			cout<<"number of non-zero elements: "<<Row[N]<<endl;
			long long M1 = (N-1);
			M1 *= N;
			M1 /= 2;
			real__t spa = 100.0 * Row[N] / M1 / 2;
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
				//fout<<Row[i]<<endl;

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
			return TRUE;
}

__global__ void partition_kernel ( real__t x, real__t *A,  double *j_l, double *j_r,const long N)
{
	__shared__ int tmp_j_l[thread_num];
	__shared__ int tmp_j_r[thread_num];
	int tmp_l = 0;
	int tmp_r = 0;
	int offset = 0;
	const int threadid =blockIdx.x*blockDim.x + threadIdx.x;

	for(offset=threadid; offset<N; offset+=blockDim.x*gridDim.x)
	{		
		if(A[offset]>x)
			tmp_l++ ;
		else if(A[offset]>x)
			tmp_r++;		
	}
	tmp_j_l[threadIdx.x] = tmp_l;
	tmp_j_r[threadIdx.x] = tmp_r;

	syncthreads();

	for(offset=1; offset+threadIdx.x<thread_num; offset*=2)
	{
			if (threadIdx.x%(2*offset)==0)  tmp_j_l[threadIdx.x]+= tmp_j_l[threadIdx.x+offset] ;
			syncthreads();
	}	
	if(threadIdx.x==0)
		j_l[blockIdx.x]=(double) tmp_j_l[0];
	if(threadIdx.x==1)
		j_l[blockIdx.x]=(double) tmp_j_r[0];
}



/*int *partition_gpu()
{}*/
real__t select_GPU(real__t *Cormat, long long M1, long long k)
{
	long long offset;

	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	
	int index,i=0;
	//long long k_next = 0, M_next = 0;
	//int onethread_n = blocksize/block_num/thread_num;


	real__t *Cormat_block;
	cudaStat = cudaMalloc ((void**)&Cormat_block, sizeof(real__t) * blocksize) ;
	if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
			
	int blocksize_num = (M1-1+blocksize)/blocksize;
	double *j_l;
	cudaMalloc ((void**)&j_l, sizeof(double) * block_num*blocksize_num) ;
	double *j_r;
	cudaMalloc ((void**)&j_r, sizeof(double) * block_num*blocksize_num) ;
	double hj_l = 0;
	double hj_r = 0; 
	
	
	real__t left  = 0.0;
	real__t right = 1.0;
	real__t x;
	stat = cublasCreate(&handle) ;
	if (stat != CUBLAS_STATUS_SUCCESS)
		return stat;
	
	i=0;
	clock_t time = clock();
	
	x = (left+right)/2.0;
	for (offset = 0; offset < M1; offset += blocksize)
	{
		int size = (M1-offset > blocksize? blocksize : M1-offset);
		cudaMemcpy(Cormat_block, Cormat+offset, sizeof(real__t) * size, cudaMemcpyHostToDevice);
		partition_kernel<<<block_num,thread_num>>>(x, Cormat_block,  j_l+block_num*i, j_r+block_num*i,(long) size);
		stat = cublasDasum(handle, block_num*blocksize_num, j_l, 1, &hj_l);
		if (stat != CUBLAS_STATUS_SUCCESS)
			return stat;
		stat = cublasDasum(handle, block_num*blocksize_num, j_r, 1, &hj_r);
		if (stat != CUBLAS_STATUS_SUCCESS)
			return stat;
		i++;
	}
	
	//cudaMemcpy(hj_r, j_r, sizeof(int)*block_num*((M1-1)/blocksize), cudaMemcpyDeviceToHost);
	if(hj_l + hj_r > M1) cout<<"partition error! hj_l = "<<hj_l<<" ; hj_r = "<<hj_r<<endl;
	else if(hj_l < k && hj_r < M1-k) return x;
	else if ( hj_l < k)  right = x;	
	else left = x;
	
	time = clock()-time;
	cout<<"first partition time = "<<time<<";  hj_l = "<<hj_l<<endl;

		/*cout<<"round "<<offset/blocksize<<endl;
		
		cout<<"bound for partition "<<x<<endl; 
		for (int i = 0; i < block_num; i++)
		{
			cout<<i<<" block :"<<hj_l[i]<<"  "<<hj_r[i]<<"  "<<hj_l[i]+hj_r[i]<<endl;
			if (hj_l[i]+hj_r[i] != thread_num * onethread_n)
			{
				cout<< "partition error! " <<i<<" block;  "<<hj_l[i]<<"  "<<hj_r[i]<<endl;
				system("pause");
			}
				int j = 0;
		}*/
				//	if (offset+i*onethread_n*thread_num>=M1)
		//		break;
		
	
	return (0);
	
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





long long find_max(real__t *Cormat, long long M1)
{
	long long offset;

	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
	long long blocksize = 1024*1024*48*8;
	int index,i=0;

	real__t *Cormat_block;
	cudaStat = cudaMalloc ((void**)&Cormat_block, sizeof(real__t) * blocksize) ;
	if (cudaStat != CUBLAS_STATUS_SUCCESS) 
			return cudaStat;
	
	stat = cublasCreate(&handle) ;
		if (stat != CUBLAS_STATUS_SUCCESS)
			return stat;

	int segnum = (M1+blocksize)/blocksize;
	real__t *tmp = new real__t [segnum];
	
	long long *tmp_index = new long long [segnum] ;
	

	for (offset = 0; offset + blocksize < M1; offset += blocksize)
	{	
		cublasSetVector(blocksize, sizeof(real__t), Cormat+offset, 1, Cormat_block, 1);
		stat = cublasIsamax(handle, blocksize, Cormat_block, 1, &index);
		tmp[i] = *(Cormat+offset+index-1);
		//cout << tmp[i]<<endl;
		tmp_index[i++] = index+offset-1;
		//cout<< tmp_index[i-1]<<endl;
	}
	cublasSetVector(M1-offset, sizeof(real__t), Cormat+offset, 1, Cormat_block, 1);
	stat = cublasIsamax(handle, M1-offset, Cormat_block, 1, &index);
	tmp[i] = *(Cormat+offset+index);
	tmp_index[i] = index+offset;
	//cout << tmp[i]<<endl;
	//cout<< tmp_index[i]<<endl;

	//real__t *tmp_gpu;
	//cudaStat = cudaMalloc ((void**)&tmp_gpu, sizeof(real__t) * segnum) ;
	//if (cudaStat != CUBLAS_STATUS_SUCCESS) 
	//		return cudaStat;
	
	//cublasSetVector(segnum, sizeof(real__t), tmp, 1, tmp_gpu, 1);
	//stat = cublasIsamax(handle, segnum, tmp_gpu, 1, &index);
	real__t max_r = tmp[0];
	for (i = 1; i<segnum; i++)
		if (tmp[i] > max_r)
		{	max_r = tmp[i]; index = i;  }  

	return tmp_index[index];
}


	