#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "memory.h"
#include <iostream>
#include <ctime>
# include <fstream>
#include <vector>
#include <Windows.h>
#include<iomanip>
//#include<algorithm>

using namespace std;

#define ep  1e-6  //third question

#pragma comment(lib,"cublas.lib")
typedef float real__t;
typedef unsigned int uint__t;

#define TOM(byteValue) (byteValue/1024/1024)
void select(vector<real__t>::iterator A,long long n,long long k);
//bool myfunction (real__t i,real__t j) ; 
//#define CPUCormat 0

bool IsNumber(double x)
{
	return (x == x);
}
bool IsFiniteNumber(double x)
{
	return (x <= DBL_MAX && x >= -DBL_MAX);
}

typedef struct cv
		{  
		 int column;
		 real__t value;
		} ColumnValueInfo;    //Global definition is necessary

const int thread_num = 256;
const int block_num = 48;
const int blocksize = 1024*1024*48;

void MatrixMultiplication(real__t * BOLD_t1, real__t * BOLD_t2,real__t * out,int Batch_size,int L);
void Thrust(vector <vector<ColumnValueInfo>>::iterator begin, real__t *out, int ii, int Batch_size, real__t r_thresh, real__t er);

int CorMat_gpu(string OutCor, real__t * BOLD, int N, int L, int Batch_size,real__t *r_thresh,real__t width, real__t *s_thresh, clock_t* aggregrate)
{
	real__t * BOLD_t1, * BOLD_t2, * tempout;
	const int Num_Blocks = (N + Batch_size - 1) / Batch_size;
	uint__t N0 = Num_Blocks * Batch_size;
	uint__t amount =	N * (*s_thresh) * (N-1)  / 100.0 / 2.0;
	amount += amount%2;

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
		cout<<"generating CSR-format matrix...  "<<endl;
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
		uint__t Num_Bins = 1.0 / width + 2; //take care! 
		vector<real__t>  bin; 
		ColumnValueInfo tmp;
		clock_t time;
		time = clock();
		for (int kk = 0, ii = 0; ii < Num_Blocks; ii++)
		{
			for (int jj = ii; jj < Num_Blocks; jj++)
			{
				  
				BOLD_t1 = BOLD_t + ii * Batch_size * L;
				BOLD_t2 = BOLD_t + jj * Batch_size * L;
				//real__t *v425 = new real__t[L];
				real__t *out = new real__t[Batch_size * Batch_size];
			//	out = Cormat + (long long) kk * Batch_size * Batch_size;
			//	kk++;

#ifdef CPUCormat
                MatrixMultiplication(BOLD_t1, BOLD_t2, out, Batch_size,L);
#else
				cudaMemset(devCormat,0,sizeof(real__t)*Batch_size*Batch_size);
				stat = cublasSgemm(handle, CUBLAS_OP_T,  CUBLAS_OP_N, Batch_size, Batch_size, L,  &alpha, devBOLD + jj * Batch_size * L, L, devBOLD + ii * Batch_size * L, L, &beta, devCormat, Batch_size);
				if (stat != CUBLAS_STATUS_SUCCESS)
					return stat;
				//time = clock();
				cudaStat = cudaMemcpy(out, devCormat, sizeof(real__t) * Batch_size * Batch_size, cudaMemcpyDeviceToHost);
				if (cudaStat != cudaSuccess) 
					return cudaStat;
#endif
				/*time = clock() - time;
				cout<<"memcpy time: "<<time<<"ms"<<endl;
				time = clock();*/
				//if (ii==jj)
				//{
				//	for (int i = 0; i < Batch_size; i++)
				//	{
				//		out[i*Batch_size+i] = 0;
				//	}
				//}
				//for (int i = 0; i < Batch_size * Batch_size; i++)
				//{
				//	if ((!IsNumber(out[i]))||(!IsFiniteNumber(out[i]))||out[i]<0)
				//	{
				//		out[i] = 0;
				//	}
				//	//cout<<BOLD_t[i]<<endl;
				//}  
				//1.screen
				/*time = clock();
				if(ii!=jj)
				{
					for (long i = 0; i<Batch_size*Batch_size;i++)
					{
						if (out[i]>1+ep)
						{
							cout<<"data error: "<<out[i]<<endl;
						}
						else if (out[i]>=(*r_thresh-ep)&&out[i]<(*r_thresh+width))
						{
							bin.push_back(out[i]);
						}					
					}
				}
				else
				{
					for (int i = 0; i < Batch_size; i++)
					{
						for (int j = i+1; j < Batch_size; j++)
						{
							if (out[i*Batch_size+j]>1+ep)
							{
							cout<<"data error: "<<out[i*Batch_size+j]<<endl;
							}
							else if (out[i*Batch_size+j]>=(*r_thresh-ep)&&out[i*Batch_size+j]<(*r_thresh+width))
							{
								bin.push_back(out[i*Batch_size+j]);
							}
						}
					}
				}	  
				time = clock() -time;
				cout<<"screening time: "<<time<<"ms"<<endl;*/
				/*time = clock();*/
				if(ii==jj)
				{
					//Thrust(ColumnAndValue.begin(), out, ii, Batch_size,  *r_thresh,  ep);
					for (int i = 0; i < Batch_size; i ++)
					{
						for (int j = 0; j < Batch_size; j++)
					    { 
							if(out[i * Batch_size + j]>(*r_thresh-ep)&&out[i * Batch_size + j]<=(1+ep)&&(i!=j))
							{
							    tmp.column = j;
								tmp.column += ii * Batch_size;
								tmp.value = out[i*Batch_size+j];
								ColumnAndValue[ii*Batch_size+i].push_back(tmp);
								if (out[i*Batch_size+j]<(*r_thresh+width)&&i<j)
								{
									bin.push_back(out[i*Batch_size+j]);
								}
							}
						}
					 }
				}
				else
				{
					//ThrustAsymmetrical(ColumnAndValue.begin(), out, ii, jj, Batch_size, *r_thresh, ep);
					for (int i = 0; i < Batch_size; i ++)
					{
						for (int j = 0; j < Batch_size; j++)
						{ 
							if( out[i * Batch_size + j]>(*r_thresh-ep) && out[i * Batch_size + j]<=(1+ep) )
							{
							    //1.push row
								tmp.column = j;
								tmp.column += jj * Batch_size;
								tmp.value = out[i*Batch_size+j];
								ColumnAndValue[ii*Batch_size+i].push_back(tmp);
								//2.push column
								tmp.column = i;
								tmp.column += ii * Batch_size;
								tmp.value = out[i*Batch_size+j];
								ColumnAndValue[jj*Batch_size+j].push_back(tmp);
								if (out[i * Batch_size + j]<(*r_thresh+width))
								{
									bin.push_back(out[i * Batch_size + j]);
								}
							}
						}
					}
				}
				/*time = clock() -time;
				cout<<"csr time: "<<time<<"ms"<<endl;*/
				delete []out;
				cout<<"Loop flag: "<<ii<<":"<<jj<<endl;
			}
			cout<<"Fulfill the "<<ii+1<<"th disposition."<<endl;
		}
		Row[0] = 0;
		for (vector <vector<ColumnValueInfo>>::iterator x = ColumnAndValue.begin(); x != ColumnAndValue.end(); x++)
		{
			Row[x-ColumnAndValue.begin() + 1] = (*x).size();
			Row[x-ColumnAndValue.begin() + 1] += Row[x-ColumnAndValue.begin()];
		}
		time = clock() - time;
		cout<<"correaltion time: "<<time<<"ms"<<endl;
		*aggregrate += time; 
		time = clock();
		cout<<"updating CSR-format matrix...  "<<endl;
		//6.gain position
		uint__t overallSize = 0;
		uint__t position = 0;
		/*for (vector <vector<ColumnValueInfo>>::iterator i = ColumnAndValue.begin(); i != ColumnAndValue.end(); i++)
		{
			overallSize += (*i).size();
		}*/
		position = Row[N]/2.0 - amount;
		position = bin.size() - position;
		select(bin.begin(),distance(bin.begin(),bin.end()), position);
		real__t finerResult = bin[position-1];	
		vector<real__t>().swap(bin);
		//7.update csr format
		for (vector <vector<ColumnValueInfo>>::iterator i = ColumnAndValue.begin(); i != ColumnAndValue.end(); i++)
		{
			for (vector<ColumnValueInfo>::iterator j = (*i).begin(); j !=(*i).end();)
			{
				real__t V = (*j).value;
				if (V<(finerResult-ep))
				{
					//location.push_back(j-(*i).begin());
					j=ColumnAndValue[i-ColumnAndValue.begin()].erase(j);
					for (uint__t x = i-ColumnAndValue.begin()+1; x < (N + 1); x++)
					{
						Row[x] -= 1;
					}
				}
				else
				{
					j++;
				}
			}	
		}
		cout<<"update is complete.  "<<endl;
		time = clock() - time;
		cout<<"update time: "<<time<<"ms"<<endl;
		*aggregrate += time; 
		cout<<"overall time for histogram plus correlation plus update: "<<*aggregrate<<"ms"<<endl;
		free(aggregrate);
	    //display and put out 
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




	