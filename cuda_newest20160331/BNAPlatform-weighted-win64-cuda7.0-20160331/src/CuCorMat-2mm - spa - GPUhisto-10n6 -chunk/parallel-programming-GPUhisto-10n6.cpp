# include <windows.h>
# include <iostream>
# include <process.h>
# include <fstream>
# include <math.h>
# include <vector>
using namespace std;
// This struct is used for passing parameters to different threads
typedef float real__t;
typedef struct cv
		{  
		 int column;
		 real__t value;
		} ColumnValueInfo;  


int THREADNUM_2 = 4;

/************************diagnoal block********************************/ 
/************************first step********************************/ 
struct First
{
	
	int id;
    int ii;
	int Batch_size;
	vector <vector<ColumnValueInfo>>::iterator begin;
	real__t *out;
	real__t r_thresh;
	real__t ep;
};
void Push(void * voidarg)
{
	First *arg = (First *)voidarg;
	int id = arg->id;
	int ii = arg->ii;
	int  Batch_size = arg->Batch_size;
	vector <vector<ColumnValueInfo>>::iterator begin = arg->begin;
	real__t *out = arg->out;
	real__t r_thresh = arg->r_thresh;
	real__t ep = arg->ep;
	ColumnValueInfo tmp;
	for ( unsigned int row = id; row < Batch_size; row += THREADNUM_2)
	{
		for (unsigned int col = 0; col < Batch_size; col++)
		{
		if (out[row * Batch_size + col]>(r_thresh-ep)&&out[row * Batch_size + col]<=(1+ep)&&(row!=col))
			{
				tmp.column = col + ii * Batch_size;
				tmp.value = out[row * Batch_size + col];
				(*(begin + ii*Batch_size+row)).push_back(tmp);
			}
		

		}
	}
}
void Thrust(vector <vector<ColumnValueInfo>>::iterator begin, real__t *out, int ii, int Batch_size, real__t r_thresh, real__t er)
{
	SYSTEM_INFO siSysInfo;
	GetSystemInfo(&siSysInfo); 
	THREADNUM_2 = (int)siSysInfo.dwNumberOfProcessors; 
	//THREADNUM = sysconf(_SC_NPROCESSORS_CONF);	

	First * Properties_arg = new First [THREADNUM_2];	
	//double * Cpsum = new double[THREADNUM_2];
	for (int i = 0; i < THREADNUM_2; i++)
	{
		Properties_arg[i].id = i;
		Properties_arg[i].ii = ii;
		Properties_arg[i].Batch_size = Batch_size;
		Properties_arg[i].begin = begin;
		Properties_arg[i].out = out;
		Properties_arg[i].r_thresh = r_thresh;
		Properties_arg[i].ep = er;
	}
	//pthread_t *t = new pthread_t[THREADNUM];
	HANDLE *tHandle = new HANDLE[THREADNUM_2];
	for (int i = 0; i < THREADNUM_2; i++)
	{
		First *temp = Properties_arg + i;
		tHandle[i] = (HANDLE) _beginthread(Push, 0, (char *)temp);
	}
	for (int i = 0; i < THREADNUM_2; i++)
		WaitForSingleObject(tHandle[i], INFINITE);

	//cout<<"mean:"<<mean_Cp/N<<endl;
	delete []Properties_arg;
	delete []tHandle;

}
/************************second step********************************/ 
struct Second
{
	
	int id;
    int ii;
	int jj;
	int Batch_size;
	vector <vector<ColumnValueInfo>>::iterator begin;
	real__t *out;
	real__t r_thresh;
	real__t ep;
};
void PushAsymmetrical(void * voidarg)
{
	Second *arg = (Second *)voidarg;
	int id = arg->id;
	int ii = arg->ii;
	int jj = arg->jj;
	int  Batch_size = arg->Batch_size;
	vector <vector<ColumnValueInfo>>::iterator begin = arg->begin;
	real__t *out = arg->out;
	real__t r_thresh = arg->r_thresh;
	real__t ep = arg->ep;
	ColumnValueInfo tmp;
	for ( unsigned int row = id; row < Batch_size; row += THREADNUM_2)
	{
		for (unsigned int col = 0; col < Batch_size; col++)
		{
			if (out[row * Batch_size + col]>(r_thresh-ep)&&out[row * Batch_size + col]<=(1+ep))
			{
				//push row
				tmp.column = col + jj * Batch_size;
				tmp.value = out[row * Batch_size + col];
				(*(begin + ii*Batch_size+row)).push_back(tmp);
				//push column
				tmp.column = row + ii * Batch_size;
				tmp.value = out[row * Batch_size + col];
				(*(begin + jj*Batch_size+col)).push_back(tmp); //have trouble
			
			}
		}
	}
}

void ThrustAsymmetrical(vector <vector<ColumnValueInfo>>::iterator begin, real__t *out, int ii, int jj, int Batch_size, real__t r_thresh, real__t er)
{
	SYSTEM_INFO siSysInfo;
	GetSystemInfo(&siSysInfo); 
	THREADNUM_2 = (int)siSysInfo.dwNumberOfProcessors; 
	//THREADNUM = sysconf(_SC_NPROCESSORS_CONF);	

	Second * Properties_arg = new Second [THREADNUM_2];	
	//double * Cpsum = new double[THREADNUM_2];
	for (int i = 0; i < THREADNUM_2; i++)
	{
		Properties_arg[i].id = i;
		Properties_arg[i].ii = ii;
		Properties_arg[i].jj = jj;
		Properties_arg[i].Batch_size = Batch_size;
		Properties_arg[i].begin = begin;
		Properties_arg[i].out = out;
		Properties_arg[i].r_thresh = r_thresh;
		Properties_arg[i].ep = er;
	}
	//pthread_t *t = new pthread_t[THREADNUM];
	HANDLE *tHandle = new HANDLE[THREADNUM_2];
	for (int i = 0; i < THREADNUM_2; i++)
	{
		Second *temp = Properties_arg + i;
		tHandle[i] = (HANDLE) _beginthread(PushAsymmetrical, 0, (char *)temp);
	}
	for (int i = 0; i < THREADNUM_2; i++)
		WaitForSingleObject(tHandle[i], INFINITE);

	//cout<<"mean:"<<mean_Cp/N<<endl;
	delete []Properties_arg;
	delete []tHandle;

}
