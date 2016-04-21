#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <fstream>
#include <iostream>
#include <ctime>
#include "dirent.h"
#include "BC_CPU.h"

using namespace std;

void read_csr(string a, int &numVertices, int &numEdges, int *&row, int *&col)
{
	cout<<"\ncalculating Betweenness for "<<a.c_str()<<" ..."<<endl;
	string csrfilename = a;
	ifstream fCSR(csrfilename.c_str(), ios::binary);
	if(!fCSR.good())
	{
		cout<<"cannot open .csr file!"<<endl<<"note: please add \".csr\" at the end of the file name "<<endl;
		exit(1);
	}
	fCSR.read((char *)&numVertices, sizeof(int));
	numVertices = numVertices - 1;	
	row = new int[numVertices+1];	// Cormat CSR row
	fCSR.read((char*)row, sizeof(int)*(numVertices + 1));
	fCSR.read((char *)&numEdges, sizeof(int));
	col = new int[numEdges];	// Cormat CSR col
	fCSR.read((char*)col, sizeof(int)*numEdges);
	fCSR.close();
	return;
}

string* file_detecton(int argc, char** argv,int* filenumber)
{
if (argc < 2) 
	{
		cerr<<"Input format: .\\CUBC.exe dir_for_csr \nFor example: .\\CUBC.exe d:\\data "<<endl;
		exit(1);	
	}

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

		if (filenametmp.find_last_of('.') == -1)
			continue;
		if(filenametmp.length()>4 && filenametmp.substr(filenametmp.find_last_of('.'),4).compare(".csr") == 0 && filenametmp.size() - filenametmp.find_last_of('.') - 1 == 3)
		{
			FileNumber++;
		}
	}
	cout<<FileNumber<<" files to be processed."<<endl;

	closedir(dp);
	string *filename = new string[FileNumber];
	dp = opendir(argv[1]);
	int i = 0;
	while((dirp = readdir(dp)) != NULL)
	{
		filenametmp = string(dirp->d_name);
		if (filenametmp.find_last_of('.') == -1)
			continue;
		if(filenametmp.length()>4 && filenametmp.substr(filenametmp.find_last_of('.'),4).compare(".csr") == 0 && filenametmp.size() - filenametmp.find_last_of('.') - 1 == 3)
		{
			filename[i++] = filenametmp;
		}
	}
	*filenumber = FileNumber;
	return filename;

}

int main( int argc, char** argv) 
{
	// input 1: file name

	// read input .csr file 
	int numVertices;	// Network vertices number
	int numEdges;	//Network egdes number
	int *row = NULL;	// Cormat CSR row
	int *col = NULL;	// Cormat CSR col
	int *Filenumber = (int*)malloc(sizeof(int));
	string *filename = file_detecton(argc,argv, Filenumber);
	for (int i = 0; i < *Filenumber; i++)
	{
		string a = string(argv[1]).append("\\").append(filename[i]);
		read_csr(a.c_str(), numVertices, numEdges, row, col);

		float *BC_GOLDEN = new float[numVertices];

		// computing Betweenness
		cout<<"Start BC computing on CPU. n = "<<numVertices<<", m = "<<numEdges<<endl;	
		clock_t cpu_time = clock();
		Betweenness_CPU(row, col, numVertices, numEdges, BC_GOLDEN);
		cpu_time = clock() - cpu_time;
		cout<<cpu_time<<" ms for CPU BC"<<endl;
		cout<<(double)numEdges*numVertices*1000/cpu_time/1024/1024<<" MTEPS"<<endl;
	
		// write to file
		string X_bc = a.substr(0, a.find_last_of('.') ).append("_BC_gloden.txt");
		ofstream fout_golden(X_bc.c_str());
		for (int i = 0; i < numVertices; i++)
			fout_golden<<BC_GOLDEN[i]<<endl;
		fout_golden.close();
	
		// record running time
		ofstream cputimelog("cputimelog.txt", ios::app);
		cputimelog<<cpu_time<<" ms."<<endl;
		cputimelog.close();
	
	}
	
	//delete []BC_GOLDEN;
	delete []row;
	delete []col;
}