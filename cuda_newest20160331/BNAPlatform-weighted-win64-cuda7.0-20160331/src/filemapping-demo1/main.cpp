#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "windows.h"
#include <iostream>
using namespace std;
LPVOID FileMapping(string PathAndName, DWORD SizeHigh)
{
		HANDLE Hfile = CreateFile(PathAndName.c_str(),                         //file name
								GENERIC_READ|GENERIC_WRITE,                    //access mode      
								0,                                             //share mode
								NULL,                                          //security attribute
								OPEN_ALWAYS,                                   //disposition
								FILE_FLAG_DELETE_ON_CLOSE,                     //other flags and attributes
								NULL);                                         //whether to use template
		if (Hfile == INVALID_HANDLE_VALUE)
		{
			cout<<"File couldn't be created. Error code: "<<GetLastError()<<endl;
			return FALSE;
		}
		string mappingname =  PathAndName.substr(PathAndName.find_last_of("\\"),PathAndName.find_last_of(".")-PathAndName.find_last_of("."));
		HANDLE HfileMap = CreateFileMapping(Hfile,                               //file name  //INVALID_HANDLE_VALUE is both permitive?
											NULL,                                //security attributes
											PAGE_READWRITE,                      //access mode
											SizeHigh,                            //high 32 bit of maxlength for file mapping
											0xFFFFFFFF,                          //low 32............
											mappingname.c_str());                        //name of file-mapping objects 
  
		  if (HfileMap == NULL)
		  {
			  cout<<"File-mapping object couldn't be created. Error code: "<<GetLastError()<<endl;
			  CloseHandle(Hfile);
			  return FALSE;
		  }                                                     
		  else return MapViewOfFile(HfileMap,FILE_MAP_ALL_ACCESS,0,0,0);  //handle of object
		                                                                  //
 
}
		
void main()
{

}