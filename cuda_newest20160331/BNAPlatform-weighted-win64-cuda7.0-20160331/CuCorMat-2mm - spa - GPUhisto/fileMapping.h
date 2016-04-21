#ifndef FILEMAPPING_H
#define FILEMAPPING_H

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
LPVOID FileMapping(string PathAndName, DWORD SizeLow)
{
		HANDLE Hfile = CreateFile(PathAndName.c_str(),                         //file name
								GENERIC_READ,                    //access mode      
								0,                                             //share mode
								NULL,                                          //security attribute
								OPEN_EXISTING,                                   //disposition
								FILE_ATTRIBUTE_READONLY,                     //other flags and attributes
								NULL);                                         //whether to use template
		if (Hfile == INVALID_HANDLE_VALUE)
		{
			cout<<"File couldn't be created. Error code: "<<GetLastError()<<endl;
			return FALSE;
		}
		//string mappingname =  PathAndName.substr(PathAndName.find_last_of("\\"),PathAndName.find_last_of(".")-PathAndName.find_last_of("."));
		//DWORD dwFileSize = GetFileSize(Hfile, NULL);
		HANDLE HfileMap = CreateFileMapping(Hfile,                               //file name  //INVALID_HANDLE_VALUE is both permitive?
											NULL,                                //security attributes
											PAGE_READONLY,                      //access mode
											0,                            //high 32 bit of maxlength for file mapping
											0,                          //low 32............
											NULL);                        //name of file-mapping objects 
  
		  if (HfileMap == NULL)
		  {
			  cout<<"File-mapping object couldn't be created. Error code: "<<GetLastError()<<endl;
			  CloseHandle(Hfile);
			  return FALSE;
		  }                                                     
		  else return MapViewOfFile(HfileMap,PAGE_READONLY,0,0,0);  //handle of object
		                                                                 
}

#endif