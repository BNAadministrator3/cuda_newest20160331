#include<Windows.h>
#include<string>
#include<iostream>
using namespace std;
void main()
{
 SYSTEM_INFO sinf;
 GetSystemInfo(&sinf);
 HANDLE hFile=CreateFile(TEXT("d://huge.txt"),GENERIC_WRITE|GENERIC_READ,0,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);//打开文件
 DWORD dwFileSizeHigh;
 _int64 qwFileSize=GetFileSize(hFile,&dwFileSizeHigh);//获得文件大小
 qwFileSize+=(((_int64)dwFileSizeHigh)<<32);//将高32位加到文件低32位上
 HANDLE hFileMap=CreateFileMapping(hFile,NULL,PAGE_READWRITE,0,0,NULL);//创建一个文件内核对象，大小默认为文件大小
 CloseHandle(hFile);
 _int64 qwFileOffset=0;//每次映射文件大小，初始值为0
 while(qwFileSize>0){
  DWORD dwBytesInBlock=sinf.dwAllocationGranularity;//预定空间的分配粒度
  if(qwFileSize<sinf.dwAllocationGranularity)
   dwBytesInBlock=(DWORD)qwFileSize;
  PCHAR pbFile=(PCHAR)MapViewOfFile(hFileMap,
   FILE_MAP_WRITE,
   (DWORD)(qwFileOffset>>32),//高32位
   (DWORD)(qwFileOffset&0xFFFFFFFF),//低32位
   dwBytesInBlock);//为文件的数据预定一块地址空间区域并将文件的数据作为物理存储器调拨给区域
  cout<<"content: "<<pbFile<<endl;
  UnmapViewOfFile(pbFile);//从进程空间撤销对文件数据的关联
  qwFileOffset+=dwBytesInBlock;
  qwFileSize-=dwBytesInBlock;
 }
 CloseHandle(hFileMap);
}
