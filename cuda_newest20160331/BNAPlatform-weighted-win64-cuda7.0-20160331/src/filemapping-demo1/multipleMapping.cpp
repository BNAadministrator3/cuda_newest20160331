#include<Windows.h>
#include<string>
#include<iostream>
using namespace std;
void main()
{
 SYSTEM_INFO sinf;
 GetSystemInfo(&sinf);
 HANDLE hFile=CreateFile(TEXT("d://huge.txt"),GENERIC_WRITE|GENERIC_READ,0,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);//���ļ�
 DWORD dwFileSizeHigh;
 _int64 qwFileSize=GetFileSize(hFile,&dwFileSizeHigh);//����ļ���С
 qwFileSize+=(((_int64)dwFileSizeHigh)<<32);//����32λ�ӵ��ļ���32λ��
 HANDLE hFileMap=CreateFileMapping(hFile,NULL,PAGE_READWRITE,0,0,NULL);//����һ���ļ��ں˶��󣬴�СĬ��Ϊ�ļ���С
 CloseHandle(hFile);
 _int64 qwFileOffset=0;//ÿ��ӳ���ļ���С����ʼֵΪ0
 while(qwFileSize>0){
  DWORD dwBytesInBlock=sinf.dwAllocationGranularity;//Ԥ���ռ�ķ�������
  if(qwFileSize<sinf.dwAllocationGranularity)
   dwBytesInBlock=(DWORD)qwFileSize;
  PCHAR pbFile=(PCHAR)MapViewOfFile(hFileMap,
   FILE_MAP_WRITE,
   (DWORD)(qwFileOffset>>32),//��32λ
   (DWORD)(qwFileOffset&0xFFFFFFFF),//��32λ
   dwBytesInBlock);//Ϊ�ļ�������Ԥ��һ���ַ�ռ����򲢽��ļ���������Ϊ����洢������������
  cout<<"content: "<<pbFile<<endl;
  UnmapViewOfFile(pbFile);//�ӽ��̿ռ䳷�����ļ����ݵĹ���
  qwFileOffset+=dwBytesInBlock;
  qwFileSize-=dwBytesInBlock;
 }
 CloseHandle(hFileMap);
}
