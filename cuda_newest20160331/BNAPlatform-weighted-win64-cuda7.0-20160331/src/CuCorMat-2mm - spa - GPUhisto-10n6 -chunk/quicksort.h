#ifndef _QUICKSORT_H
#define _QUICKSORT_H


#include<stdlib.h>
#include<iterator>
#include <algorithm>    // std::stable_partition
using namespace std;
int randomNumber(int p, int q) 
{
	return p+(int)((double)(q-p)*rand()/(RAND_MAX)); 
}
template <typename Iterator, typename Comparator>
Iterator randomizedPartition(Iterator p, Iterator r, Comparator comp)
{
	int n = distance(p,r);
	int i = randomNumber(0,n-1);
	Iterator tmp_p = p, tmp_r = r; 
	advance(tmp_p,i);
	iter_swap(tmp_p,--tmp_r);
	return stable_partition(p,r,bind2nd(comp,*tmp_r));
}
template<typename Iterator, typename Comparator>
void quickSort(Iterator p, Iterator r, Comparator comp)
{
	long n = distance(p,r);
	if (n>1)
	{
		Iterator q = randomizedPartition(p,r,comp);
		quickSort(p,q, comp);
		quickSort(q,r,comp);
	}
}


#endif   /* _QUICKSORT_H */