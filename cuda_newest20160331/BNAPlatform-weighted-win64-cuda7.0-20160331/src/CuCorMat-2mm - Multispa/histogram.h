#ifndef HISTOGRAM_H
#define HISTOGRAM_H
/*
template <typename Ptr1, 
          typename Vector2,
          typename Vector3>
void dense_histogram(const Ptr1& input,int Number, float width,Vector2& temphisto,
                           Vector3& histogram);
template <typename Ptr1, 
          typename Vector2,
          typename Vector3>
void dense_histogram2(const Ptr1& input,int Number, float width,Vector2& temphisto,
                           Vector3& histogram);*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <iomanip>
#include <iterator>
// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}
//slow version x = scalar * x
template <typename Vector>
void sax_slow(float A, Vector& X) 
{   
	typedef typename Vector::value_type IndexType;
	thrust::device_vector<IndexType> temp(X.size()); 
	// temp <- A 
	thrust::fill(temp.begin(), temp.end(), A); 
//	print_vector("temp",temp);
	// temp <- A * X 
	thrust::transform(temp.begin(), temp.end(), X.begin(), X.begin(), thrust::multiplies<float>()); 
//	print_vector("x",	X);
}
template <typename Vector>
void substract(Vector& X, unsigned int a)
{
	typedef typename Vector::value_type IndexType;
	thrust::device_vector<IndexType> temp(X.size()); 
	thrust::fill(temp.begin(), temp.end(), a); 
//	print_vector("temp",temp);
//	print_vector("prehistogram",X);
	thrust::transform(X.begin(), X.end(),temp.begin(), temp.begin(),  thrust::minus<IndexType>());
//	print_vector("tempafter",temp);
	thrust::copy(temp.begin(), temp.end(), X.begin());
   //print_vector("histogramafter",X);
	//thrust::adjacent_difference(X.begin(), X.end(), X.begin());
	//print_vector("difference_histogram",X);
}
// dense histogram using binary search
template <typename Ptr1, 
          typename Vector2>
void dense_histogram(const Ptr1& input,unsigned long Number, float width, Vector2& temphisto,
                           Vector2& histogram)
{
  typedef typename Ptr1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type
  thrust::device_vector<ValueType> data(input,input+Number);
  //print_vector("data",data);
  thrust::sort(data.begin(), data.end());
  
  unsigned int num_bins = (int)(1.0 / width) + 2;
  thrust::counting_iterator<float> search_begin(1);
  thrust::device_vector<float> absciss(search_begin,search_begin + num_bins);
  sax_slow(width, absciss);
  thrust::upper_bound(data.begin(), data.end(),
	                  absciss.begin(), absciss.end(),
                      temphisto.begin());
  //print_vector("temphisto",temphisto);
  thrust::transform(temphisto.begin(), temphisto.end(), histogram.begin(), histogram.begin(), thrust::plus<IndexType>());
   //thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());
  //print_vector("histogram",histogram);
  }
// dense histogram using binary search and result * 2
template <typename Ptr1, 
          typename Vector2>
void dense_histogram2(const Ptr1& input,unsigned int Number, float width, Vector2& temphisto,
                           Vector2& histogram)
{
  typedef typename Ptr1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type
  unsigned int num_bins = (int)(1.0 / width) + 2;
  thrust::device_vector<ValueType> data(input,input+Number);
  //1.sort
  thrust::sort(data.begin(), data.end());
  //2.generate intervals
  thrust::counting_iterator<ValueType> search_begin(1);
  thrust::device_vector<ValueType> absciss(search_begin,search_begin + num_bins);
  sax_slow(width, absciss);
//  for(int i = 0; i < absciss.size(); i++) std::cout << "absciss[" << i << "] = " << absciss[i] << std::endl;


  //3.obtain cumulative temporary histogram
  thrust::upper_bound(data.begin(), data.end(), absciss.begin(), absciss.end(),temphisto.begin()); 
 // thrust::copy(data.begin()+8473160, data.begin()+8473168, std::ostream_iterator<float>(std::cout, " "));
  //thrust::adjacent_difference(temphisto.begin(), temphisto.end(), temphisto.begin());
  //print_vector("temphisto",temphisto);
  //4.result * 2
//  float multiple = 2;
  sax_slow(2, temphisto);
  //print_vector("temphistoM2",temphisto);
  //5.plus previous histogram
  //print_vector("temphisto",temphisto);
  thrust::transform(temphisto.begin(), temphisto.end(), histogram.begin(), histogram.begin(), thrust::plus<IndexType>());
  thrust::adjacent_difference(temphisto.begin(), temphisto.end(), temphisto.begin()); 
  //print_vector("difference_temphisto",temphisto);
  //print_vector("2histogram",histogram);

}

#endif