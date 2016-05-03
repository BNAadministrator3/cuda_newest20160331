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
void sax_slow(float A, thrust::device_vector<float>& X) 
{ 
	thrust::device_vector<float> temp(X.size()); 
	// temp <- A 
	thrust::fill(temp.begin(), temp.end(), A); 
	print_vector("temp",temp);
	// temp <- A * X 
	thrust::transform(temp.begin(), temp.end(), X.begin(), X.begin(), thrust::multiplies<float>()); 
	print_vector("x",	X);
}

// dense histogram using binary search
template <typename Ptr1, 
          typename Vector2,
          typename Vector3>
void dense_histogram(const Ptr1& input,int Number, float width, Vector2& temphisto,
                           Vector3& histogram)
{
  typedef typename Ptr1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  thrust::device_vector<ValueType> data(input,input+Number);
  thrust::sort(data.begin(), data.end());
  
  unsigned int num_bins = (int)(1.0 / width) + 2;
  thrust::counting_iterator<float> search_begin(1);
  thrust::device_vector<float> absciss(search_begin,search_begin + num_bins);
  sax_slow(width, absciss);
  thrust::upper_bound(data.begin(), data.end(),
	                  absciss.begin(), absciss.end(),
                      temphisto.begin());
  thrust::transform(temphisto.begin(), temphisto.end(), histogram.begin(), histogram.begin(), thrust::plus<float>());
  }
// dense histogram using binary search and result * 2
template <typename Ptr1, 
          typename Vector2,
          typename Vector3>
void dense_histogram2(const Ptr1& input,int Number, float width, Vector2& temphisto,
                           Vector3& histogram)
{
  typedef typename Ptr1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type
  unsigned int num_bins = (int)(1.0 / width) + 2;
  thrust::device_vector<ValueType> data(input,input+Number);
  //1.sort
  thrust::sort(data.begin(), data.end());
  //2.generate intervals
  thrust::counting_iterator<float> search_begin(1);
  thrust::device_vector<float> absciss(search_begin,search_begin + num_bins);
  sax_slow(width, absciss);
  //3.obtain cumulative temporary histogram
  thrust::upper_bound(data.begin(), data.end(), absciss.begin(), absciss.end(),temphisto.begin()); 
  //4.result * 2
  sax_slow(2, temphisto);
  //5.plus previous histogram
  thrust::transform(temphisto.begin(), temphisto.end(), histogram.begin(), histogram.begin(), thrust::plus<float>());
}
/*
int main(void)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 9);

  const int N = 40;
  const int S = 4;

  // generate random data on the host
  thrust::host_vector<float> input(N);
  for(int i = 0; i < N; i++)
  {
    int sum = 0;
    for (int j = 0; j < S; j++)
      sum += dist(rng);
    input[i] = sum / S / 9.0;
  }

  // demonstrate dense histogram method
  {
    std::cout << "Dense Histogram" << std::endl;
    
	thrust::device_vector<int> temphistogram;
	float interval = 0.1 ;
	unsigned int num_bins = (int)(1.0 / interval) + 2;

   // resize histogram storage
	thrust::device_vector<int> histogram(num_bins,100);
	temphistogram.resize(num_bins);
	
    dense_histogram(input,interval,temphistogram,histogram);
  
  }
  */
  // demonstrate sparse histogram method
  /*{
    std::cout << "Sparse Histogram" << std::endl;
    thrust::device_vector<int> histogram_values;
    thrust::device_vector<int> histogram_counts;
    sparse_histogram(input, histogram_values, histogram_counts);
  }*/

  // Note: 
  // A dense histogram can be converted to a sparse histogram
  // using stream compaction (i.e. thrust::copy_if).
  // A sparse histogram can be expanded into a dense histogram
  // by initializing the dense histogram to zero (with thrust::fill)
  // and then scattering the histogram counts (with thrust::scatter).
/*
  return 0;
}*/
