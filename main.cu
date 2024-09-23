#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "segmerge.h"

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
  std::cout << "CUDA error (" << _s << "): " \
            << cudaGetErrorString(_e) \
            << std::endl; \
  return 0; }


int main (int argc, char* argv[]) {

  using K = int;
  using T = int;

  if (argc != 2) {
    std::cerr << "usage: ./run n\n";
    std::exit(1);
  }

  // params
  int n = std::atoi(argv[1]); // size of array
  int largest_key = 20;
  int max_seg_size = 10;
  int min_seg_size = 0;
  
  // info
  std::cout << "Merging two arrays of " << n << " keys and vals\n";
  std::cout << "Largest key       : " << largest_key << std::endl;
  std::cout << "Smallest key      : " << 0 << std::endl;
  std::cout << "Largest value     : " << 1 << std::endl;
  std::cout << "Smallest value    : " << 1 << std::endl;
  std::cout << "Largest seg size  : " << max_seg_size << std::endl;
  std::cout << "Smallest seg size : " << min_seg_size << std::endl; 


  // seed
  srand(static_cast<unsigned>(time(0)));

  // create arrays
  std::vector<K> key_a(n);
  std::vector<K> key_b(n);
  std::vector<K> key_c;
  std::vector<T> val_a(n);
  std::vector<T> val_b(n);
  std::vector<T> val_c;
  
  for (std::size_t i = 0; i < n; i++) {
    key_a[i] = std::rand() % largest_key;
    key_b[i] = std::rand() % largest_key;
  }
  for (std::size_t i = 0; i < n; i++) {
    val_a[i] = 1;
    val_b[i] = 1;
  }

  std::vector<int> seg;
  std::vector<int> seg_c;
  int start = 0;
  while (start < n) {
    seg.emplace_back(start);
    //int sz = std::rand() % (max_seg_size - min_seg_size + 1) + min_seg_size;
    int sz = std::rand() % max_seg_size + min_seg_size;
    start = seg.back() + sz;
  }
  int m = seg.size();
  seg_c.emplace_back(0);
  std::cout << "Number of segments: " << m+1 << std::endl;

  // allocate gpu memory
  cudaError_t err;
  K* key_a_d;
  K* key_b_d;
  K* key_c_d;
  T* val_a_d;
  T* val_b_d;
  T* val_c_d;
  int* seg_d;
  int* seg_c_d;
  err = cudaMalloc(&key_a_d, sizeof(K)*n);
  CUDA_CHECK(err, "alloc key_a_d");
  err = cudaMalloc(&key_b_d, sizeof(K)*n);
  CUDA_CHECK(err, "alloc key_b_d");
  err = cudaMalloc(&key_c_d, sizeof(K)*2*n);
  CUDA_CHECK(err, "alloc key_c_d");
  err = cudaMalloc(&val_a_d, sizeof(T)*n);
  CUDA_CHECK(err, "alloc val_a_d");
  err = cudaMalloc(&val_b_d, sizeof(T)*n);
  CUDA_CHECK(err, "alloc val_b_d");
  err = cudaMalloc(&val_c_d, sizeof(T)*2*n);
  CUDA_CHECK(err, "alloc val_c_d");
  err = cudaMalloc(&seg_d, sizeof(int)*m);
  CUDA_CHECK(err, "alloc seg_d");
  err = cudaMalloc(&seg_c_d, sizeof(int)*m);
  CUDA_CHECK(err, "alloc seg_d_d");
    
  // copy h2d
  err = cudaMemcpy(key_a_d, &key_a[0], sizeof(K)*n, cudaMemcpyDefault);
  CUDA_CHECK(err, "copy to key_a_d"); 
  err = cudaMemcpy(key_b_d, &key_b[0], sizeof(K)*n, cudaMemcpyDefault);
  CUDA_CHECK(err, "copy to key_b_d"); 
  err = cudaMemcpy(val_a_d, &val_a[0], sizeof(T)*n, cudaMemcpyDefault);
  CUDA_CHECK(err, "copy to val_a_d"); 
  err = cudaMemcpy(val_b_d, &val_b[0], sizeof(T)*n, cudaMemcpyDefault);
  CUDA_CHECK(err, "copy to val_b_d"); 
  err = cudaMemcpy(seg_d, &seg[0], sizeof(int)*m, cudaMemcpyDefault);
  CUDA_CHECK(err, "copy to seg_d"); 

  auto begin = std::chrono::steady_clock::now();
  int new_n = segmerge(
    key_a_d, key_b_d, key_c_d,
    val_a_d, val_b_d, val_c_d,
    seg_d,   seg_c_d, n, m
  );
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  std::cout << "CUDA runtime (us) : " <<
    std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
    << std::endl;

  // copy d2h
  std::vector<int> seg_c_h(m);
  std::vector<K> key_c_h(new_n); 
  std::vector<T> val_c_h(new_n); 
  err = cudaMemcpy(&seg_c_h[0], seg_c_d, sizeof(int)*m, cudaMemcpyDefault);
  CUDA_CHECK(err, "copy from seg_c_d");
  err = cudaMemcpy(&key_c_h[0], key_c_d, sizeof(K)*new_n, cudaMemcpyDefault); 
  CUDA_CHECK(err, "copy from key_c_d");
  err = cudaMemcpy(&val_c_h[0], val_c_d, sizeof(T)*new_n, cudaMemcpyDefault);
  CUDA_CHECK(err, "copy from val_c_d");

  // free
  err = cudaFree(key_a_d);
  CUDA_CHECK(err, "free key_a_d");
  err = cudaFree(key_b_d);
  CUDA_CHECK(err, "free key_b_d");
  err = cudaFree(key_c_d);
  CUDA_CHECK(err, "free key_c_d");
  err = cudaFree(val_a_d);
  CUDA_CHECK(err, "free val_a_d");
  err = cudaFree(val_b_d);
  CUDA_CHECK(err, "free val_b_d"); 
  err = cudaFree(val_c_d);
  CUDA_CHECK(err, "free val_c_d");
  err = cudaFree(seg_d);
  CUDA_CHECK(err, "free seg_d");

  // cpu
  begin = std::chrono::steady_clock::now();
  gold_segsort(key_a, val_a, n, seg, m);
  gold_segsort(key_b, val_b, n, seg, m);
  gold_segmerge(key_a, key_b, key_c,
                val_a, val_b, val_c,
                n, seg, m, seg_c);
  end = std::chrono::steady_clock::now();
  std::cout << "CPU runtime (us)  : " << 
    std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
    << std::endl;
  
  
  // check
  int cnt = 0;
  for (std::size_t i = 0; i < m; i++)
    if (seg_c[i] != seg_c_h[i]) cnt++;
  if (cnt != 0) 
    std::cout << "[NOT PASSED] checking segs: #err = " <<  cnt << std::endl;
  else 
    std::cout << "[PASSED] checking segs\n";
  cnt = 0;
  for (std::size_t i = 0; i < new_n; i++)
    if (key_c[i] != key_c_h[i]) cnt++;
  if (cnt != 0)  
    std::cout << "[NOT PASSED] checking keys: #err = " <<  cnt << std::endl;
  else
    std::cout << "[PASSED] checking keys\n";
  for (std::size_t i = 0; i < new_n; i++)
    if (val_c[i] != val_c_h[i]) cnt++;
  if (cnt != 0)  
    std::cout << "[NOT PASSED] checking vals: #err = " <<  cnt << std::endl;
  else
    std::cout << "[PASSED] checking vals\n";

  // print
  //std::cout << "key_c:\n"; 
  //print(seg_c, key_c);
  //std::cout << "val_c:\n"; 
  //print(seg_c, val_c);
  //std::cout << "key_c_h:\n";
  //print(seg_c_h, key_c_h);
  //std::cout << "val_c_h:\n"; 
  //print(seg_c_h, val_c_h);

  return 0;
}


