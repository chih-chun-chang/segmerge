#include <cuda.h>

template<typename K, typename T>
__global__ void filln(
  K* key_a, K* key_b, K* key_c,
  T* val_a, T* val_b, T* val_c,
  int* seg, int* seg_c, 
  int n, int m)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < m) {
    int beg = seg[tid];
    int end = (tid + 1) < m ? seg[tid+1] : n;
    int sz = end - beg;
    std::size_t i = beg;
    std::size_t j = beg*2;
    std::size_t k = beg*2+sz;
    while (i < end && j < end*2 && k < end*2) {
      key_c[j] = key_a[i];
      val_c[j] = val_a[i];
      key_c[k] = key_b[i];
      val_c[k] = val_b[i];
      i++; j++; k++;
    }
    seg_c[tid] = beg * 2; 
  }
}

template<typename K, typename T>
__global__ void merge(
  K* key, T* val, int* seg, int* count, int n, int m)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < m) {
    int beg = seg[tid]; 
    int end = (tid + 1) < m ? seg[tid+1] : n;
    int du = 0;
    K currentKey = 0;
    T currentSum = 0;
    std::size_t i = beg;
    std::size_t j = i + 1;
    while (i < end && j < end) { 
      currentKey = key[i];
      currentSum = val[i];
      while (key[j] == currentKey && j < end) {
        currentSum += val[j];
        key[j] = -1;
        val[j] = -1;
        j++;
        du++;
      }
      val[i] = currentSum;
      i = j;
      j++;
    }
    count[tid] = du; 
  }
}

__global__ void sub(int* seg, int* count, int m)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < m) {
    seg[tid] -= count[tid];
  }
}
