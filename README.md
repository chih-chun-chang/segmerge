# segmerge
This project aims to design an algorithm for performing parallel segmented merge operations on a GPU using CUDA. Segmented merge operations are applied in various algorithms, such as the ESC algorithm for sparse matrix multiplication and graph structure-related algorithms.

**Input**
```
segs: [0, 5, 8]
A keys: [1, 0, 4, 4, 2] [5, 1, 3] [2, 2, 5]
A vals: [1, 1, 1, 1, 1] [1, 1, 1] [1, 1, 1]
B keys: [4, 4, 0, 2, 1] [3, 8, 1] [3, 2, 4]
B vals: [1, 1, 1, 1, 1] [1, 1, 1] [1, 1, 1]
```

**Output**
```
C segs: [0, 4, 9]
C keys: [0, 1, 2, 4] [1, 3, 5, 8] [2, 3, 4, 5]
C vals: [2, 2, 2, 4] [2, 2, 1, 1] [3, 1, 1, 1]
```

## Compile & Run
In this project, we use [bb_segsort](https://github.com/vtsynergy/bb_segsort/tree/master) for segmented sorting.
```

```

## Expected Output
```
Merging two arrays of 131072 keys and vals
Largest key       : 20
Smallest key      : 0
Largest value     : 1
Smallest value    : 1
Largest seg size  : 10
Smallest seg size : 0
Number of segments: 29034
CUDA runtime (us) : 1256
CPU runtime (us)  : 11006
[PASSED] checking segs
[PASSED] checking keys
[PASSED] checking vals
```

## Related Projects and References
1. Green, Oded, Robert McColl, and David A. Bader. "GPU merge path: a GPU merging algorithm." Proceedings of the 26th ACM international conference on Supercomputing. 2012. [[link]](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.cs.ucdavis.edu/~amenta/f15/GPUmp.pdf&ved=2ahUKEwiAq8fG7tmIAxXxhYkEHTskHp0QFnoECBUQAQ&usg=AOvVaw2NQmsIy6UAamQE-VWqDUkG)
2. Hou, Kaixi, et al. "Fast segmented sort on gpus." Proceedings of the International Conference on Supercomputing. 2017. [[link]](https://dl.acm.org/doi/10.1145/3079079.3079105)
3. Ji, Haonan, et al. "Segmented merge: A new primitive for parallel sparse matrix computations." International Journal of Parallel Programming 49 (2021): 732-744. [[link]](https://www.ssslab.cn/assets/papers/2021-ji-segmerge.pdf)
4. Zachariadis, Orestis, et al. "Accelerating sparse matrix–matrix multiplication with GPU Tensor Cores." Computers & Electrical Engineering 88 (2020): 106848. [[link]](https://arxiv.org/abs/2009.14600)
5. Chang, Chih-Chun, Boyang Zhang, and Tsung-Wei Huang. "GSAP: A GPU-Accelerated Stochastic Graph Partitioner." Proceedings of the 53rd International Conference on Parallel Processing. 2024. [[link]](https://tsung-wei-huang.github.io/papers/2024-ICPP-GSAP.pdf)

## Useful Parallel Libraries
1. Thrust: The C++ Parallel Algorithms Library [[link]](https://nvidia.github.io/cccl/thrust/)
2. ModernGPU: Patterns and behaviors for GPU computing [[link]](https://github.com/moderngpu/moderngpu)
3. NVIDIA CUB [[link]](https://docs.nvidia.com/cuda/cub/index.html)
4. CUSP : A C++ Templated Sparse Matrix Library [[link]](https://github.com/cusplibrary/cusplibrary)

## TODO
- [ ] Enable to accept different values of A and B
- [ ] Enable to accept different segment sizes of A and B
- [ ] Efficient merging algorithms
