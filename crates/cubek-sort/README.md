# CubeK Sort

Implements a portal radix sort that is hardware agnostic, and supports key/value pairs.

## Implementation

The implementation is based on the device radix sort from [b0nes164](https://github.com/b0nes164/GPUSorting). While the single pass decoupled look back would be faster, it requires forward progress guarantees, which is not guaranteed by all runtimes. This could be added in future versions as variant of the kernels.

## Other features

- Radix sorting is a _stable_ sorting algorithm, which means values with the same key are preserved in their original order.
- Supports sorting of key/value pairs.
- Supports sorting floating point values as well as integers.

## Resources

https://gpuopen.com/learn/boosting_gpu_radix_sort/
https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
https://linebender.org/wiki/gpu/sorting/
https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
