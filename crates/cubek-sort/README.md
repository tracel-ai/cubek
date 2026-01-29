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

# Claude: Plan

**Note: This section will be removed before sending a PR request.**

We are trying to implement Radix sorting as a CubeCL algorithm. This will be a PR to CubeK at some point, tracel's official collection of CubeCL algorithms. 

I have stubbed out a crate in cubek-sort. In here, we are going to implement the device radix sort from the b0nes164 library. This is their simpler, more portable version, which will establish a great baseline.

## Testing

We should generate comprehensive tests comparing against CPU rust standard library sorting algorithms. This should at least cover cases such as:
- Arrays with a single element
- Arrays with duplicate elements
- Arrays with negative elements
- Arrays with floating point elements
- Arrays with i32/u32/i16/u16 elements/keys
- Arrays with a massive size
- Arrays with edge case sizes such as multiples of warp sizes, threadgroup sizes, etc.
- Arrays with key/value pairs with duplicate keys, testing for sorting stablity.

## Benchmarks

We should generate benchmarks akin to the [b0nes164](https://github.com/b0nes164/GPUSorting) library's benchmarks. We can use normal CubeCL synchronization to measure throughput (measured in GB/s).

## Implementation notes

CubeCLs syntax has been undergoing rapid changes, so you might find that some code no longer compiles that you thought was correct. The general direction of development is to _more closely_ match normal rust semantics.

We should make use of subgroup operations where applicable as they are now portable even on WebGPU.

The kernels should make good use of comptime, to conditionally compile parts of the code we might not need, such as handling floating point data (which requires a bit flip) and sorting key/value pairs vs. sorting keys only.

Also note that unlike "traditional" GPU programming CubeCL fully supports higher level concepts such as functions, structs, and even traits. Please see the other crates in CubeK for inspiration. This means that porting the GPU sort code line-by-line might not be optimal, we might have significant oppurtunities for organisation using more advanced programming structures.
