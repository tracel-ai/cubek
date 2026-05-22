//! Per-device limits used when launching the FFT kernels.

use cubecl::prelude::*;

/// Largest power-of-two `n_fft` such that a shared-memory radix-2 butterfly
/// over `f32` fits in this device's per-cube shared memory.
///
/// Every shared-memory FFT kernel in this crate allocates two
/// `SharedMemory<f32>::new(n_fft)` buffers (one for the real part, one for
/// the imaginary part), so the byte budget is
/// `2 * size_of::<f32>() * n_fft <= hardware.max_shared_memory_size`.
/// We floor to a power of two because the butterfly requires it.
pub(crate) fn max_shared_fft_n<R: Runtime>(client: &ComputeClient<R>) -> usize {
    let max_smem = client.properties().hardware.max_shared_memory_size;
    let max_elems = max_smem / (2 * core::mem::size_of::<f32>());
    floor_power_of_two(max_elems)
}

/// Hardware-reported maximum number of units (threads) per cube.
pub(crate) fn max_units_per_cube<R: Runtime>(client: &ComputeClient<R>) -> usize {
    client.properties().hardware.max_units_per_cube as usize
}

fn floor_power_of_two(n: usize) -> usize {
    assert!(n > 0, "device reports zero shared memory / units");
    if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two() >> 1
    }
}
