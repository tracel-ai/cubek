//! Complex FFT used internally by the large-`n_fft` real-FFT paths.
//!
//! Two flavours live here:
//!
//! * A shared-memory kernel (`cfft_kernel`) for any size up to the device's
//!   per-cube shared-memory budget (see [`max_shared_fft_n`]). Structurally
//!   identical to `rfft_kernel` / `irfft_kernel` except the imaginary input
//!   is read rather than zeroed and all `N` bins are written out (no
//!   Hermitian truncation).
//! * A four-step Cooley-Tukey orchestrator for larger `N` that factors
//!   `N = N1 * N2` with both factors inside that budget. Each sub-FFT
//!   reuses the shared-memory butterfly via a dedicated "strided / twiddled"
//!   radix kernel (`cfft_four_step_radix_kernel`). A single transpose kernel
//!   converts the (N1, N2) internal layout to natural linear bin order on
//!   the way out.
//!
//! The public API of this module is the single
//! [`cfft_launch_any_size`] function which picks the right path.
//!
//! The caller is responsible for allocating any scratch buffers; see
//! `rfft_large` for the allocator shape contract.

use core::f32::consts::PI;

use cubecl::prelude::*;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::{
    fft::{
        FftMode,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
        limits::{max_shared_fft_n, max_units_per_cube},
    },
    layout::BatchSignalLayout,
};

pub(crate) struct CfftBindings<R: Runtime> {
    pub(crate) input_re: TensorBinding<R>,
    pub(crate) input_im: TensorBinding<R>,
    pub(crate) output_re: TensorBinding<R>,
    pub(crate) output_im: TensorBinding<R>,
}

#[derive(Clone, Copy)]
struct CfftPlan {
    dim: usize,
    count: usize,
    n_fft: usize,
    fft_mode: FftMode,
}

/// Factor `n_fft = N1 * N2` for the four-step FFT. Both factors are powers
/// of two, both `<= max_shared_n_fft`, and the split is as balanced as
/// possible.
pub(crate) fn factor_four_step(n_fft: usize, max_shared_n_fft: usize) -> (usize, usize) {
    assert!(
        n_fft.is_power_of_two(),
        "four-step needs power-of-two n_fft"
    );
    let log2_n = n_fft.trailing_zeros() as usize;
    let max_log2 = max_shared_n_fft.trailing_zeros() as usize;
    // Balanced split, then push each factor up to the shared-mem cap if the
    // other factor would otherwise exceed it.
    let log2_n1 = log2_n / 2;
    let log2_n2 = log2_n - log2_n1;
    let (log2_n1, log2_n2) = if log2_n2 > max_log2 {
        (log2_n - max_log2, max_log2)
    } else {
        (log2_n1, log2_n2)
    };
    assert!(
        log2_n1 <= max_log2 && log2_n2 <= max_log2,
        "four-step cannot handle n_fft = {n_fft} with max shared-mem n = {max_shared_n_fft}",
    );
    (1 << log2_n1, 1 << log2_n2)
}

/// Entry point: complex FFT of `input` into `output`, along `dim`.
/// `input` and `output` must be contiguous and have identical shape. The
/// caller may safely pass the same buffer for input and output (aliasing is
/// allowed for the small path; the large path does its own scratch
/// management).
pub(crate) fn cfft_launch_any_size<R: Runtime>(
    client: &ComputeClient<R>,
    bindings: CfftBindings<R>,
    dim: usize,
    dtype: StorageType,
    fft_mode: FftMode,
) -> Result<(), LaunchError> {
    let n_fft = bindings.input_re.shape[dim];
    assert!(n_fft.is_power_of_two(), "cfft needs power-of-two n_fft");
    assert!(n_fft >= 2);
    let count: usize = bindings
        .input_re
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();
    if count == 0 {
        return Ok(());
    }
    let plan = CfftPlan {
        dim,
        count,
        n_fft,
        fft_mode,
    };

    if n_fft <= max_shared_fft_n(client) {
        cfft_shared_launch::<R>(client, bindings, plan)
    } else {
        cfft_four_step_launch::<R>(client, bindings, dtype, plan)
    }
}

fn cfft_shared_launch<R: Runtime>(
    client: &ComputeClient<R>,
    bindings: CfftBindings<R>,
    plan: CfftPlan,
) -> Result<(), LaunchError> {
    let log2_n = plan.n_fft.trailing_zeros() as usize;
    let threads_per_cube = (plan.n_fft / 2).clamp(1, max_units_per_cube(client));
    let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
    let cube_count =
        cubecl::calculate_cube_count_elemwise(client, plan.count, CubeDim::new_single());

    cfft_shared_kernel::launch::<f32, R>(
        client,
        cube_count,
        cube_dim,
        bindings.input_re.into_tensor_arg(),
        bindings.input_im.into_tensor_arg(),
        bindings.output_re.into_tensor_arg(),
        bindings.output_im.into_tensor_arg(),
        plan.count as u32,
        plan.n_fft,
        log2_n,
        threads_per_cube,
        plan.dim,
        plan.fft_mode,
    );
    Ok(())
}

#[cube(launch)]
fn cfft_shared_kernel<F: Float>(
    input_re: &Tensor<F>,
    input_im: &Tensor<F>,
    output_re: &mut Tensor<F>,
    output_im: &mut Tensor<F>,
    num_windows: u32,
    #[comptime] n_fft: usize,
    #[comptime] log2_n: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
    #[comptime] fft_mode: FftMode,
) {
    let window_index = CUBE_POS;
    if (window_index as u32) >= num_windows {
        terminate!();
    }

    let input_re_view = input_re.view(BatchSignalLayout::new(input_re, window_index, dim));
    let input_im_view = input_im.view(BatchSignalLayout::new(input_im, window_index, dim));
    let output_re_view = output_re.view_mut(BatchSignalLayout::new(&*output_re, window_index, dim));
    let output_im_view = output_im.view_mut(BatchSignalLayout::new(&*output_im, window_index, dim));

    let mut shared_re = SharedMemory::<F>::new(n_fft);
    let mut shared_im = SharedMemory::<F>::new(n_fft);

    let mut i = UNIT_POS as usize;
    while i < n_fft {
        let j = bit_reverse(i, log2_n);
        shared_re[j] = input_re_view.read_checked(i);
        shared_im[j] = input_im_view.read_checked(i);
        i += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n_fft,
        log2_n,
        threads_per_cube,
        fft_mode,
    );

    let mut k = UNIT_POS as usize;
    while k < n_fft {
        output_re_view.write_checked(k, shared_re[k]);
        output_im_view.write_checked(k, shared_im[k]);
        k += threads_per_cube;
    }
}

// --- Four-step path ----------------------------------------------------

/// Four-step complex FFT for `n_fft > max_shared_fft_n(client)`.
///
/// Layout convention: each window's `n_fft` axis is viewed as
/// `(N1, N2)` row-major with the flat index `n = n1 * N2 + n2`. After the
/// four-step pipeline the output has `X[k1 + k2 * N1]` at flat index
/// `k2 * N1 + k1` (natural linear order over `k`). Caller's `output_re` /
/// `output_im` tensors receive this natural order.
fn cfft_four_step_launch<R: Runtime>(
    client: &ComputeClient<R>,
    bindings: CfftBindings<R>,
    dtype: StorageType,
    plan: CfftPlan,
) -> Result<(), LaunchError> {
    let max_n = max_shared_fft_n(client);
    let max_units = max_units_per_cube(client);
    let (n1, n2) = factor_four_step(plan.n_fft, max_n);

    // Scratch buffer, same shape as input. Two passes ping-pong through
    // scratch and output; the transpose at the end lands in `output`.
    let scratch_shape: Vec<usize> = bindings.input_re.shape.to_vec();
    let elems: usize = scratch_shape.iter().product();
    let scratch_re = TensorHandle::<R>::new_contiguous(
        scratch_shape.clone(),
        client.empty(elems * dtype.size()),
        dtype,
    );
    let scratch_im = TensorHandle::<R>::new_contiguous(
        scratch_shape.clone(),
        client.empty(elems * dtype.size()),
        dtype,
    );

    // Step 1: strided FFT_{N1} along the n1 axis of (N1, N2). One cube per
    // (window, n2). Reads from `input_*`, writes to `scratch_*` with fused
    // twiddle multiplication by W_N^{k1 * n2} for the inter-stage factor.
    {
        let threads_per_cube = (n1 / 2).clamp(1, max_units);
        let log2_n1 = n1.trailing_zeros() as usize;
        let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
        let cube_count =
            cubecl::calculate_cube_count_elemwise(client, plan.count * n2, CubeDim::new_single());

        cfft_four_step_radix1_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            bindings.input_re.into_tensor_arg(),
            bindings.input_im.into_tensor_arg(),
            scratch_re.clone().binding().into_tensor_arg(),
            scratch_im.clone().binding().into_tensor_arg(),
            (plan.count * n2) as u32,
            n1,
            n2,
            log2_n1,
            threads_per_cube,
            plan.dim,
            plan.fft_mode,
        );
    }

    // Step 2: contiguous FFT_{N2} along the n2 axis of (N1, N2). One cube
    // per (window, k1). Reads/writes scratch in place.
    {
        let threads_per_cube = (n2 / 2).clamp(1, max_units);
        let log2_n2 = n2.trailing_zeros() as usize;
        let cube_dim = CubeDim::new_1d(threads_per_cube as u32);
        let cube_count =
            cubecl::calculate_cube_count_elemwise(client, plan.count * n1, CubeDim::new_single());

        cfft_four_step_radix2_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            scratch_re.clone().binding().into_tensor_arg(),
            scratch_im.clone().binding().into_tensor_arg(),
            (plan.count * n1) as u32,
            n1,
            n2,
            log2_n2,
            threads_per_cube,
            plan.dim,
            plan.fft_mode,
        );
    }

    // Step 3: transpose (N1, N2) -> (N2, N1). Writes natural-order output.
    {
        let total = plan.count * plan.n_fft;
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, total, cube_dim);

        cfft_four_step_transpose_kernel::launch::<f32, R>(
            client,
            cube_count,
            cube_dim,
            scratch_re.binding().into_tensor_arg(),
            scratch_im.binding().into_tensor_arg(),
            bindings.output_re.into_tensor_arg(),
            bindings.output_im.into_tensor_arg(),
            total as u32,
            n1,
            n2,
            plan.dim,
        );
    }

    Ok(())
}

/// First four-step pass: FFT_{N1} along n1 (strided by N2 in the packed
/// `n_fft` axis), with a fused W_N^{k1 * n2} twiddle on the way out.
///
/// Grid: `count * N2` cubes. `CUBE_POS = window * N2 + n2`.
#[cube(launch)]
fn cfft_four_step_radix1_kernel<F: Float>(
    input_re: &Tensor<F>,
    input_im: &Tensor<F>,
    scratch_re: &mut Tensor<F>,
    scratch_im: &mut Tensor<F>,
    num_cubes: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] log2_n1: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
    #[comptime] fft_mode: FftMode,
) {
    let cube_pos = CUBE_POS;
    if cube_pos >= num_cubes as usize {
        terminate!();
    }

    let window = cube_pos / n2;
    let n2_idx = cube_pos - window * n2;
    let input_re_view = input_re.view(BatchSignalLayout::new(input_re, window, dim));
    let input_im_view = input_im.view(BatchSignalLayout::new(input_im, window, dim));
    let scratch_re_view = scratch_re.view_mut(BatchSignalLayout::new(&*scratch_re, window, dim));
    let scratch_im_view = scratch_im.view_mut(BatchSignalLayout::new(&*scratch_im, window, dim));

    let mut shared_re = SharedMemory::<F>::new(n1);
    let mut shared_im = SharedMemory::<F>::new(n1);

    // Load x[window, n1, n2] at bit-reversed destinations so the butterfly
    // can run directly without a pre-permute pass.
    let mut i = UNIT_POS as usize;
    while i < n1 {
        let j = bit_reverse(i, log2_n1);
        let flat = i * n2 + n2_idx;
        shared_re[j] = input_re_view.read_checked(flat);
        shared_im[j] = input_im_view.read_checked(flat);
        i += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n1,
        log2_n1,
        threads_per_cube,
        fft_mode,
    );

    // Post-twiddle and strided write. W_N^{k1 * n2} with the total-N phase
    // factor is the Cooley-Tukey tying factor between the two sub-FFTs.
    let sign = F::new(fft_mode.sign());
    let n_total = comptime![n1 * n2];
    let two_pi = F::new(2.0 * PI);

    let mut k1 = UNIT_POS as usize;
    while k1 < n1 {
        let theta = sign * two_pi * F::cast_from(k1 * n2_idx) / F::cast_from(n_total);
        let w_re = theta.cos();
        let w_im = theta.sin();
        let ar = shared_re[k1];
        let ai = shared_im[k1];
        let flat = k1 * n2 + n2_idx;
        scratch_re_view.write_checked(flat, w_re * ar - w_im * ai);
        scratch_im_view.write_checked(flat, w_re * ai + w_im * ar);
        k1 += threads_per_cube;
    }
}

/// Second four-step pass: FFT_{N2} along n2 (contiguous), in place.
///
/// Grid: `count * N1` cubes. `CUBE_POS = window * N1 + k1`.
#[cube(launch)]
fn cfft_four_step_radix2_kernel<F: Float>(
    scratch_re: &mut Tensor<F>,
    scratch_im: &mut Tensor<F>,
    num_cubes: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] log2_n2: usize,
    #[comptime] threads_per_cube: usize,
    #[comptime] dim: usize,
    #[comptime] fft_mode: FftMode,
) {
    let cube_pos = CUBE_POS;
    if cube_pos >= num_cubes as usize {
        terminate!();
    }

    let window = cube_pos / n1;
    let k1 = cube_pos - window * n1;
    let row_base = k1 * n2;
    let scratch_re_view = scratch_re.view_mut(BatchSignalLayout::new(&*scratch_re, window, dim));
    let scratch_im_view = scratch_im.view_mut(BatchSignalLayout::new(&*scratch_im, window, dim));

    let mut shared_re = SharedMemory::<F>::new(n2);
    let mut shared_im = SharedMemory::<F>::new(n2);

    let mut i = UNIT_POS as usize;
    while i < n2 {
        let j = bit_reverse(i, log2_n2);
        shared_re[j] = scratch_re_view.read_checked(row_base + i);
        shared_im[j] = scratch_im_view.read_checked(row_base + i);
        i += threads_per_cube;
    }
    sync_cube();

    fft_butterfly_parallel::<F>(
        &mut shared_re,
        &mut shared_im,
        n2,
        log2_n2,
        threads_per_cube,
        fft_mode,
    );

    let mut k2 = UNIT_POS as usize;
    while k2 < n2 {
        scratch_re_view.write_checked(row_base + k2, shared_re[k2]);
        scratch_im_view.write_checked(row_base + k2, shared_im[k2]);
        k2 += threads_per_cube;
    }
}

/// Transpose (N1, N2) -> (N2, N1) in each selected-axis window.
/// Converts four-step output `X'[k1, k2]` at flat `k1*N2 + k2` into natural
/// linear order `X[k]` at flat `k = k1 + k2*N1` (= `k2*N1 + k1`).
///
/// One thread per output element.
#[cube(launch)]
fn cfft_four_step_transpose_kernel<F: Float>(
    scratch_re: &Tensor<F>,
    scratch_im: &Tensor<F>,
    output_re: &mut Tensor<F>,
    output_im: &mut Tensor<F>,
    total: u32,
    #[comptime] n1: usize,
    #[comptime] n2: usize,
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= total as usize {
        terminate!();
    }

    let m = comptime![n1 * n2];
    let pos_u = pos;
    let inner = pos_u % m;
    let window = pos_u / m;
    let scratch_re_view = scratch_re.view(BatchSignalLayout::new(scratch_re, window, dim));
    let scratch_im_view = scratch_im.view(BatchSignalLayout::new(scratch_im, window, dim));
    let output_re_view = output_re.view_mut(BatchSignalLayout::new(&*output_re, window, dim));
    let output_im_view = output_im.view_mut(BatchSignalLayout::new(&*output_im, window, dim));
    // pos's inner index is the destination linear index k = k1 + k2 * N1.
    let k2 = inner / n1;
    let k1 = inner - k2 * n1;
    let src = k1 * n2 + k2;

    output_re_view.write_checked(inner, scratch_re_view.read_checked(src));
    output_im_view.write_checked(inner, scratch_im_view.read_checked(src));
}
