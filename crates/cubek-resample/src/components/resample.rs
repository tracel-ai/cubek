use crate::components::{
    footprint::{Footprint, get_footprint},
    kernel::kernel_weight,
    layout::{Layout, LayoutExpand, NdLayout},
    semiring::{semiring_combine, semiring_identity, semiring_reduce},
};
use crate::definition::Resample;
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

/// Resample kernel.
#[cube(launch_unchecked)]
pub fn resample_kernel<C: Float>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    out_layout: NdLayout,
    in_layout: NdLayout,
    #[comptime] config: Resample,
    #[comptime] spatial_axis: usize,
    #[define(C)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS as usize;
    if index >= output.len() {
        terminate!();
    }

    let out_coord = out_layout.from_linear(index);

    resample_coord::<C>(
        input,
        output,
        &in_layout,
        &out_coord,
        index,
        config,
        spatial_axis,
    );
}

/// Resample a single output coord.
#[cube]
pub fn resample_coord<C: Float>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    in_layout: &NdLayout,
    out_coord: &CoordsDyn,
    index: usize,
    #[comptime] config: Resample,
    #[comptime] spatial_axis: usize,
) {
    let out_pos = out_coord[spatial_axis] as usize;

    let footprint = get_footprint::<C>(config, out_pos);

    let mut acc = semiring_identity::<C>(config.semiring);

    accumulate_taps::<C>(
        input,
        in_layout,
        out_coord,
        footprint,
        &mut acc,
        config,
        spatial_axis,
    );

    output[index] = acc;
}

// Accumulate tap weights to produce a single tap value.
#[cube]
fn accumulate_taps<C: Float>(
    input: &Tensor<C>,
    in_layout: &NdLayout,
    out_coord: &CoordsDyn,
    footprint: Footprint<C>,
    acc: &mut C,
    #[comptime] config: Resample,
    #[comptime] spatial_axis: usize,
) {
    for i in 0..footprint.num_taps {
        let tap_pos = footprint.start_tap + i as isize;

        let x = C::cast_from(i as isize - footprint.radius as isize) - footprint.frac;
        let weight = kernel_weight::<C>(config.kernel, x);

        let mut in_coord = out_coord.clone();
        in_coord[spatial_axis] = tap_pos as u32;

        let (in_idx, is_in_bounds) = in_layout.to_source_pos_checked(in_coord);

        if is_in_bounds {
            let value = input[in_idx];

            let combined = semiring_combine::<C>(config.semiring, value, weight);

            *acc = semiring_reduce::<C>(config.semiring, *acc, combined);
        }
    }
}
