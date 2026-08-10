//! Convolution as a client of the tile DSL: a gather-reduce whose reduce axes are *abstract*.
//!
//! A convolution's window axes are not physical axes of the input. They address the same physical
//! axis its output axes address, at their own coefficient: `Ih = Oh*stride + Rh*dilation`. So the
//! input tile spans more logical axes than its buffer has physical ones, consecutive tiles overlap
//! by the receptive field,
//! and one input element is read by several output positions. That mapping is the operand's
//! [`Projection`]; everything else (the space, the walk, `Tile::at`, the leaf) is the same
//! machinery matmul runs on.
#![allow(non_snake_case)]

use cubecl::{
    Runtime, TestRuntime,
    prelude::*,
    zspace::{Shape, shape},
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};

use cubek_tile::*;

// Output positions, output channels, window taps, input channels. `OW`/`RW` are the second
// spatial pair the 2-D case adds.
const OH: Axis = Axis(0);
const CO: Axis = Axis(1);
const RH: Axis = Axis(2);
const CI: Axis = Axis(3);
const OW: Axis = Axis(4);
const RW: Axis = Axis(5);

/// The same body matmul runs: the operands' spaces say what is contracted, their projections say
/// how they are addressed, and `mma` does the rest.
#[cube(launch)]
fn conv_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: StorageType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

/// [`conv_kernel`] serving the input in two-wide lines. The gathered operand lines along its
/// fastest contracted axis, which is the one the leaf splits into a line index and a lane, so the
/// width has to be a real one somewhere to exercise that fold at all.
#[cube(launch)]
fn conv_kernel_lined<E: Numeric>(
    input: &TileArg<'_, E, Const<2>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: StorageType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

/// Small integers, so the accumulation is exact in `f32` and the reference can be compared for
/// equality rather than closeness.
fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    in_shape: Shape,
    w_shape: Shape,
    out_shape: Shape,
    in_spec: TileSpec,
    w_axes: &[Axis],
    out_spec: TileSpec,
    space: Space,
    in_v: usize,
) -> (HostData, Vec<f32>, Vec<f32>) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::as_type_native_unchecked().storage_type();

    let in_data = ramp(in_shape.num_elements(), 7);
    let w_data = ramp(w_shape.num_elements(), 5);

    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // Every binding stays scalar-unit: the kernel's element type carries the width, and `Tile::of`
    // is what converts the shape and strides to lines.
    let in_binding = in_handle.binding();
    // `#[cube(launch)]` mints its own element type per kernel, so the two arms cannot share
    // pre-built arguments; only the bindings are common.
    let w_binding = w_handle.binding();
    let out_binding = out_handle.clone().binding();
    match in_v {
        1 => conv_kernel::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            TileArgLaunch::new(in_binding.into_tensor_arg(), in_spec),
            TileArgLaunch::new(w_binding.into_tensor_arg(), TileSpec::direct(w_axes)),
            TileArgLaunch::new(out_binding.into_tensor_arg(), out_spec),
            space,
            f32_ty,
        ),
        2 => conv_kernel_lined::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            TileArgLaunch::new(in_binding.into_tensor_arg(), in_spec),
            TileArgLaunch::new(w_binding.into_tensor_arg(), TileSpec::direct(w_axes)),
            TileArgLaunch::new(out_binding.into_tensor_arg(), out_spec),
            space,
            f32_ty,
        ),
        other => panic!("conv: no kernel at input width {other}"),
    }

    (
        HostData::from_tensor_handle(&client, out_handle, HostDataType::F32),
        in_data,
        w_data,
    )
}

// ---- 1-D -------------------------------------------------------------------

/// `out[o, co] = Σ_{r, ci} w[r, ci, co] · in[o·stride + r·dilation, ci]`.
struct Conv1d {
    oh: usize,
    co: usize,
    rh: usize,
    ci: usize,
    stride: usize,
    dilation: usize,
}

impl Conv1d {
    /// The input length a valid (unpadded) convolution consumes: the last output position's last
    /// tap, plus one.
    fn in_len(&self) -> usize {
        (self.oh - 1) * self.stride + (self.rh - 1) * self.dilation + 1
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.oh * self.co];
        for o in 0..self.oh {
            for c in 0..self.co {
                let mut acc = 0.0f32;
                for r in 0..self.rh {
                    for i in 0..self.ci {
                        let x = input[(o * self.stride + r * self.dilation) * self.ci + i];
                        let w = weight[(r * self.ci + i) * self.co + c];
                        acc += x * w;
                    }
                }
                out[o * self.co + c] = acc;
            }
        }
        out
    }

    fn check(&self, tile_oh: usize, tile_co: usize) {
        self.check_at(tile_oh, tile_co, 1, false, Schedule::Direct)
    }

    /// `check` with the input served in `in_v`-wide lines and, when `checked`, both the input and
    /// the output bounds-masked: the two axes of the gather path a plain `check` leaves at their
    /// degenerate values. `schedule` says whether the leaf gathers straight out of gmem
    /// (`Direct`) or out of a compacted stage (`Staged`).
    fn check_at(
        &self,
        tile_oh: usize,
        tile_co: usize,
        in_v: usize,
        checked: bool,
        schedule: Schedule,
    ) {
        let space = Tiling::new()
            .extents(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)])
            .level(WalkOrder::RowMajor, schedule, |l| {
                l.axis(OH, Cut::sequential(tile_oh))
                    .axis(CO, Cut::sequential(tile_co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(CI, Cut::sequential(self.ci))
            })
            .build();

        // The input's one gathered physical axis: the output position at `stride`, the tap at
        // `dilation`.
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, self.stride), (RH, self.dilation)]),
                PhysicalAxisMap::of(CI),
            ],
        ))
        .checked(checked);

        let (got, input, weight) = run(
            shape![self.in_len(), self.ci],
            shape![self.rh, self.ci, self.co],
            shape![self.oh, self.co],
            in_spec,
            &[RH, CI, CO],
            TileSpec::direct(&[OH, CO]).checked(checked),
            space,
            in_v,
        );

        let want = self.reference(&input, &weight);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "conv1d {schedule:?} stride {} dilation {} tile {tile_oh}x{tile_co}: \
                     wrong at ({o}, {c})",
                    self.stride,
                    self.dilation
                );
            }
        }
    }
}

/// Stride 1, dilation 1: consecutive windows overlap by `rh - 1`, the plainest gather there is.
#[test]
fn conv1d_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check(4, 4);
}

/// Stride 2: the output steps twice as fast as the input, so a tile's span is no longer its edge.
#[test]
fn conv1d_strided() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check(4, 4);
}

/// Dilation 2: the taps themselves are spread, so the two coefficients differ.
#[test]
fn conv1d_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 2,
    }
    .check(3, 4);
}

/// Both at once, and a tap count that does not divide anything.
#[test]
fn conv1d_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check(2, 2);
}

/// A 1x1 window is a matmul wearing a projection: the gather degenerates and must still agree.
#[test]
fn conv1d_single_tap_is_a_matmul() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check(4, 4);
}

/// One output tile covering everything: the whole receptive field is one window.
#[test]
fn conv1d_single_tile() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check(6, 4);
}

/// The gathered operand in two-wide lines. Its innermost axis is `CI`, the fastest contracted one,
/// so the leaf splits each reduce step into the line index it reads and the lane it broadcasts:
/// the one arithmetic on the gather path that a width of `1` leaves dead.
#[test]
fn conv1d_vectorized_input() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 2, false, Schedule::Direct);
}

/// Vectorized with stride and dilation both off `1`, so the line fold and the affine advance are
/// exercised at once rather than one at a time.
#[test]
fn conv1d_vectorized_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 2,
        dilation: 2,
    }
    .check_at(3, 2, 2, false, Schedule::Direct);
}

/// An output extent the tile edge does not divide: the last tile's receptive field runs past the
/// input's real length. The masking happens under the projection, so the tap that overhangs must
/// read `0` and the output cell that does not exist must not be written, leaving every valid
/// position equal to the reference.
#[test]
fn conv1d_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, true, Schedule::Direct);
}

/// The same overhang with a stride, where the window runs further past the end: the mask is on the
/// physical axis, so how far the gather reached past it must not matter.
#[test]
fn conv1d_masked_overhang_strided() {
    Conv1d {
        oh: 5,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(2, 4, 1, true, Schedule::Direct);
}

// ---- 2-D -------------------------------------------------------------------

/// `out[oh, ow, co] = Σ_{rh, rw, ci} w[rh, rw, ci, co] · in[oh·sh + rh·dh, ow·sw + rw·dw, ci]`.
/// Two abstract dimensions, each gathering its own input physical axis.
struct Conv2d {
    oh: usize,
    ow: usize,
    co: usize,
    rh: usize,
    rw: usize,
    ci: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
}

impl Conv2d {
    fn in_h(&self) -> usize {
        (self.oh - 1) * self.sh + (self.rh - 1) * self.dh + 1
    }
    fn in_w(&self) -> usize {
        (self.ow - 1) * self.sw + (self.rw - 1) * self.dw + 1
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let (iw, ci, co) = (self.in_w(), self.ci, self.co);
        let mut out = vec![0.0f32; self.oh * self.ow * co];
        for oh in 0..self.oh {
            for ow in 0..self.ow {
                for c in 0..co {
                    let mut acc = 0.0f32;
                    for rh in 0..self.rh {
                        for rw in 0..self.rw {
                            for i in 0..ci {
                                let h = oh * self.sh + rh * self.dh;
                                let w_ = ow * self.sw + rw * self.dw;
                                let x = input[(h * iw + w_) * ci + i];
                                let k = weight[((rh * self.rw + rw) * ci + i) * co + c];
                                acc += x * k;
                            }
                        }
                    }
                    out[(oh * self.ow + ow) * co + c] = acc;
                }
            }
        }
        out
    }

    fn check(&self, tile_oh: usize, tile_ow: usize, tile_co: usize) {
        self.check_at(tile_oh, tile_ow, tile_co, Schedule::Direct)
    }

    /// `check` under `schedule`: `Direct` gathers straight out of gmem, `Staged` compacts the two
    /// gathered physical axes into a dense stage first.
    fn check_at(&self, tile_oh: usize, tile_ow: usize, tile_co: usize, schedule: Schedule) {
        let space = Tiling::new()
            .extents(&[
                (OH, self.oh),
                (OW, self.ow),
                (CO, self.co),
                (RH, self.rh),
                (RW, self.rw),
                (CI, self.ci),
            ])
            .level(WalkOrder::RowMajor, schedule, |l| {
                l.axis(OH, Cut::sequential(tile_oh))
                    .axis(OW, Cut::sequential(tile_ow))
                    .axis(CO, Cut::sequential(tile_co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(RW, Cut::sequential(self.rw))
                    .axis(CI, Cut::sequential(self.ci))
            })
            .build();

        // Two gathered physical axes, one per spatial axis pair; the channel axis rides identity.
        let in_spec = TileSpec::new(Projection::new(
            &[OH, OW, RH, RW, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, self.sh), (RH, self.dh)]),
                PhysicalAxisMap::affine(&[(OW, self.sw), (RW, self.dw)]),
                PhysicalAxisMap::of(CI),
            ],
        ));

        let (got, input, weight) = run(
            shape![self.in_h(), self.in_w(), self.ci],
            shape![self.rh, self.rw, self.ci, self.co],
            shape![self.oh, self.ow, self.co],
            in_spec,
            &[RH, RW, CI, CO],
            TileSpec::direct(&[OH, OW, CO]),
            space,
            1,
        );

        let want = self.reference(&input, &weight);
        for oh in 0..self.oh {
            for ow in 0..self.ow {
                for c in 0..self.co {
                    assert_eq!(
                        got.get_f32(&[oh, ow, c]),
                        want[(oh * self.ow + ow) * self.co + c],
                        "conv2d {schedule:?}: wrong at ({oh}, {ow}, {c})"
                    );
                }
            }
        }
    }
}

#[test]
fn conv2d_dense_window() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 3,
        rw: 3,
        ci: 2,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check(2, 2, 4);
}

/// Different stride and dilation per spatial axis, so the two projected physical axes cannot be
/// confused.
#[test]
fn conv2d_asymmetric_stride_and_dilation() {
    Conv2d {
        oh: 4,
        ow: 3,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 1,
        dw: 2,
    }
    .check(2, 3, 2);
}

/// One tile covering the whole output: every axis's window is the full receptive field.
#[test]
fn conv2d_single_tile() {
    Conv2d {
        oh: 3,
        ow: 3,
        co: 2,
        rh: 2,
        rw: 2,
        ci: 3,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check(3, 3, 2);
}

// ---- staged ----------------------------------------------------------------

// The same convolutions with the level `Staged`. A gathered operand's region is copied into a
// shared-memory tile shaped like the physical *window* it reads (`span(oh, rh) × ci`), compacted
// onto the lattice its stride and dilation reach, so each input element is stored once. The stage
// keeps the operand's own projection, so the leaf gathers out of smem exactly as it gathered out of
// gmem, and every case here must agree with its `Direct` twin.

/// Stride 1: consecutive windows overlap by `rh - 1`, so the window the stage holds
/// (`oh + rh - 1`) is smaller than the `oh × rh` logical cells reading it.
#[test]
fn conv1d_staged_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, false, Schedule::Staged);
}

/// Stride and dilation both off `1`: the fill's per-axis advance and the receptive-field span are
/// what place each staged element, so neither can collapse to the edge.
#[test]
fn conv1d_staged_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check_at(2, 2, 1, false, Schedule::Staged);
}

/// A single tap at unit stride: the window is exactly the output's own extent, so the stage is the
/// smallest it ever gets while the projection stays affine (`Rh` is still a declared axis at
/// coefficient `1`, so the leaf stays on the N-D nest). The answer must not move.
#[test]
fn conv1d_staged_single_tap() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, false, Schedule::Staged);
}

/// One region covering everything: the walk runs once, so the fill is the whole kernel.
#[test]
fn conv1d_staged_single_tile() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(6, 4, 1, false, Schedule::Staged);
}

/// Staged in two-wide lines: the fill moves whole lines, and the innermost axis is the one the
/// projection is required to carry at coefficient `1` for that to be sound.
#[test]
fn conv1d_staged_vectorized_input() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 2, false, Schedule::Staged);
}

/// Vectorized with stride and dilation both off `1`.
#[test]
fn conv1d_staged_vectorized_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 2,
        dilation: 2,
    }
    .check_at(3, 2, 2, false, Schedule::Staged);
}

/// The overhanging last tile: the mask sits under the projection, so a tap past the input's real
/// length must stage a `0` rather than whatever the buffer holds there.
#[test]
fn conv1d_staged_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, true, Schedule::Staged);
}

/// The same overhang with a stride, where the window runs further past the end.
#[test]
fn conv1d_staged_masked_overhang_strided() {
    Conv1d {
        oh: 5,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(2, 4, 1, true, Schedule::Staged);
}

/// A single tap at stride 2 reaches only every second input position, so the stage keeps half of
/// the window it spans and the fill steps by two. One of the three non-unit-step cases, with
/// `conv1d_staged_vectorized_strided_and_dilated` (`gcd(2, 2) = 2`, above) and
/// `conv2d_staged_mixed_steps`.
#[test]
fn conv1d_staged_single_tap_strided() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(4, 4, 1, false, Schedule::Staged);
}

/// The overhang masked *and* the input lined: the fill steps whole lines through a window whose
/// last taps read past the input, so the line fold and the mask have to hold at once.
#[test]
fn conv1d_staged_vectorized_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 2, true, Schedule::Staged);
}

/// Double buffering drives the same fill from two slots on alternating regions, so a gathered
/// source has to re-window per slot rather than once per walk.
#[test]
fn conv1d_staged_double_buffered() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(2, 4, 1, false, Schedule::DoubleBuffered);
}

/// Two gathered physical axes staged at once: the fill decodes one destination coordinate per
/// logical axis, so both affine maps are resolved from the same index.
#[test]
fn conv2d_staged_dense_window() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 3,
        rw: 3,
        ci: 2,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check_at(2, 2, 4, Schedule::Staged);
}

/// Different stride and dilation per spatial axis, so the two staged physical axes cannot be
/// confused for one another.
#[test]
fn conv2d_staged_asymmetric_stride_and_dilation() {
    Conv2d {
        oh: 4,
        ow: 3,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 1,
        dw: 2,
    }
    .check_at(2, 3, 2, Schedule::Staged);
}

/// One compacted physical axis per step: `h` has `gcd(2, 2) = 2`, so its stage keeps every second
/// row and the fill steps through it, while `w` is dense and steps by one. The only case where
/// `StepUp` carries more than one distinct step, so a transposed or broadcast step shows up here.
#[test]
fn conv2d_staged_mixed_steps() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 2,
        dw: 1,
    }
    .check_at(2, 2, 4, Schedule::Staged);
}

// ---- the projected 2-D view ------------------------------------------------

/// Reads a gathered operand through [`Tile::matrix`] and writes every batch matrix out flat, so
/// the host can check the projected layout against the same gather done by hand. The 2-D door for
/// an operand whose logical axes outnumber its buffer's physical ones: the matrix coordinate
/// resolves through the leading axes first, then folds onto the window through the projection.
#[cube(launch)]
fn projected_matrix_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    out: &mut Tensor<f32>,
    #[comptime] space: Space,
    #[comptime] matrices: usize,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
    #[define(E)] _dtype: StorageType,
) {
    let input = input.tile(space);
    let size!(W) = input.vector_size();

    #[unroll]
    for m in 0..matrices {
        let view = input.matrix::<W>(m);
        #[unroll]
        for r in 0..rows {
            #[unroll]
            for c in 0..cols {
                let value = view.read((r as u32, c as u32).runtime());
                out[comptime!((m * rows + r) * cols + c)] = f32::cast_from(value);
            }
        }
    }
}

/// The one 2-D convolution both view tests read, strided and dilated on both spatial axes so no
/// two logical coordinates land on the same input cell by accident.
struct Conv2dViewSetup {
    /// `(oh, ow, rh, rw, ci)`, the logical axes in the space's own order.
    logical: (usize, usize, usize, usize, usize),
    /// `(sh, sw, dh, dw)`, the strides then the dilations.
    steps: (usize, usize, usize, usize),
    in_w: usize,
    space: Space,
    in_spec: TileSpec,
    in_data: Vec<f32>,
    in_handle: cubecl::std::tensor::TensorHandle<TestRuntime>,
}

fn setup_conv2d_view() -> Conv2dViewSetup {
    let (oh, ow, rh, rw, ci) = (3usize, 4usize, 2usize, 3usize, 2usize);
    let (sh, sw, dh, dw) = (2usize, 1usize, 1usize, 2usize);
    let in_h = (oh - 1) * sh + (rh - 1) * dh + 1;
    let in_w = (ow - 1) * sw + (rw - 1) * dw + 1;

    let space = Tiling::new()
        .extents(&[(OH, oh), (OW, ow), (RH, rh), (RW, rw), (CI, ci)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(OH, Cut::sequential(oh))
                .axis(OW, Cut::sequential(ow))
                .axis(RH, Cut::sequential(rh))
                .axis(RW, Cut::sequential(rw))
                .axis(CI, Cut::sequential(ci))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[OH, OW, RH, RW, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, sh), (RH, dh)]),
            PhysicalAxisMap::affine(&[(OW, sw), (RW, dw)]),
            PhysicalAxisMap::of(CI),
        ],
    ));

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let in_data = ramp(in_h * in_w * ci, 7);
    let (in_handle, _) = TestInput::builder(client.clone(), shape![in_h, in_w, ci])
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();

    Conv2dViewSetup {
        logical: (oh, ow, rh, rw, ci),
        steps: (sh, sw, dh, dw),
        in_w,
        space,
        in_spec,
        in_data,
        in_handle,
    }
}

/// The 2-D view over the 2-D convolution's input: logical `[OH, OW, RH, RW, CI]` over the physical
/// `[IH, IW, CI]`, so three leading axes are pinned and the trailing `RW x CI` pair is the matrix.
/// Three pinned axes is what makes this worth running: the unravel's weights are a product of
/// several extents, and reading those off the window instead of the space would silently pick up
/// the receptive field's span (`IH`, `IW`) rather than the logical edges.
#[test]
fn conv2d_projected_matrix_view() {
    let s = setup_conv2d_view();
    let (oh, ow, rh, rw, ci) = s.logical;
    let (sh, sw, dh, dw) = s.steps;
    let (in_w, in_data, space) = (s.in_w, s.in_data, s.space);

    let matrices = oh * ow * rh;
    let (rows, cols) = (rw, ci);

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let out_handle = TestInput::builder(client.clone(), shape![matrices, rows, cols])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    projected_matrix_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(s.in_handle.binding().into_tensor_arg(), s.in_spec),
        out_handle.clone().binding().into_tensor_arg(),
        space,
        matrices,
        rows,
        cols,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for m in 0..matrices {
        // The pinned axes, unraveled the way the layout unravels them: row-major over `[OH, OW, RH]`.
        let (o_h, o_w, r_h) = (m / (ow * rh), (m / rh) % ow, m % rh);
        for r_w in 0..rows {
            for c_i in 0..cols {
                let h = o_h * sh + r_h * dh;
                let w = o_w * sw + r_w * dw;
                assert_eq!(
                    got.get_f32(&[m, r_w, c_i]),
                    in_data[(h * in_w + w) * ci + c_i],
                    "projected matrix {m} ({o_h}, {o_w}, {r_h}) at ({r_w}, {c_i})"
                );
            }
        }
    }
}

/// Reads a gathered operand through [`Tile::fragment_matrix`] and writes the whole matrix out, so
/// the host can check it against the im2col expansion done by hand. This is the face an mma
/// fragment reads: the output positions flattened into the row edge, the taps and channels into
/// the column edge, resolved onto the compact window underneath.
#[cube(launch)]
fn fragment_matrix_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    out: &mut Tensor<f32>,
    #[comptime] space: Space,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
    #[define(E)] _dtype: StorageType,
) {
    let input = input.tile(space);
    let size!(W) = input.vector_size();
    let view = input.fragment_matrix::<E, W, W>(rows, cols);

    #[unroll]
    for r in 0..rows {
        #[unroll]
        for c in 0..cols {
            let value = view.read((r as u32, c as u32).runtime());
            out[comptime!(r * cols + c)] = f32::cast_from(value);
        }
    }
}

/// The 2-D convolution input as one `(OH·OW) x (RH·RW·CI)` matrix: five logical axes flattened
/// into two edges over three physical ones. Neither edge is a single axis, which is the whole
/// point: no pinning of trailing axes produces this face, and it is the one an mma contracts.
#[test]
fn conv2d_fragment_matrix_view() {
    let s = setup_conv2d_view();
    let (oh, ow, rh, rw, ci) = s.logical;
    let (sh, sw, dh, dw) = s.steps;
    let (in_w, in_data, space) = (s.in_w, s.in_data, s.space);

    let (rows, cols) = (oh * ow, rh * rw * ci);

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let out_handle = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    fragment_matrix_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(s.in_handle.binding().into_tensor_arg(), s.in_spec),
        out_handle.clone().binding().into_tensor_arg(),
        space,
        rows,
        cols,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for r in 0..rows {
        let (o_h, o_w) = (r / ow, r % ow);
        for c in 0..cols {
            // The column edge unravels row-major over `[RH, RW, CI]`, as the layout groups it.
            let (r_h, r_w, c_i) = (c / (rw * ci), (c / ci) % rw, c % ci);
            let h = o_h * sh + r_h * dh;
            let w = o_w * sw + r_w * dw;
            assert_eq!(
                got.get_f32(&[r, c]),
                in_data[(h * in_w + w) * ci + c_i],
                "fragment matrix at ({r}, {c}) = ({o_h}, {o_w}, {r_h}, {r_w}, {c_i})"
            );
        }
    }
}

// ---- the manual-mma leaf ---------------------------------------------------

/// The resident promote → zero → mma → drain kernel of the matmul tests, with a *gathered* lhs.
/// The accumulator is sized by the whole contraction (taps times channels), the input stages into
/// its compacted smem window, and the fragment load flattens the tap and channel axes back into
/// the `k` edge as it reads that window.
#[cube(launch)]
fn conv_mma_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: StorageType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    let mut acc = out.promote(&input);
    acc.zero();
    acc.mma(&input, &weight);
    out.copy_from(&acc);
}

#[test]
fn conv1d_mma_leaf() {
    conv1d_mma_leaf_with(MmaIOConfig::manual());
}

#[test]
fn conv1d_mma_leaf_gathered_lhs_ignores_ldmatrix() {
    conv1d_mma_leaf_with(MmaIOConfig {
        lhs_load_method: LoadMethod::LoadMatrix,
        ..MmaIOConfig::manual()
    });
}

fn conv1d_mma_leaf_with(io: MmaIOConfig) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.mma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no manual mma (features.matmul.mma) support".to_string(),
        ))
        .enforce();
        return;
    }

    // `k = rh * ci = 8`, an 8x8x8 instruction, as the matmul mma test uses.
    let (oh, co, rh, ci) = (8usize, 8usize, 2usize, 4usize);
    let (stride, dilation) = (1usize, 1usize);
    let in_len = (oh - 1) * stride + (rh - 1) * dilation + 1;
    let leaf = Leaf::Mma { io };

    let space = Tiling::new()
        .extents(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)])
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axis(OH, Cut::sequential(oh))
                .axis(CO, Cut::sequential(co))
                .axis(RH, Cut::sequential(rh))
                .axis(CI, Cut::sequential(ci))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .leaf(leaf);

    let f32_ty = f32::as_type_native_unchecked().storage_type();
    let in_data = ramp(in_len * ci, 7);
    let w_data = ramp(rh * ci * co, 5);

    let (in_handle, _) = TestInput::builder(client.clone(), shape![in_len, ci])
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), shape![rh, ci, co])
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![oh, co])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    conv_mma_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            w_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[RH, CI, CO]).leaf(leaf),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[OH, CO]).leaf(leaf),
        ),
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for o in 0..oh {
        for c in 0..co {
            let want: f32 = (0..rh)
                .flat_map(|r| (0..ci).map(move |i| (r, i)))
                .map(|(r, i)| {
                    let h = o * stride + r * dilation;
                    in_data[h * ci + i] * w_data[(r * ci + i) * co + c]
                })
                .sum();
            assert_eq!(
                got.get_f32(&[o, c]),
                want,
                "conv1d mma leaf: wrong at ({o}, {c})"
            );
        }
    }
}
